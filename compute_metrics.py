import torch
import numpy as np
from typing import Tuple, Any, Optional
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from input_generation import (
    InputGenerator,
    DiscreteGenerator,
    CircularGenerator,
    ModulatedCircularGenerator,
)
from params import SimulationParameters
from utils import (
    circular_discrepancy,
    circular_kde,
    circular_smooth_values,
    circular_smooth_median,
    circular_smooth_huber,
    power_law_regression,
)
from scipy.interpolate import interp1d


def renormalise(
    curve: Float[torch.Tensor, "num_latents"],
) -> Float[torch.Tensor, "num_latents"]:
    """Renormalise a curve to sum to 1."""
    return curve / curve.sum()


@jaxtyped(typechecker=typechecked)
def compute_firing_rates(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    u: Float[torch.Tensor, "N_E num_stimuli"],
    parameters: SimulationParameters,
    v_init: Optional[Float[torch.Tensor, "N_I num_stimuli"]] = None,
) -> tuple[
    Float[torch.Tensor, "N_I num_stimuli"], Float[torch.Tensor, "N_I num_stimuli"]
]:
    """
    Compute the firing rates of a neural network given the input and weight matrices.

    Args:
        W (torch.Tensor): The feedforward weight matrix.
        M (torch.Tensor): The recurrent weight matrix.
        u (torch.Tensor): The input to the network.
        activation_function (Callable): The non-linearity of the neural network.

    Returns:
        torch.Tensor: The firing rates of the network.
    """
    # Unpack from parameters
    dt = parameters.dt
    tau_v = parameters.tau_v
    activation_function = parameters.activation_function
    threshold = parameters.rate_computation_threshold
    max_iter = parameters.rate_computation_iterations

    # Initialise the input h and the voltage v
    h = W @ u
    if v_init is not None:
        v = v_init
    else:
        v = torch.zeros_like(h)
    r = activation_function(v)
    r_dot = float("inf")
    counter = 0
    # Iterate until the rates have converged
    while r_dot > threshold and counter < max_iter:
        v = v + (dt / tau_v) * (h - M @ r - v)
        r_new = activation_function(v)
        r_dot = torch.mean(torch.abs(r_new - r)) / dt
        r = r_new
        counter += 1

    # if counter == max_iter:
    #     wandb.alert(
    #         title="Firing rate computation did not converge",
    #         text="The firing rate computation did not converge within the maximum number of iterations",
    #         level=wandb.AlertLevel.WARN,
    #     )

    return r, v


@jaxtyped(typechecker=typechecked)
def compute_firing_rates_momentum(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    u: Float[torch.Tensor, "N_E num_stimuli"],
    parameters: SimulationParameters,
    v_init: Optional[Float[torch.Tensor, "N_I num_stimuli"]] = None,
    beta: float = 0.9,  # Momentum factor
) -> tuple[
    Float[torch.Tensor, "N_I num_stimuli"], Float[torch.Tensor, "N_I num_stimuli"]
]:
    """
    Compute the firing rates of a neural network given the input and weight matrices.
    Uses momentum acceleration for faster convergence.

    Args:
        W (torch.Tensor): The feedforward weight matrix.
        M (torch.Tensor): The recurrent weight matrix.
        u (torch.Tensor): The input to the network.
        parameters (SimulationParameters): Parameters for the simulation.
        v_init (Optional[torch.Tensor]): Initial voltage values.
        beta (float): Momentum coefficient (0 = no momentum, 0.9 = high momentum).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The firing rates and voltages of the network.
    """
    # Unpack from parameters
    dt = parameters.dt
    tau_v = parameters.tau_v
    activation_function = parameters.activation_function
    threshold = parameters.rate_computation_threshold
    max_iter = parameters.rate_computation_iterations

    # Initialise the input h and the voltage v
    h = W @ u
    if v_init is not None:
        v = v_init
    else:
        v = torch.zeros_like(h)
    r = activation_function(v)

    # Initialize momentum variables
    delta_r_prev = torch.zeros_like(r)

    r_dot = float("inf")
    counter = 0
    # Iterate until the rates have converged
    while r_dot > threshold and counter < max_iter:
        v = v + (dt / tau_v) * (h - M @ r - v)
        r_new = activation_function(v)

        # Calculate current change
        delta_r = r_new - r

        # Apply momentum
        r_new = r_new + beta * delta_r_prev

        # Store current change for next iteration
        delta_r_prev = delta_r

        # Check convergence (using original delta for convergence check)
        r_dot = torch.mean(torch.abs(delta_r)) / dt

        r = r_new
        counter += 1

    return r, v


@jaxtyped(typechecker=typechecked)
def compute_firing_rates_newton(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    u: Float[torch.Tensor, "N_E num_stimuli"],
    parameters: SimulationParameters,
    r_init: Optional[Float[torch.Tensor, "N_I num_stimuli"]] = None,
) -> tuple[
    Float[torch.Tensor, "N_I num_stimuli"], Float[torch.Tensor, "N_I num_stimuli"]
]:
    # Unpack parameters
    activation_function = parameters.activation_function
    threshold = parameters.rate_computation_threshold
    max_iter = parameters.rate_computation_iterations
    N_I = parameters.N_I
    k_I = parameters.k_I
    damping = 1.0

    # Initialize
    h = W @ u
    r = r_init if r_init is not None else torch.zeros_like(h)

    # Pre-allocate tensors for computations
    r_expected = torch.empty_like(r)
    residual = torch.empty_like(r)

    for i in range(max_iter):
        v_current = h - M @ r  # [N_I num_stimuli]
        r_expected = activation_function(v_current)

        # Compute residual: r - f(h - M @ r)
        residual = r - r_expected

        # Check convergence
        if torch.max(torch.abs(residual)) <= threshold:
            break

        derivatives = activation_function.derivative(v_current)  # [N_I num_stimuli]

        preconditioner = 1.0 / (
            1.0 + derivatives * torch.diag(M).unsqueeze(1)
        )  # [N_I num_stimuli]
        try:
            delta_r = -residual * preconditioner  # [N_I num_stimuli]
            r = r + delta_r  # [N_I num_stimuli]
            r.clamp_(min=0)  # Ensure non-negativity
        except:
            # Fallback to a simple update if numerical issues
            r = r_expected

    # Final voltage computation
    v = h - M @ r

    return r, v


@jaxtyped(typechecker=typechecked)
def compute_tuning_curve_widths(
    rates: Float[torch.Tensor, "N_I num_stimuli"],
    stimulus_space: Float[torch.Tensor, "num_stimuli"],
) -> Float[torch.Tensor, "N_I"]:
    """
    Compute the width of tuning curves using circular standard deviation.

    Args:
        rates: Tuning curve responses with shape [N_I, num_stimuli]
               where rates[i, j] is the response of neuron i to stimulus j

    Returns:
        widths: Standard deviation-based width for each neuron [N_I]
    """
    # Normalize each tuning curve to create probability distributions
    # Subtract minimum and ensure non-negative
    rates_shifted = rates - rates.min(dim=1, keepdim=True)[0]  # [N_I, num_stimuli]

    # Handle edge case where tuning curve is constant (extremely unlikely)
    curve_sums = rates_shifted.sum(dim=1, keepdim=True)  # [N_I, 1]
    curve_sums = torch.where(curve_sums == 0, torch.ones_like(curve_sums), curve_sums)

    # Normalize to probability distribution
    p = rates_shifted / curve_sums  # [N_I, num_stimuli]

    # Compute circular mean resultant vector for each neuron
    # R = |sum(p[i] * exp(i * θ[i]))|
    cos_component = torch.sum(p * torch.cos(stimulus_space), dim=1)  # [N_I]
    sin_component = torch.sum(p * torch.sin(stimulus_space), dim=1)  # [N_I]

    # Magnitude of resultant vector
    R = torch.sqrt(cos_component**2 + sin_component**2)  # [N_I]

    # Circular variance: σ² = -2 * log(R)
    epsilon = 1e-8
    R_clamped = torch.clamp(R, min=epsilon, max=1.0)
    circular_variance = -2 * torch.log(R_clamped)  # [N_I]

    # Circular standard deviation
    circular_std = torch.sqrt(circular_variance)  # [N_I]

    # Convert to width measure (you can adjust this multiplier as needed)
    # Factor of 2 gives full width, similar to 2*sigma for Gaussian
    widths = 2 * circular_std

    return widths


def compute_population_response_metrics(
    rates: Float[torch.Tensor, "N_I num_latents"],
    input_generator: ModulatedCircularGenerator,
    parameters: SimulationParameters,
) -> dict[str, Any]:
    r"""For consistency, everything here is going to be in torch. Then we'll convert when we return."""

    N_E = parameters.N_E
    N_I = parameters.N_I

    stimuli_probabilities = input_generator.stimuli_probabilities  # [num_latents]
    # If the input generator has latent_stimuli_probabilities, use those instead
    if hasattr(input_generator, "latent_stimuli_probabilities"):
        stimuli_probabilities = input_generator.latent_stimuli_probabilities

    stimuli_modulation_curve = input_generator.modulation_curve  # [num_latents]

    argmax_rates = rates.argmax(
        axis=1
    )  # [N_I] (indices of max response in num_latents)
    stimulus_space = input_generator.stimuli_positions.squeeze()  # [num_latents]
    argmax_stimuli = stimulus_space[
        argmax_rates
    ].flatten()  # [N_I] (stimuli of max response)
    max_rates, _ = rates.max(axis=1)  # [N_I]

    total_rate = rates.sum(axis=0)  # [num_latents]

    bw_multiplier = N_I ** (-0.2)
    max_rate_range = max_rates.max() - max_rates.min()

    gains = circular_smooth_huber(
        argmax_stimuli,
        max_rates,
        stimulus_space,
        bw=bw_multiplier * 0.4,
        delta=0.25 * max_rate_range.item(),
    )  # [num_latents]

    # Find the width at half height of the tuning curve
    tuning_curve_widths = compute_tuning_curve_widths(rates, stimulus_space)  # [N_I]
    width_range = tuning_curve_widths.max() - tuning_curve_widths.min()

    widths = circular_smooth_huber(
        argmax_stimuli,
        tuning_curve_widths,
        stimulus_space,
        bw=bw_multiplier * 0.4,
        delta=0.25 * width_range.item(),
    )

    density = circular_kde(
        argmax_stimuli, stimulus_space, bw=bw_multiplier * 0.2
    )  # [num_latents]

    population_response_metrics = {
        "c": torch.ones_like(total_rate),
        "i": stimuli_modulation_curve,
        "i^2": stimuli_modulation_curve**2,
        "p": stimuli_probabilities,
        "p^{-1/alpha}": stimuli_probabilities ** (-1 / parameters.homeostasis_power),
        "p i": stimuli_probabilities * stimuli_modulation_curve,
        "p^{2/alpha} i^2": (
            stimuli_probabilities ** (2 / parameters.homeostasis_power)
            * stimuli_modulation_curve**2
        ),
        "p^{1/alpha} i^2": (
            stimuli_probabilities ** (1 / parameters.homeostasis_power)
            * stimuli_modulation_curve**2
        ),
        "r": total_rate,
        "d": density,
        "g": gains,
        "w": widths,
        "g d": density * gains,
        "g^2 d": density * (gains**2),
    }

    # Normalise the curves to sum to 1
    population_response_metrics = {
        key: renormalise(value) for key, value in population_response_metrics.items()
    }

    population_response_metrics.update(
        {
            "rates": rates,
            "argmax_stimuli": argmax_stimuli,
            "average_rates": torch.einsum(
                "ij,j->i", rates, stimuli_probabilities
            ),  # [N_I]
            "stimulus_space": stimulus_space,
        }
    )

    # Convert to numpy arrays for logging
    population_response_metrics = {
        key: value.detach().cpu().numpy()
        for key, value in population_response_metrics.items()
    }

    return population_response_metrics


@jaxtyped(typechecker=typechecked)
def compute_discrepancies(
    population_response_metrics: dict[str, Any],
) -> dict[str, float]:

    stable_curves_keys = [
        "c",
        "i",
        "i^2",
        "p",
        "p^{-1/alpha}",
        "p i",
        "p^{2/alpha} i^2",
        "p^{1/alpha} i^2",
    ]

    non_stable_curves_keys = {
        "r",
        "d",
        "g",
        "g d",
        "g^2 d",
        "c",
    }

    # Extract the curves from the population_response_metrics dictionary
    stable_curves = {
        key: population_response_metrics[key] for key in stable_curves_keys
    }
    non_stable_curves = {
        key: population_response_metrics[key] for key in non_stable_curves_keys
    }

    # Compute the discrepancies between stable curves and non-stable curves
    discrepancies = {}

    # Loop through each stable curve
    for stable_curve_name, stable_curve in stable_curves.items():
        # Compare with each non-stable curve
        for non_stable_curve_name, non_stable_curve in non_stable_curves.items():
            # Compute the circular discrepancy between the two curves
            discrepancy = circular_discrepancy(stable_curve, non_stable_curve)
            # Store the discrepancy in the dictionary
            discrepancies[f"diff/diff({stable_curve_name},{non_stable_curve_name})"] = (
                discrepancy
            )

    return discrepancies


@jaxtyped(typechecker=typechecked)
def compute_regressions(
    population_response_metrics: dict[str, Any],
) -> dict[str, float]:

    stable_curves_keys = [
        "c",
        "i",
        "p",
    ]

    non_stable_curves_keys = {
        "r",
        "g",
        "d",
        "c",
        "w",
    }

    # Extract the curves from the population_response_metrics dictionary
    stable_curves = {
        key: population_response_metrics[key] for key in stable_curves_keys
    }
    non_stable_curves = {
        key: population_response_metrics[key] for key in non_stable_curves_keys
    }

    # Compute the discrepancies between stable curves and non-stable curves
    regression_stats = {}

    # Loop through each stable curve
    for stable_curve_name, stable_curve in stable_curves.items():
        # Compare with each non-stable curve
        for non_stable_curve_name, non_stable_curve in non_stable_curves.items():
            # Compute the circular discrepancy between the two curves
            gamma, r_squared = power_law_regression(
                stable_curve,
                non_stable_curve,
            )
            # Store the discrepancy in the dictionary
            regression_stats[
                f"reg/gamma({stable_curve_name},{non_stable_curve_name})"
            ] = gamma
            regression_stats[
                f"reg/r_squared({stable_curve_name},{non_stable_curve_name})"
            ] = r_squared

    return regression_stats


@jaxtyped(typechecker=typechecked)
def dynamics_log(
    dW: Float[torch.Tensor, "N_I N_E"],
    dM: Float[torch.Tensor, "N_I N_I"],
    k_E: Float[torch.Tensor, "N_I"],
    new_k_E: Float[torch.Tensor, "N_I"],
    r: Float[torch.Tensor, "N_I num_latents"],
    probabilities: Float[torch.Tensor, "num_latents"],
    parameters: SimulationParameters,
    iteration_step: int,
) -> dict[str, Any]:
    N_E = parameters.N_E
    N_I = parameters.N_I
    dt = parameters.dt

    recurrent_update_magnitude = torch.sum(torch.abs(dM)).item() / (N_I * N_I * dt)
    feedforward_update_magnitude = torch.sum(torch.abs(dW)).item() / (N_E * N_I * dt)
    excitatory_mass_update_magnitude = torch.sum(torch.abs(new_k_E - k_E)).item() / (
        N_I * dt
    )
    avg_rates = torch.einsum("ij,j->i", r, probabilities)  # [N_I]
    centred_r = r - avg_rates.unsqueeze(-1)  # [N_I, num_latents]
    avg_rate = avg_rates.mean().item()
    avg_var = (
        torch.einsum("ij,ij,j->i", centred_r, centred_r, probabilities).mean().item()
    )
    avg_second_moment = torch.einsum("ij,ij,j->i", r, r, probabilities).mean().item()

    log_dict = {
        "dynamics/recurrent_update_magnitude": recurrent_update_magnitude,
        "dynamics/feedforward_update_magnitude": feedforward_update_magnitude,
        "dynamics/excitatory_mass_update_magnitude": excitatory_mass_update_magnitude,
        "dynamics/average_excitatory_mass": torch.mean(new_k_E).item(),
        "dynamics/average_neuron_rate": avg_rate,
        "dynamics/average_variance": avg_var,
        "dynamics/avg_second_moment": avg_second_moment,
        "time": dt * iteration_step,
    }

    return log_dict


# @jaxtyped(typechecker=typechecked)
# def compute_discrepancies(
#     population_response_metrics: dict[str, Any],
# ) -> dict[str, float]:
#     # Extract metrics from the population_metrics dictionary
#     normalised_total_rate = population_response_metrics[
#         "normalised_total_rate"
#     ]  # [num_latents]
#     preferred_stimulus_density = population_response_metrics[
#         "preferred_stimulus_density"
#     ]  # [num_latents]
#     squared_stimuli_probabilities = population_response_metrics[
#         "squared_stimuli_probabilities"
#     ]  # [num_latents]
#     stimuli_probabilities = population_response_metrics[
#         "stimuli_probabilities"
#     ]  # [num_latents]
#     stimuli_modulation = population_response_metrics[
#         "stimuli_modulation_curve"
#     ]  # [num_latents]
#     squared_stimuli_modulation = population_response_metrics[
#         "squared_stimuli_modulation"
#     ]  # [num_latents]
#     constant_curve = np.ones_like(normalised_total_rate) / len(
#         normalised_total_rate
#     )  # [num_latents]
#     gains = population_response_metrics["gains"]  # [num_latents]
#     inverse_gains = population_response_metrics["inverse_gains"]  # [num_latents]
#     mmpp = population_response_metrics["m^2 p^2"]  # [num_latents]
#     mmp = population_response_metrics["m^2 p"]  # [num_latents]
#     mp = population_response_metrics["m p"]  # [num_latents]


# @jaxtyped(typechecker=typechecked)
# def rate_mode_log(
#     population_response_metrics: dict[str, Any],
#     parameters: SimulationParameters,
# ) -> dict[str, Any]:

#     # Extract metrics from the population_response_metrics dictionary
#     rates = population_response_metrics["rates"]
#     stimuli_probabilities = population_response_metrics["stimuli_probabilities"]

#     num_latents = parameters.num_latents
#     mean_rate = np.mean(rates, axis=(0, 1))  # Scalar
#     var_rate = np.sum(np.var(rates, axis=1), axis=0)  # Scalar

#     stimuli_rates = np.sum(rates, axis=0)  # [num_latents]
#     rate_probability_ratio = stimuli_rates / stimuli_probabilities
#     mean_rate_probability_ratio = np.mean(rate_probability_ratio)
#     normalised_rate_probability_ratio = (
#         rate_probability_ratio / mean_rate_probability_ratio
#     )

#     rate_allocation_error = np.mean(
#         np.abs(rate_probability_ratio - mean_rate_probability_ratio)
#         / np.abs(mean_rate_probability_ratio)
#     )

#     # Construct a dictionary with the quantities to be logged:
#     log_dict = {
#         "steady_state/population_rate": mean_rate,
#         "steady_state/population_variance": var_rate,
#     }
#     log_dict.update(
#         {
#             f"steady_state/stimuli_rate_{jj}": stimuli_rates[jj]
#             for jj in range(num_latents)
#         }
#     )
#     log_dict.update(
#         {
#             f"steady_state/rate_probability_ratio_{jj}": rate_probability_ratio[jj]
#             for jj in range(num_latents)
#         }
#     )
#     log_dict.update(
#         {
#             f"steady_state/normalised_rate_probability_ratio_{jj}": normalised_rate_probability_ratio[
#                 jj
#             ]
#             for jj in range(num_latents)
#         }
#     )
#     log_dict.update(
#         {
#             "steady_state/rate_allocation_error": rate_allocation_error,
#         }
#     )

#     return log_dict


# def compute_population_response_metrics(
#     rates: Float[torch.Tensor, "N_I num_latents"],
#     input_generator: ModulatedCircularGenerator,
#     parameters: SimulationParameters,
# ) -> dict[str, Any]:
#     r"""For consistency, everything here is going to be in torch. Then we'll convert when we return."""

#     N_E = parameters.N_E
#     num_latents = parameters.num_latents
#     N_I = parameters.N_I

#     stimuli_probabilities = input_generator.stimuli_probabilities  # [num_latents]
#     squared_stimuli_probabilities = (
#         stimuli_probabilities**2
#     )  # [num_latents] (squared probabilities of each stimulus)
#     # Normalise this to sum to 1
#     squared_stimuli_probabilities = (
#         squared_stimuli_probabilities / squared_stimuli_probabilities.sum()
#     )  # [num_latents]

#     stimuli_modulation_curve = input_generator.modulation_curve  # [num_latents]
#     squared_stimuli_modulation = (
#         stimuli_modulation_curve**2
#     )  # [num_latents] (squared modulation of each stimulus)

#     # Renormalise both of these curves to sum to 1
#     stimuli_modulation_curve = (
#         stimuli_modulation_curve / stimuli_modulation_curve.sum()
#     )  # [num_latents]
#     squared_stimuli_modulation = (
#         squared_stimuli_modulation / squared_stimuli_modulation.sum()
#     )  # [num_latents]

#     stimuli_patterns = input_generator.stimuli_patterns  # [N_E, num_latents]
#     average_input_activation = stimuli_patterns @ stimuli_probabilities.to(
#         dtype=stimuli_patterns.dtype
#     )  # [N_E]

#     normalised_average_input_activation = (N_E / num_latents) * (
#         average_input_activation / average_input_activation.sum()
#     )  # [N_E]

#     average_rates = rates @ stimuli_probabilities  # [N_I] (average rate of each neuron)

#     # Compute the integral of the tuning curves against the stimuli patterns
#     pattern_overlaps = (
#         rates @ stimuli_patterns.T
#     )  # [N_I, num_latents] @ [num_latents, N_E] = [N_I, N_E]
#     # Normalise to turn the sum into an integral
#     pattern_overlaps = pattern_overlaps * (2 * torch.pi / num_latents)  # [N_I, N_E]

#     # Compute the input pattern overlap matrix
#     overlap_matrix = (
#         stimuli_patterns @ stimuli_patterns.T
#     )  # [N_E, num_latents] @ [num_latents, N_E] = [N_E, N_E]

#     # Compute the inverse overlap matrix
#     # inverse_overlap_matrix = torch.linalg.inv(overlap_matrix)  # [N_E, N_E]
#     generalised_gain_matrix = torch.linalg.solve(
#         overlap_matrix,
#         pattern_overlaps.T,
#     )  # [N_E, N_E] @ [N_E, N_I] = [N_E, N_I]
#     # This assumes that the overlap matrix is invertible.
#     # This requires that num_latents >= N_E.

#     # Compute the generalised density
#     generalised_density = (
#         generalised_gain_matrix / generalised_gain_matrix.sum(axis=0, keepdim=True)
#     ).sum(
#         axis=1
#     )  # [N_E]

#     # Compute the generalised gain
#     generalised_gain = (generalised_gain_matrix / generalised_density.unsqueeze(1)).sum(
#         axis=1
#     )  # [N_E]

#     # Take the inner product against the input patterns to get the density of the preferred stimulus
#     generalised_density = (
#         generalised_density @ stimuli_patterns
#     )  # [N_E] @ [N_E, num_latents] = [num_latents]

#     # Take the inner product against the input patterns to get the gain of the preferred stimulus
#     generalised_gain = (
#         generalised_gain @ stimuli_patterns
#     )  # [N_E] @ [N_E, num_latents] = [num_latents]

#     argmax_rates = rates.argmax(
#         axis=1
#     )  # [N_I] (indices of max response in num_latents)
#     stimulus_space = input_generator.stimuli_positions.squeeze()  # [num_latents]
#     argmax_stimuli = stimulus_space[
#         argmax_rates
#     ].flatten()  # [N_I] (stimuli of max response)
#     max_rates, _ = rates.max(axis=1)  # [N_I]

#     normalised_max_rates = (N_I / num_latents) * max_rates / max_rates.sum()  # [N_I]

#     total_rate = rates.sum(axis=0)  # [num_latents]
#     normalised_total_rate = total_rate / total_rate.sum()  # [num_latents]

#     argmax_stimuli = argmax_stimuli  # [N_I] (stimuli of max response)

#     bw_multiplier = N_I ** (-0.2)
#     max_rate_range = max_rates.max() - max_rates.min()

#     smoothed_max_rates = circular_smooth_huber(
#         argmax_stimuli,
#         max_rates,
#         stimulus_space,
#         bw=bw_multiplier * 0.4,
#         delta=0.25 * max_rate_range.item(),
#     )  # [num_latents]

#     preferred_stimulus_density = circular_kde(
#         argmax_stimuli, stimulus_space, bw=bw_multiplier * 0.2
#     )  # [num_latents]

#     density_times_gain = preferred_stimulus_density * smoothed_max_rates
#     density_times_gain = density_times_gain / density_times_gain.sum()  # [num_latents]

#     # Squared modulation curve times squared stimulus probabilities
#     mmpp = squared_stimuli_modulation * squared_stimuli_probabilities  # [num_latents]
#     mmpp = mmpp / mmpp.sum()  # [num_latents]
#     mmp = squared_stimuli_modulation * stimuli_probabilities  # [num_latents]
#     mmp = mmp / mmp.sum()  # [num_latents]
#     mp = stimuli_modulation_curve * squared_stimuli_probabilities  # [num_latents]
#     mp = mp / mp.sum()  # [num_latents]

#     # Compute the inverse gains
#     inverse_gains = 1 / smoothed_max_rates
#     # Normalise
#     inverse_gains = inverse_gains / inverse_gains.sum()

#     gains = smoothed_max_rates / smoothed_max_rates.sum()  # [num_latents]

#     population_response_metrics = {
#         "stimuli_probabilities": stimuli_probabilities.detach().cpu().numpy(),
#         "stimuli_patterns": stimuli_patterns.detach().cpu().numpy(),
#         "stimulus_space": stimulus_space.detach().cpu().numpy(),
#         "normalised_average_input_activation": normalised_average_input_activation.detach()
#         .cpu()
#         .numpy(),
#         "rates": rates.detach().cpu().numpy(),
#         "normalised_max_rates": normalised_max_rates.detach().cpu().numpy(),
#         "argmax_stimuli": argmax_stimuli.detach().cpu().numpy(),
#         "normalised_total_rate": normalised_total_rate.detach().cpu().numpy(),
#         "smoothed_max_rates": smoothed_max_rates.detach().cpu().numpy(),
#         "gains": gains.detach().cpu().numpy(),
#         "preferred_stimulus_density": preferred_stimulus_density.detach().cpu().numpy(),
#         "density_times_gain": density_times_gain.detach().cpu().numpy(),
#         "average_rates": average_rates.detach().cpu().numpy(),
#         "pattern_overlaps": pattern_overlaps.detach().cpu().numpy(),
#         "generalised_density": generalised_density.detach().cpu().numpy(),
#         "generalised_gain": generalised_gain.detach().cpu().numpy(),
#         "squared_stimuli_probabilities": squared_stimuli_probabilities.detach()
#         .cpu()
#         .numpy(),
#         "stimuli_modulation_curve": stimuli_modulation_curve.detach().cpu().numpy(),
#         "squared_stimuli_modulation": squared_stimuli_modulation.detach().cpu().numpy(),
#         "m^2 p^2": mmpp.detach().cpu().numpy(),
#         "m^2 p": mmp.detach().cpu().numpy(),
#         "m p": mp.detach().cpu().numpy(),
#         "inverse_gains": inverse_gains.detach().cpu().numpy(),
#     }

#     return population_response_metrics
