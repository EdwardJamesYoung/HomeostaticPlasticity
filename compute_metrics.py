import torch
import numpy as np
from typing import Tuple, Any, Optional
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from input_generation import InputGenerator, DiscreteGenerator, CircularGenerator
from params import SimulationParameters
from utils import (
    circular_discrepancy,
    circular_kde,
    circular_smooth_values,
    circular_smooth_median,
    circular_smooth_huber,
)
from scipy.interpolate import interp1d


@jaxtyped(typechecker=typechecked)
def compute_firing_rates(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    u: Float[torch.Tensor, "N_E num_stimuli"],
    parameters: SimulationParameters,
    v_init: Optional[Float[torch.Tensor, "N_I num_stimuli"]] = None,
    threshold=1e-6,
    max_iter=100000,
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


def compute_population_response_metrics(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    input_generator: CircularGenerator,
    parameters: SimulationParameters,
) -> dict[str, Any]:
    r"""For consistency, everything here is going to be in torch. Then we'll convert when we return."""

    N_E = parameters.N_E
    num_latents = parameters.num_latents
    N_I = parameters.N_I

    stimuli_probabilities = input_generator.stimuli_probabilities  # [num_latents]
    squared_stimuli_probabilities = (
        stimuli_probabilities**2
    )  # [num_latents] (squared probabilities of each stimulus)
    # Normalise this to sum to 1
    squared_stimuli_probabilities = (
        squared_stimuli_probabilities / squared_stimuli_probabilities.sum()
    )  # [num_latents]

    stimuli_patterns = input_generator.stimuli_patterns  # [N_E, num_latents]
    average_input_activation = stimuli_patterns @ stimuli_probabilities.to(
        dtype=stimuli_patterns.dtype
    )  # [N_E]

    normalised_average_input_activation = (N_E / num_latents) * (
        average_input_activation / average_input_activation.sum()
    )  # [N_E]

    rates, _ = compute_firing_rates(
        W,
        M,
        stimuli_patterns,
        parameters=parameters,
        threshold=1e-8,
    )  # [N_I, num_latents]
    average_rates = rates @ stimuli_probabilities  # [N_I] (average rate of each neuron)

    # Compute the integral of the tuning curves against the stimuli patterns
    pattern_overlaps = (
        rates @ stimuli_patterns.T
    )  # [N_I, num_latents] @ [num_latents, N_E] = [N_I, N_E]
    # Normalise to turn the sum into an integral
    pattern_overlaps = pattern_overlaps * (2 * torch.pi / num_latents)  # [N_I, N_E]

    # Compute the input pattern overlap matrix
    overlap_matrix = (
        stimuli_patterns @ stimuli_patterns.T
    )  # [N_E, num_latents] @ [num_latents, N_E] = [N_E, N_E]

    # Compute the inverse overlap matrix
    # inverse_overlap_matrix = torch.linalg.inv(overlap_matrix)  # [N_E, N_E]
    generalised_gain_matrix = torch.linalg.solve(
        overlap_matrix,
        pattern_overlaps.T,
    )  # [N_E, N_E] @ [N_E, N_I] = [N_E, N_I]
    # This assumes that the overlap matrix is invertible.
    # This requires that num_latents >= N_E.

    # Compute the generalised density
    generalised_density = (
        generalised_gain_matrix / generalised_gain_matrix.sum(axis=0, keepdim=True)
    ).sum(
        axis=1
    )  # [N_E]

    # Compute the generalised gain
    generalised_gain = (generalised_gain_matrix / generalised_density.unsqueeze(1)).sum(
        axis=1
    )  # [N_E]

    # Take the inner product against the input patterns to get the density of the preferred stimulus
    generalised_density = (
        generalised_density @ stimuli_patterns
    )  # [N_E] @ [N_E, num_latents] = [num_latents]

    # Take the inner product against the input patterns to get the gain of the preferred stimulus
    generalised_gain = (
        generalised_gain @ stimuli_patterns
    )  # [N_E] @ [N_E, num_latents] = [num_latents]

    argmax_rates = rates.argmax(
        axis=1
    )  # [N_I] (indices of max response in num_latents)
    stimulus_space = input_generator.stimuli_positions.squeeze()  # [num_latents]
    argmax_stimuli = stimulus_space[
        argmax_rates
    ].flatten()  # [N_I] (stimuli of max response)
    max_rates, _ = rates.max(axis=1)  # [N_I]

    normalised_max_rates = (N_I / num_latents) * max_rates / max_rates.sum()  # [N_I]

    total_rate = rates.sum(axis=0)  # [num_latents]
    normalised_total_rate = total_rate / total_rate.sum()  # [num_latents]

    argmax_stimuli = argmax_stimuli  # [N_I] (stimuli of max response)

    bw_multiplier = N_I ** (-0.2)
    max_rate_range = max_rates.max() - max_rates.min()

    smoothed_max_rates = circular_smooth_huber(
        argmax_stimuli,
        max_rates,
        stimulus_space,
        bw=bw_multiplier * 0.4,
        delta=0.25 * max_rate_range.item(),
    )  # [num_latents]

    preferred_stimulus_density = circular_kde(
        argmax_stimuli, stimulus_space, bw=bw_multiplier * 0.2
    )  # [num_latents]

    density_times_gain = preferred_stimulus_density * smoothed_max_rates
    density_times_gain = density_times_gain / density_times_gain.sum()  # [num_latents]

    population_response_metrics = {
        "stimuli_probabilities": stimuli_probabilities.detach().cpu().numpy(),
        "stimuli_patterns": stimuli_patterns.detach().cpu().numpy(),
        "stimulus_space": stimulus_space.detach().cpu().numpy(),
        "normalised_average_input_activation": normalised_average_input_activation.detach()
        .cpu()
        .numpy(),
        "rates": rates.detach().cpu().numpy(),
        "normalised_max_rates": normalised_max_rates.detach().cpu().numpy(),
        "argmax_stimuli": argmax_stimuli.detach().cpu().numpy(),
        "normalised_total_rate": normalised_total_rate.detach().cpu().numpy(),
        "smoothed_max_rates": smoothed_max_rates.detach().cpu().numpy(),
        "preferred_stimulus_density": preferred_stimulus_density.detach().cpu().numpy(),
        "density_times_gain": density_times_gain.detach().cpu().numpy(),
        "average_rates": average_rates.detach().cpu().numpy(),
        "pattern_overlaps": pattern_overlaps.detach().cpu().numpy(),
        "generalised_density": generalised_density.detach().cpu().numpy(),
        "generalised_gain": generalised_gain.detach().cpu().numpy(),
        "squared_stimuli_probabilities": squared_stimuli_probabilities.detach()
        .cpu()
        .numpy(),
    }

    return population_response_metrics


@jaxtyped(typechecker=typechecked)
def compute_discrepancies(
    population_response_metrics: dict[str, Any],
) -> dict[str, float]:
    # Extract metrics from the population_metrics dictionary
    normalised_total_rate = population_response_metrics[
        "normalised_total_rate"
    ]  # [num_latents]
    preferred_stimulus_density = population_response_metrics[
        "preferred_stimulus_density"
    ]  # [num_latents]
    squared_stimuli_probabilities = population_response_metrics[
        "squared_stimuli_probabilities"
    ]  # [num_latents]
    stimuli_probabilities = population_response_metrics[
        "stimuli_probabilities"
    ]  # [num_latents]
    constant_curve = np.ones_like(normalised_total_rate) / len(
        normalised_total_rate
    )  # [num_latents]
    # stimulus_space = population_response_metrics["stimulus_space"]  # [num_latents]

    # Create interpolated version of average_input_activation
    # First, create a mapping from N_E input space to the circular stimulus space
    # N_E = len(average_input_activation)
    # num_latents = len(stimulus_space)

    # # Interpolate from N_E space to num_latents space using 1D linear interpolation
    # input_indices = np.linspace(0, 2 * np.pi, N_E, endpoint=False)

    # # Create interpolation function
    # interpolation_function = interp1d(
    #     input_indices,
    #     average_input_activation,
    #     kind="linear",
    #     bounds_error=False,
    #     fill_value=(average_input_activation[-1], average_input_activation[0]),
    # )

    # # Evaluate at stimulus positions
    # target_positions = np.linspace(0, 2 * np.pi, num_latents, endpoint=False)
    # interpolated_average_input_activation = interpolation_function(target_positions)

    # # Normalize
    # interpolated_average_input_activation = (
    #     interpolated_average_input_activation
    #     / interpolated_average_input_activation.sum()
    # )

    # What are all the quantities of interest that we want to compute the distance matrix for?
    # 1. The (normalised) total rate
    # 2. The probabilities
    # 3. The density
    # 4. The activated stimulus probabilities
    # 5. The constant curve

    curves = {
        "r": normalised_total_rate,
        "p": stimuli_probabilities,
        "p^2": squared_stimuli_probabilities,
        "d": preferred_stimulus_density,
        "c": constant_curve,
    }

    # Compute the discrepancies between the curves
    discrepancies = {}
    curve_names = list(curves.keys())

    # Loop through unique pairs
    for i, curve_1_name in enumerate(curve_names):
        curve_1 = curves[curve_1_name]
        # Start from i+1 to avoid duplicates
        for j in range(i + 1, len(curve_names)):
            curve_2_name = curve_names[j]
            curve_2 = curves[curve_2_name]

            # Compute the circular discrepancy between the two curves
            discrepancy = circular_discrepancy(curve_1, curve_2)
            # Store the discrepancy in the dictionary
            discrepancies[f"diff/diff({curve_1_name},{curve_2_name})"] = discrepancy

    return discrepancies


@jaxtyped(typechecker=typechecked)
def rate_mode_log(
    population_response_metrics: dict[str, Any],
    parameters: SimulationParameters,
) -> dict[str, Any]:

    # Extract metrics from the population_response_metrics dictionary
    rates = population_response_metrics["rates"]
    stimuli_probabilities = population_response_metrics["stimuli_probabilities"]

    num_latents = parameters.num_latents
    mean_rate = np.mean(rates, axis=(0, 1))  # Scalar
    var_rate = np.sum(np.var(rates, axis=1), axis=0)  # Scalar

    stimuli_rates = np.sum(rates, axis=0)  # [num_latents]
    rate_probability_ratio = stimuli_rates / stimuli_probabilities
    mean_rate_probability_ratio = np.mean(rate_probability_ratio)
    normalised_rate_probability_ratio = (
        rate_probability_ratio / mean_rate_probability_ratio
    )

    rate_allocation_error = np.mean(
        np.abs(rate_probability_ratio - mean_rate_probability_ratio)
        / np.abs(mean_rate_probability_ratio)
    )

    # Construct a dictionary with the quantities to be logged:
    log_dict = {
        "steady_state/population_rate": mean_rate,
        "steady_state/population_variance": var_rate,
    }
    log_dict.update(
        {
            f"steady_state/stimuli_rate_{jj}": stimuli_rates[jj]
            for jj in range(num_latents)
        }
    )
    log_dict.update(
        {
            f"steady_state/rate_probability_ratio_{jj}": rate_probability_ratio[jj]
            for jj in range(num_latents)
        }
    )
    log_dict.update(
        {
            f"steady_state/normalised_rate_probability_ratio_{jj}": normalised_rate_probability_ratio[
                jj
            ]
            for jj in range(num_latents)
        }
    )
    log_dict.update(
        {
            "steady_state/rate_allocation_error": rate_allocation_error,
        }
    )

    return log_dict


def dynamics_log(
    dW: Float[torch.Tensor, "N_I N_E"],
    dM: Float[torch.Tensor, "N_I N_I"],
    k_E: Float[torch.Tensor, "N_I"],
    new_k_E: Float[torch.Tensor, "N_I"],
    r: Float[torch.Tensor, "N_I"],
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

    log_dict = {
        "dynamics/recurrent_update_magnitude": recurrent_update_magnitude,
        "dynamics/feedforward_update_magnitude": feedforward_update_magnitude,
        "dynamics/excitatory_mass_update_magnitude": excitatory_mass_update_magnitude,
        "dynamics/average_excitatory_mass": torch.mean(new_k_E).item(),
        "dynamics/average_neuron_rate": torch.mean(r).item(),
        "time": dt * iteration_step,
    }

    return log_dict
