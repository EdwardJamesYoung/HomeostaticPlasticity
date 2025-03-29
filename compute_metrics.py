import torch
import numpy as np
from typing import Tuple, Any
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from input_generation import InputGenerator, DiscreteGenerator, CircularGenerator
from params import SimulationParameters
from utils import circular_discrepancy, circular_kde, circular_smooth_values
from scipy.interpolate import interp1d


@jaxtyped(typechecker=typechecked)
def compute_firing_rates(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    u: Float[torch.Tensor, "N_E num_stimuli"],
    parameters: SimulationParameters,
    threshold=1e-8,
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
    dt = 0.5 * parameters.dt
    tau_v = parameters.tau_v
    activation_function = parameters.activation_function

    # Initialise the input h and the voltage v
    h = W @ u
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

    if counter == max_iter:
        wandb.alert(
            title="Firing rate computation did not converge",
            text="The firing rate computation did not converge within the maximum number of iterations",
            level=wandb.AlertLevel.WARN,
        )

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

    smoothed_max_rates = circular_smooth_values(
        argmax_stimuli, normalised_max_rates, stimulus_space, bw=bw_multiplier * 0.4
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
    density_times_gain = population_response_metrics[
        "density_times_gain"
    ]  # [num_latents]
    average_input_activation = population_response_metrics[
        "normalised_average_input_activation"
    ]  # [N_E]
    preferred_stimulus_density = population_response_metrics[
        "preferred_stimulus_density"
    ]  # [num_latents]
    stimulus_space = population_response_metrics["stimulus_space"]  # [num_latents]

    # Create interpolated version of average_input_activation
    # First, create a mapping from N_E input space to the circular stimulus space
    N_E = len(average_input_activation)
    num_latents = len(stimulus_space)

    # Interpolate from N_E space to num_latents space using 1D linear interpolation
    input_indices = np.linspace(0, 2 * np.pi, N_E, endpoint=False)

    # Create interpolation function
    interpolation_function = interp1d(
        input_indices,
        average_input_activation,
        kind="linear",
        bounds_error=False,
        fill_value=(average_input_activation[-1], average_input_activation[0]),
    )

    # Evaluate at stimulus positions
    target_positions = np.linspace(0, 2 * np.pi, num_latents, endpoint=False)
    interpolated_average_input_activation = interpolation_function(target_positions)

    # Normalize
    interpolated_average_input_activation = (
        interpolated_average_input_activation
        / interpolated_average_input_activation.sum()
    )

    discrepancies = {
        "discrepancy/diff(constant, normalised_total_rate)": circular_discrepancy(
            normalised_total_rate,
            np.ones_like(normalised_total_rate) / len(normalised_total_rate),
        ),
        "discrepancy/diff(constant, density_times_gain)": circular_discrepancy(
            density_times_gain,
            np.ones_like(density_times_gain) / len(density_times_gain),
        ),
        "discrepancy/diff(density_times_gain, normalised_total_rate)": circular_discrepancy(
            density_times_gain, normalised_total_rate
        ),
        "discrepancy/diff(normalised_average_input_activation, preferred_stimulus_density)": circular_discrepancy(
            interpolated_average_input_activation, preferred_stimulus_density
        ),
    }

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
