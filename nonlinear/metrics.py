import torch
from typing import Any
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from params import SimulationParameters

from input_generation import InputGenerator
from params import SimulationParameters
from utils import (
    circular_discrepancy,
    circular_kde,
    circular_smooth_huber,
    power_law_regression,
)


@jaxtyped(typechecker=typechecked)
def compute_tuning_curve_widths(
    rates: Float[torch.Tensor, "batch N_I num_stimuli"],
    stimulus_locations: Float[torch.Tensor, "num_stimuli num_dimensions"],
) -> Float[torch.Tensor, "batch N_I"]:
    num_dimensions = stimulus_locations.shape[-1]

    if num_dimensions == 1:
        return _compute_circular_width_1d(rates, stimulus_locations.squeeze(-1))
    elif num_dimensions == 2:
        return _compute_circular_width_2d(rates, stimulus_locations)
    else:
        raise ValueError(f"Unsupported number of dimensions: {num_dimensions}")


@jaxtyped(typechecker=typechecked)
def _compute_circular_width_1d(
    rates: Float[torch.Tensor, "batch N_I num_stimuli"],
    stimulus_angles: Float[torch.Tensor, "num_stimuli"],
) -> Float[torch.Tensor, "batch N_I"]:
    """
    Compute circular width for 1D case using circular standard deviation.
    """
    # Normalize each tuning curve to create probability distributions
    rates_shifted = (
        rates - rates.min(dim=-1, keepdim=True)[0]
    )  # [batch, N_I, num_stimuli]

    # Handle edge case where tuning curve is constant
    curve_sums = rates_shifted.sum(dim=-1, keepdim=True)  # [batch, N_I, 1]
    curve_sums = torch.where(curve_sums == 0, torch.ones_like(curve_sums), curve_sums)

    # Normalize to probability distribution
    p = rates_shifted / curve_sums  # [batch, N_I, num_stimuli]

    # Compute circular mean resultant vector for each neuron
    # R = |sum(p[i] * exp(i * θ[i]))|
    cos_component = torch.sum(p * torch.cos(stimulus_angles), dim=-1)  # [batch, N_I]
    sin_component = torch.sum(p * torch.sin(stimulus_angles), dim=-1)  # [batch, N_I]

    # Magnitude of resultant vector
    R = torch.sqrt(cos_component**2 + sin_component**2)  # [batch, N_I]

    # Circular variance: σ² = -2 * log(R)
    epsilon = 1e-8
    R_clamped = torch.clamp(R, min=epsilon, max=1.0)
    circular_variance = -2 * torch.log(R_clamped)  # [batch, N_I]

    # Circular standard deviation
    circular_std = torch.sqrt(circular_variance)  # [batch, N_I]

    # Convert to width measure (factor of 2 gives full width)
    widths = 2 * circular_std

    return widths


@jaxtyped(typechecker=typechecked)
def _compute_circular_width_2d(
    rates: Float[torch.Tensor, "batch N_I num_stimuli"],
    stimulus_locations: Float[torch.Tensor, "num_stimuli 2"],
) -> Float[torch.Tensor, "batch N_I"]:
    """
    Compute circular width for 2D case using trace of circular covariance matrix.

    This computes the circular variance for each dimension separately and
    averages them, which gives a rotationally-invariant width measure.
    """
    # Normalize each tuning curve to create probability distributions
    rates_shifted = (
        rates - rates.min(dim=-1, keepdim=True)[0]
    )  # [batch, N_I, num_stimuli]

    # Handle edge case where tuning curve is constant
    curve_sums = rates_shifted.sum(dim=-1, keepdim=True)  # [batch, N_I, 1]
    curve_sums = torch.where(curve_sums == 0, torch.ones_like(curve_sums), curve_sums)

    # Normalize to probability distribution
    p = rates_shifted / curve_sums  # [batch, N_I, num_stimuli]

    # Extract angles for each dimension
    theta1 = stimulus_locations[:, 0]  # [num_stimuli]
    theta2 = stimulus_locations[:, 1]  # [num_stimuli]

    # Compute circular mean resultant vectors for each dimension
    # Dimension 1
    cos1 = torch.sum(p * torch.cos(theta1), dim=-1)  # [batch, N_I]
    sin1 = torch.sum(p * torch.sin(theta1), dim=-1)  # [batch, N_I]
    R1 = torch.sqrt(cos1**2 + sin1**2)  # [batch, N_I]

    # Dimension 2
    cos2 = torch.sum(p * torch.cos(theta2), dim=-1)  # [batch, N_I]
    sin2 = torch.sum(p * torch.sin(theta2), dim=-1)  # [batch, N_I]
    R2 = torch.sqrt(cos2**2 + sin2**2)  # [batch, N_I]

    # Compute circular variances for each dimension
    epsilon = 1e-8
    R1_clamped = torch.clamp(R1, min=epsilon, max=1.0)
    R2_clamped = torch.clamp(R2, min=epsilon, max=1.0)

    var1 = -2 * torch.log(R1_clamped)  # [batch, N_I]
    var2 = -2 * torch.log(R2_clamped)  # [batch, N_I]

    # Average the variances (trace of covariance matrix / 2)
    average_variance = (var1 + var2) / 2  # [batch, N_I]

    # Take square root to get standard deviation measure
    circular_std = torch.sqrt(average_variance)  # [batch, N_I]

    # Convert to width measure (factor of 2 gives full width)
    widths = 2 * circular_std

    return widths


def compute_population_response_metrics(
    rates: Float[torch.Tensor, "batch N_I num_latents"],
    input_generator: InputGenerator,
    parameters: SimulationParameters,
) -> dict[str, Any]:
    r"""For consistency, everything here is going to be in torch. Then we'll convert to numpy when we return."""

    batch_size = parameters.batch_size
    N_E = parameters.N_E
    N_I = parameters.N_I

    stimuli_probabilities = input_generator.stimuli_probabilities  # [num_latents]
    # If the input generator has latent_stimuli_probabilities, use those instead
    if hasattr(input_generator, "latent_stimuli_probabilities"):
        stimuli_probabilities = input_generator.latent_stimuli_probabilities

    stimuli_modulation_curve = input_generator.modulation_curve  # [num_latents]

    argmax_rates = rates.argmax(
        axis=-1
    )  # [batch, N_I] (indices of max response in num_latents)
    stimulus_space = input_generator.stimuli_positions.squeeze()  # [num_latents]
    argmax_stimuli = stimulus_space[
        argmax_rates
    ].flatten()  # [batch, N_I] (stimuli of max response)
    max_rates, _ = rates.max(axis=-1)  # [batch, N_I]

    total_rate = rates.sum(dim=(0, 1))  # [num_latents]

    bw_multiplier = (batch_size * N_I) ** (-0.2)
    max_rate_range = max_rates.max() - max_rates.min()

    gains = circular_smooth_huber(
        argmax_stimuli.flatten(),
        max_rates.flatten(),
        stimulus_space,
        bw=bw_multiplier * 0.4,
        delta=0.25 * max_rate_range.item(),
    )  # [num_latents]

    # Find the width at half height of the tuning curve
    tuning_curve_widths = compute_tuning_curve_widths(
        rates, stimulus_space
    )  # [batch, N_I]
    width_range = tuning_curve_widths.max() - tuning_curve_widths.min()

    widths = circular_smooth_huber(
        argmax_stimuli.flatten(),
        tuning_curve_widths.flatten(),
        stimulus_space,
        bw=bw_multiplier * 0.4,
        delta=0.25 * width_range.item(),
    )

    density = circular_kde(
        argmax_stimuli.flatten(), stimulus_space, bw=bw_multiplier * 0.2
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
                "bij,j->bi", rates, stimuli_probabilities
            ),  # [batch, N_I]
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
    W: Float[torch.Tensor, "batch N_I N_E"],
    dW: Float[torch.Tensor, "batch N_I N_E"],
    M: Float[torch.Tensor, "batch N_I N_I"],
    dM: Float[torch.Tensor, "batch N_I N_I"],
    k_E: Float[torch.Tensor, "batch N_I"],
    dk_E: Float[torch.Tensor, "batch N_I"],
    parameters: SimulationParameters,
    iteration_step: int,
) -> dict[str, float]:
    dt = parameters.dt
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Calculate metrics per batch item
    recurrent_update_magnitude = torch.mean(torch.abs(dM), dim=(-2, -1)) / dt
    recurrent_percentage_change = recurrent_update_magnitude / (
        torch.mean(torch.abs(M), dim=(-2, -1)) + 1e-12
    )

    feedforward_update_magnitude = torch.mean(torch.abs(dW), dim=(-2, -1)) / dt
    feedforward_percentage_change = feedforward_update_magnitude / (
        torch.mean(torch.abs(W), dim=(-2, -1)) + 1e-12
    )

    excitatory_mass_update_magnitude = torch.mean(torch.abs(dk_E), dim=-1) / dt
    average_excitatory_mass = torch.mean(k_E, dim=-1)
    excitatory_mass_percentage_change = excitatory_mass_update_magnitude / (
        average_excitatory_mass + 1e-12
    )

    metrics_to_log = {
        "recurrent_update_magnitude": recurrent_update_magnitude,
        "recurrent_percentage_change": recurrent_percentage_change,
        "feedforward_update_magnitude": feedforward_update_magnitude,
        "feedforward_percentage_change": feedforward_percentage_change,
        "excitatory_mass_update_magnitude": excitatory_mass_update_magnitude,
        "average_excitatory_mass": average_excitatory_mass,
        "excitatory_mass_percentage_change": excitatory_mass_percentage_change,
    }

    log_dict = {}
    for name, metric_tensor in metrics_to_log.items():
        for q in quantiles:
            q_value = torch.quantile(metric_tensor, q).item()
            key = f"dynamics/{name}_q{int(q*100)}"
            log_dict[key] = q_value

    log_dict["time"] = dt * iteration_step

    return log_dict
