import torch
from typing import Any
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from params import SimulationParameters

from input_generation import InputGenerator
from params import SimulationParameters
from utils import (
    circular_kde,
    circular_smooth_huber,
)


@jaxtyped(typechecker=typechecked)
def compute_tuning_curve_widths(
    rates: Float[torch.Tensor, "repeats batch N_I num_stimuli"],
    stimulus_locations: Float[torch.Tensor, "num_stimuli num_dimensions"],
) -> Float[torch.Tensor, "repeats batch N_I"]:
    num_dimensions = stimulus_locations.shape[-1]

    if num_dimensions == 1:
        return _compute_circular_width_1d(rates, stimulus_locations.squeeze(-1))
    elif num_dimensions == 2:
        return _compute_circular_width_2d(rates, stimulus_locations)
    else:
        raise ValueError(f"Unsupported number of dimensions: {num_dimensions}")


@jaxtyped(typechecker=typechecked)
def _compute_circular_width_1d(
    rates: Float[torch.Tensor, "repeats batch N_I num_stimuli"],
    stimulus_angles: Float[torch.Tensor, "num_stimuli"],
) -> Float[torch.Tensor, "repeats batch N_I"]:
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
    rates: Float[torch.Tensor, "repeats batch N_I num_stimuli"],
    stimulus_locations: Float[torch.Tensor, "num_stimuli 2"],
) -> Float[torch.Tensor, "repeats batch N_I"]:
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


@jaxtyped(typechecker=typechecked)
def curves_log(
    rates: Float[torch.Tensor, "repeats batch N_I num_stimuli"],
    input_generator: InputGenerator,
    parameters: SimulationParameters,
) -> dict[str, Float[torch.Tensor, "repeats batch num_stimuli"]]:
    N_I = parameters.N_I

    stimuli_locations = input_generator.stimuli_locations  # [num_stimuli, num_dims]

    argmax_indices = rates.argmax(
        axis=-1
    )  # [repeats, batch, N_I] (indices of max response in num_stimuli)
    argmax_stimuli = stimuli_locations[
        argmax_indices.view(-1)
    ]  # [repeats, batch, N_I, num_dims]
    argmax_stimuli = argmax_stimuli.view(
        argmax_indices.shape[0], argmax_indices.shape[1], argmax_indices.shape[2], -1
    )  # [repeats, batch, N_I, num_dims]

    max_rates = rates.max(axis=-1)[0]  # [repeats, batch, N_I]

    bw_multiplier = N_I ** (-0.2)
    max_rate_range = max_rates.max() - max_rates.min()

    stimuli_locations_expanded = stimuli_locations.unsqueeze(0).unsqueeze(
        0
    )  # [1, 1, num_stimuli, num_dims]

    gains = circular_smooth_huber(
        argmax_stimuli,
        max_rates,
        stimuli_locations_expanded,  # [1, 1, num_stimuli, num_dims]
        bw=bw_multiplier * 0.4,
        delta=0.25 * max_rate_range.item(),
    )  # [repeats, batch, num_stimuli]

    # Find the width at half height of the tuning curve
    tuning_curve_widths = compute_tuning_curve_widths(
        rates, stimuli_locations
    )  # [batch, N_I]
    width_range = tuning_curve_widths.max() - tuning_curve_widths.min()

    widths = circular_smooth_huber(
        argmax_stimuli,
        tuning_curve_widths,
        stimuli_locations_expanded,
        bw=bw_multiplier * 0.4,
        delta=0.25 * width_range.item(),
    )  # [batch, num_stimuli]

    density = circular_kde(
        argmax_stimuli, stimuli_locations_expanded, bw=bw_multiplier * 0.4
    )  # [batch, num_stimuli]

    return {
        "curves/gains": gains,
        "curves/widths": widths,
        "curves/density": density,
    }


@jaxtyped(typechecker=typechecked)
def dynamics_log(
    W: Float[torch.Tensor, "repeats batch N_I N_E"],
    dW: Float[torch.Tensor, "repeats batch N_I N_E"],
    M: Float[torch.Tensor, "repeats batch N_I N_I"],
    dM: Float[torch.Tensor, "repeats batch N_I N_I"],
    k_E: Float[torch.Tensor, "repeats batch N_I"],
    dk_E: Float[torch.Tensor, "repeats batch N_I"],
    parameters: SimulationParameters,
) -> dict[str, Float[torch.Tensor, "repeats batch"]]:
    dt = parameters.dt

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
        "dynamics/recurrent_update_magnitude": recurrent_update_magnitude,
        "dynamics/recurrent_percentage_change": recurrent_percentage_change,
        "dynamics/feedforward_update_magnitude": feedforward_update_magnitude,
        "dynamics/feedforward_percentage_change": feedforward_percentage_change,
        "dynamics/excitatory_mass_update_magnitude": excitatory_mass_update_magnitude,
        "dynamics/average_excitatory_mass": average_excitatory_mass,
        "dynamics/excitatory_mass_percentage_change": excitatory_mass_percentage_change,
    }

    return metrics_to_log
