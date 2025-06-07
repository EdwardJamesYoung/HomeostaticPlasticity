import numpy as np
import torch
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import scipy.stats as stats
from typing import Tuple


@jaxtyped(typechecker=typechecked)
def circular_discrepancy(
    curve_1: Float[np.ndarray, "num_latents"],
    curve_2: Float[np.ndarray, "num_latents"],
):
    r"""Computes the discrepancy between two curves over the stimulus space.
    Uses the l^1 distance between the curves.
    """
    # Renormalise the curves:
    normalised_curve_1 = curve_1 / curve_1.sum()
    normalised_curve_2 = curve_2 / curve_2.sum()
    # Compute the difference
    diff = np.abs(normalised_curve_1 - normalised_curve_2)
    # Integrate over [-\pi,\pi)
    discrepancy = np.sum(diff) * (2 * np.pi / len(diff))
    return discrepancy


@jaxtyped(typechecker=typechecked)
def power_law_regression(
    x: Float[np.ndarray, "num_latents"],
    y: Float[np.ndarray, "num_latents"],
    epsilon: float = 1e-10,
) -> Tuple[float, float]:
    """
    Performs log-log regression to test for power-law relationship y proportional to x to the gamma.

    Args:
        x: Independent variable (e.g., probabilities p)
        y: Dependent variable (e.g., density d)
        epsilon: Small value added before taking logs to avoid numerical issues

    Returns:
        gamma: Power-law exponent (slope in log-log space)
        r_squared: Coefficient of determination (R²)
    """
    # Add epsilon and take logs
    log_x = np.log(x + epsilon)
    log_y = np.log(y + epsilon)

    # Compute means
    mean_log_x = np.mean(log_x)
    mean_log_y = np.mean(log_y)

    # Compute covariance and variance
    cov_xy = np.mean((log_x - mean_log_x) * (log_y - mean_log_y))
    var_x = np.mean((log_x - mean_log_x) ** 2)

    # Compute slope (gamma) and intercept
    gamma = cov_xy / (var_x + epsilon)  # Add epsilon to prevent division by zero
    intercept = mean_log_y - gamma * mean_log_x

    # Compute predicted values
    log_y_pred = gamma * log_x + intercept

    # Compute R-squared
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    ss_tot = np.sum((log_y - mean_log_y) ** 2)
    r_squared = 1 - (
        ss_res / (ss_tot + epsilon)
    )  # Add epsilon to prevent division by zero

    return float(gamma), float(r_squared)


@jaxtyped(typechecker=typechecked)
def circular_kde(
    argmax_rates: Float[torch.Tensor, "N_I"],
    stimulus_space: Float[torch.Tensor, "num_latents"],
    bw: float = 0.25,
) -> Float[torch.Tensor, "num_latents"]:
    extended_data = torch.concatenate(
        [
            argmax_rates - 2 * torch.pi,
            argmax_rates,
            argmax_rates + 2 * torch.pi,
        ]
    )

    # Compute KDE on extended data
    kde = stats.gaussian_kde(extended_data.cpu(), bw_method=bw)
    y = kde(stimulus_space.cpu())
    y = y / y.sum()  # Normalise the density

    y = torch.tensor(y, device=argmax_rates.device)
    return y


@jaxtyped(typechecker=typechecked)
def circular_smooth_median(
    argmax_stimuli: Float[torch.Tensor, "N_I"],
    max_rates: Float[torch.Tensor, "N_I"],
    stimulus_space: Float[torch.Tensor, "num_latents"],
    bw: float = 0.25,
) -> Float[torch.Tensor, "num_latents"]:
    diff = torch.abs(
        stimulus_space.unsqueeze(0) - argmax_stimuli.unsqueeze(1)
    )  # [N_I, num_latents]
    diff = torch.minimum(diff, 2 * torch.pi - diff)  # [N_I, num_latents]
    gaussians = torch.exp(-0.5 * (diff / bw) ** 2) / (
        bw * np.sqrt(2 * torch.pi)
    )  # [N_I, num_latents]
    weights = gaussians / gaussians.sum(dim=0)  # [N_I, num_latents]

    sorted_indices = torch.argsort(max_rates)  # [N_I]
    sorted_rates = max_rates[sorted_indices]  # [N_I]
    sorted_weights = weights[sorted_indices, :]  # [N_I, num_latents]

    cumulative_weights = torch.cumsum(sorted_weights, dim=0)  # [N_I, num_latents]

    # This creates a mask where True indicates cumulative weight >= 0.5
    mask = cumulative_weights >= 0.5  # [N_I, num_latents]

    first_true_indices = torch.argmax(mask.to(torch.int32), dim=0)  # [num_latents]

    result = sorted_rates[first_true_indices]  # [num_latents]

    result = (result / result.sum()).squeeze()
    return result


@jaxtyped(typechecker=typechecked)
def circular_smooth_huber(
    argmax_stimuli: Float[torch.Tensor, "N_I"],
    max_rates: Float[torch.Tensor, "N_I"],
    stimulus_space: Float[torch.Tensor, "num_latents"],
    bw: float = 0.25,
    delta: float = 1.0,  # Huber threshold parameter
) -> Float[torch.Tensor, "num_latents"]:
    diff = torch.abs(
        stimulus_space.unsqueeze(0) - argmax_stimuli.unsqueeze(1)
    )  # [N_I, num_latents]
    diff = torch.minimum(diff, 2 * torch.pi - diff)  # [N_I, num_latents]
    gaussians = torch.exp(-0.5 * (diff / bw) ** 2) / (
        bw * np.sqrt(2 * torch.pi)
    )  # [N_I, num_latents]
    weights = gaussians / gaussians.sum(dim=0)  # [N_I, num_latents]

    # For each stimulus point, compute weighted center (similar to weighted mean)
    weighted_center = torch.sum(
        max_rates.unsqueeze(1) * weights, dim=0
    )  # [num_latents]

    # Compute absolute deviations from the weighted center
    abs_deviations = torch.abs(
        max_rates.unsqueeze(1) - weighted_center.unsqueeze(0)
    )  # [N_I, num_latents]

    # Apply Huber function to deviations
    huber_weights = torch.zeros_like(abs_deviations)

    # For small deviations, use squared error (behaves like mean)
    small_deviation_mask = abs_deviations <= delta
    huber_weights[small_deviation_mask] = weights[small_deviation_mask]

    # For large deviations, use absolute error (behaves like median)
    large_deviation_mask = abs_deviations > delta
    huber_weights[large_deviation_mask] = weights[large_deviation_mask] * (
        delta / abs_deviations[large_deviation_mask]
    )

    # Normalize Huber weights
    huber_weights = huber_weights / huber_weights.sum(dim=0)

    # Compute final estimate using Huber weights
    result = torch.sum(max_rates.unsqueeze(1) * huber_weights, dim=0)  # [num_latents]

    # Normalize the result
    result = (result / result.sum()).squeeze()
    return result


@jaxtyped(typechecker=typechecked)
def circular_smooth_values(
    argmax_stimuli: Float[torch.Tensor, "N_I"],
    max_rates: Float[torch.Tensor, "N_I"],
    stimulus_space: Float[torch.Tensor, "num_latents"],
    bw: float = 0.25,
) -> Float[torch.Tensor, "num_latents"]:

    diff = torch.abs(
        stimulus_space.unsqueeze(0) - argmax_stimuli.unsqueeze(1)
    )  # [N_I, num_latents]
    diff = torch.minimum(diff, 2 * torch.pi - diff)  # [N_I, num_latents]
    gaussians = torch.exp(-0.5 * (diff / bw) ** 2) / (
        bw * np.sqrt(2 * torch.pi)
    )  # [N_I, num_latents]

    weights = gaussians / gaussians.sum(axis=0)  # [N_I, num_latents]

    y = max_rates @ weights.to(dtype=max_rates.dtype)  # [num_latents]

    y = (y / y.sum()).squeeze()

    return y


def save_matrix(matrix: torch.Tensor, name: str) -> None:
    """
    Save a PyTorch tensor as a wandb artifact.

    Args:
        matrix: PyTorch tensor to save
        name: name for the artifact
    """
    # Create artifact with shape information
    artifact = wandb.Artifact(
        name=name, type="matrix", description=f"Matrix of shape {tuple(matrix.shape)}"
    )

    # Move tensor to CPU and convert to numpy if needed
    matrix_np = matrix.detach().cpu().numpy()

    # Add matrix directly to artifact
    with artifact.new_file("matrix.npy", mode="wb") as f:
        np.save(f, matrix_np)

    # Log artifact
    wandb.log_artifact(artifact)
