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
def compute_circular_distance(
    loc_1: Float[torch.Tensor, "... batch_1 num_dimensions"],
    loc_2: Float[torch.Tensor, "... batch_2 num_dimensions"],
) -> Float[torch.Tensor, "... batch_1 batch_2"]:
    diff = torch.abs(
        loc_1.unsqueeze(-2) - loc_2.unsqueeze(-3)
    )  # [..., batch_1, batch_2, num_dimensions]

    circular_diffs = torch.minimum(
        diff, 2 * torch.pi - diff
    )  # [..., batch_1, batch_2, num_dimensions]

    return torch.sqrt(torch.sum(circular_diffs**2, dim=-1))  # [..., batch_1, batch_2]


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
    data_points: Float[torch.Tensor, "#repeats #batch N_points num_dims"],
    eval_points: Float[torch.Tensor, "#repeats #batch N_eval num_dims"],
    bw: float = 0.25,
) -> Float[torch.Tensor, "#repeats #batch N_eval"]:
    # Compute circular distances between eval points and data points
    distances = compute_circular_distance(
        eval_points, data_points
    )  # [repeats, batch, N_eval, N_points]

    # Apply Gaussian kernel
    kernel_values = torch.exp(
        -0.5 * (distances / bw) ** 2
    )  # [repeats, batch, N_eval, N_points]

    # Average over data points
    kde_values = kernel_values.mean(dim=-1)  # [repeats, batch, N_eval]

    # Normalize each batch element to sum to 1
    kde_values = kde_values / kde_values.sum(dim=-1, keepdim=True)

    return kde_values


@jaxtyped(typechecker=typechecked)
def circular_smooth_huber(
    data_points: Float[torch.Tensor, "#repeats #batch N_points num_dims"],
    data_values: Float[torch.Tensor, "#repeats #batch N_points"],
    eval_points: Float[torch.Tensor, "#repeats #batch N_eval num_dims"],
    bw: float = 0.25,
    delta: float = 1.0,
    max_iter: int = 10,
    tolerance: float = 1e-6,
) -> Float[torch.Tensor, "#repeats #batch N_eval"]:

    # Compute circular distances
    distances = compute_circular_distance(
        eval_points, data_points
    )  # [batch, N_eval, N_points]

    # Compute Gaussian kernel weights (normalized for each eval point)
    kernel_weights = torch.exp(
        -0.5 * (distances / bw) ** 2
    )  # [batch, N_eval, N_points]
    kernel_weights = kernel_weights / (
        kernel_weights.sum(dim=-1, keepdim=True) + 1e-8
    )  # Normalise across each evaluation point

    # Initialize with Nadaraya-Watson estimates
    huber_estimates = torch.sum(
        kernel_weights * data_values.unsqueeze(-2), dim=-1
    )  # [repeats, batch, N_eval]

    for _ in range(max_iter):
        residuals = data_values.unsqueeze(-2) - huber_estimates.unsqueeze(
            -1
        )  # [repeats, batch, N_eval, N_points]

        # Compute Huber weights: w_i = min(1, delta / |r_i|)
        abs_residuals = torch.abs(residuals)
        huber_weights = torch.minimum(
            torch.ones_like(abs_residuals), delta / (abs_residuals + 1e-8)
        )  # [repeats, batch, N_eval, N_points]

        # Combine kernel weights and Huber weights
        combined_weights = (
            kernel_weights * huber_weights
        )  # [repeats, batch, N_eval, N_points]
        combined_weights_normalised = combined_weights / (
            combined_weights.sum(dim=-1, keepdim=True) + 1e-8
        )  # [repeats, batch, N_eval, N_points]

        # Update estimates
        new_huber_estimates = torch.sum(
            combined_weights_normalised * data_values.unsqueeze(-2), dim=-1
        )  # [repeats, batch, N_eval]

        # Check convergence
        max_change = torch.max(torch.abs(new_huber_estimates - huber_estimates))
        if max_change < tolerance:
            break

        huber_estimates = new_huber_estimates

    return huber_estimates


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


def renormalise(
    curve: Float[torch.Tensor, "num_latents"],
) -> Float[torch.Tensor, "num_latents"]:
    """Renormalise a curve to sum to 1."""
    return curve / curve.sum()
