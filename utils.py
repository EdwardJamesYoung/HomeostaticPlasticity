import numpy as np
import torch
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
import scipy.stats as stats


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
