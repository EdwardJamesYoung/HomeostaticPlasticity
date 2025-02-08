import numpy as np
import torch
import wandb
import scipy.stats

from jaxtyping import Float, jaxtyped
from typeguard import typechecked


@typechecked
def generate_conditions(
    N_E: int, N_I: int, k_I: float, sig2: float, run_number: int, device: torch.device
) -> tuple[
    Float[torch.Tensor, "N_I N_E"],
    Float[torch.Tensor, "N_I N_I"],
    Float[torch.Tensor, "N_E N_E"],
    Float[torch.Tensor, "N_E"],
]:
    torch.manual_seed(run_number)
    np.random.seed(run_number)

    dtype = torch.float64

    # Draw an input weight matrix at random
    initial_W = (
        torch.sqrt(torch.tensor(sig2, device=device, dtype=dtype))
        * k_I
        * torch.randn(N_I, N_E, device=device, dtype=dtype)
    )

    # Construct M to be diagonally dominant
    initial_M = torch.rand(N_I, N_I, device=device, dtype=dtype) + N_I * torch.eye(
        N_I, device=device, dtype=dtype
    )
    # Renormalise M
    initial_M = k_I * initial_M / torch.sum(initial_M, dim=1, keepdim=True)

    # For orthogonal matrix generation, we'll still use numpy and then convert
    input_eigenbasis = torch.tensor(
        scipy.stats.ortho_group.rvs(N_E), device=device, dtype=dtype
    )
    input_eigenspectrum = torch.sort(
        torch.rand(N_E, device=device, dtype=dtype), descending=True
    )[0]

    return initial_W, initial_M, input_eigenbasis, input_eigenspectrum


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
