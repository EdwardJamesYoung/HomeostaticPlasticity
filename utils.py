import numpy as np
import torch
import wandb


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
