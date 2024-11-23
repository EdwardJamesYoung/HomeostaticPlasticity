import numpy as np
import wandb
import os

def save_matrix(matrix: np.ndarray, name: str) -> None:
    """
    Save a single matrix as a wandb artifact.
    
    Args:
        matrix: numpy array to save
        name: name for the artifact
    """
    # Create artifact
    artifact = wandb.Artifact(
        name=name,
        type="matrix",
        description=f"Matrix of shape {matrix.shape}"
    )
    
    # Add matrix directly to artifact
    with artifact.new_file('matrix.npy', mode='wb') as f:
        np.save(f, matrix)
    
    # Log artifact
    wandb.log_artifact(artifact)