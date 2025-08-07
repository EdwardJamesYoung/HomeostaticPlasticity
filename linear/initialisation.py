import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Tuple
import numpy as np

from params import LinearParameters


@jaxtyped(typechecker=typechecked)
def generate_initial_weights(parameters: LinearParameters) -> tuple[
    Float[torch.Tensor, "{parameters.batch_size} {parameters.N_I} {parameters.N_E}"],
    Float[torch.Tensor, "{parameters.batch_size} {parameters.N_I} {parameters.N_I}"],
]:
    # Unpack parameters
    batch_size = parameters.batch_size
    N_E = parameters.N_E
    N_I = parameters.N_I
    k_I = parameters.k_I
    target_variance = parameters.target_variance
    dtype = parameters.dtype
    device = parameters.device
    random_seed = parameters.random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    k_E = (k_I + 1) * np.sqrt(2 * target_variance * N_E / torch.pi)  # [scalar]

    # Draw an input weight matrix at random
    initial_W = torch.randn(
        batch_size, N_I, N_E, device=device, dtype=dtype
    )  # [batch, N_I, N_E]

    # Normalise the sum of each row to be k_E
    initial_W = (
        k_E * initial_W / torch.sum(torch.abs(initial_W), dim=-1, keepdim=True)
    )  # [batch, N_I, N_E]

    # Construct M to be strongly diagonal
    initial_M = torch.rand(
        batch_size, N_I, N_I, device=device, dtype=dtype
    ) + N_I * torch.eye(N_I, device=device, dtype=dtype).unsqueeze(0).repeat(
        batch_size, 1, 1
    )

    # Renormalise M
    initial_M = k_I * initial_M / torch.sum(initial_M, dim=-1, keepdim=True)

    return initial_W, initial_M


def generate_conditions(
    parameters: LinearParameters,
) -> Tuple[
    Float[torch.Tensor, "{parameters.batch_size} {parameters.N_E}"],
    Float[torch.Tensor, "{parameters.batch_size} {parameters.N_E} {parameters.N_E}"],
]:
    # Unpack parameters
    batch_size = parameters.batch_size
    N_E = parameters.N_E
    device = parameters.device
    dtype = parameters.dtype
    random_seed = parameters.random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Generate random matrices
    random_matrices = torch.randn(batch_size, N_E, N_E, device=device, dtype=dtype)

    # Apply QR decomposition
    Q, R = torch.linalg.qr(random_matrices)

    # Correct for the sign ambiguity to get Haar-distributed matrices
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    basis = Q * signs.unsqueeze(-2)  # [batch, N_E, N_E]

    # Sample N_E i.i.d. Exp(1) random variables
    Z = (
        torch.distributions.exponential.Exponential(1.0)
        .sample((batch_size, N_E))
        .to(device=device)
    )
    # Take cumulative sum to get Z_1, Z_1+Z_2, Z_1+Z_2+Z_3, etc.
    spectrum = torch.cumsum(Z, dim=-1)  # [batch, N_E]

    # Normalise the spectrum to have unit sum
    spectrum, _ = torch.sort(spectrum, dim=-1, descending=True)
    spectrum = spectrum / torch.sum(spectrum, dim=-1, keepdim=True)  # [batch_size, N_E]

    return spectrum, basis
