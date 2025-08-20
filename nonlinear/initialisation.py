import torch
import scipy.stats
import numpy as np
from abc import ABC, abstractmethod
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Dict, Any, Tuple, Optional
from params import SimulationParameters


@jaxtyped(typechecker=typechecked)
def generate_initial_weights(parameters: SimulationParameters) -> tuple[
    Float[torch.Tensor, "repeats N_I N_E"],
    Float[torch.Tensor, "repeats N_I N_I"],
]:
    # Unpack parameters
    repeats = parameters.repeats
    N_E = parameters.N_E
    N_I = parameters.N_I
    activation_function = parameters.activation_function
    homeostasis_target = parameters.homeostasis_target
    k_I = parameters.k_I
    dtype = parameters.dtype
    device = parameters.device
    random_seed = parameters.random_seed

    torch.manual_seed(random_seed)

    k_E = (
        k_I + activation_function.inverse(homeostasis_target)
    ) / homeostasis_target  # Think I might need to multiply by N_E.

    # Draw an input weight matrix at random
    initial_W = torch.randn(repeats, N_I, N_E, device=device, dtype=dtype)
    # Take the absolute value to ensure non-negative weights
    initial_W = torch.abs(initial_W)
    # Normalise the sum of each row to be k_E
    initial_W = k_E * initial_W / torch.sum(initial_W, dim=-1, keepdim=True)

    # Construct M to be strongly diagonal
    initial_M = torch.rand(repeats, N_I, N_I, device=device, dtype=dtype) + (
        N_I
    ) * torch.eye(N_I, device=device, dtype=dtype).unsqueeze(0).repeat(repeats, 1, 1)

    # Renormalise M
    initial_M = k_I * initial_M / torch.sum(initial_M, dim=-1, keepdim=True)

    return initial_W, initial_M
