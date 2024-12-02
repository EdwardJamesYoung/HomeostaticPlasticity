import torch
import scipy.stats
import numpy as np
from typing import Optional
import wandb


def generate_conditions(
    N_E: int, N_I: int, k_I: float, sig2: float, run_number: int, device: torch.device
):
    torch.manual_seed(run_number)
    np.random.seed(run_number)

    dtype = torch.float64

    # Draw an input weight matrix at random
    initial_W = (
        torch.sqrt(torch.tensor(sig2))
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


def run_simulation(
    initial_W: torch.Tensor,  # The initial feedforward weight matrix
    initial_M: torch.Tensor,  # The initial recurrent weight matrix
    input_eigenbasis: torch.Tensor,  # The eigenbasis of the input covariance
    input_eigenspectrum: torch.Tensor,  # The eigenspectrum of the input covariance
    dt: float = 0.01,
    T: float = 1000.0,
    zeta: float = 1.0,
    alpha: float = 1.0,
    tau_k: float | bool = 100,
    sig2: float = 0.2,
    k_I: float = 1.5,
    tau_M: float = 1.0,
    tau_W: float = 10.0,
    log_t: float = 1.0,
    wandb_logging: bool = False,
):
    N_I, N_E = initial_W.shape
    device = initial_W.device

    # Verify orthogonality
    assert torch.allclose(
        input_eigenbasis @ input_eigenbasis.T,
        torch.eye(N_E, device=device, dtype=input_eigenbasis.dtype),
    ), f"Input eigenbasis times its transpose is equal to {input_eigenbasis @ input_eigenbasis.T}"

    # Construct the input covariance matrix
    input_covariance = (
        input_eigenbasis @ torch.diag(input_eigenspectrum) @ input_eigenbasis.T
    )

    total_number_of_updates = int(T / dt)

    W = initial_W.clone()
    M = initial_M.clone()

    # Initialize k_E
    k_E = torch.sum(torch.abs(W), dim=1)

    eye_N_I = torch.eye(N_I, device=device)

    for ii in range(total_number_of_updates):
        # Use torch.linalg.solve instead of explicit inverse for better numerical stability
        X = torch.linalg.solve(eye_N_I + M, W)
        population_covariance = X @ input_covariance @ X.T

        dM = (dt / tau_M) * population_covariance
        dW = (dt / tau_W) * X @ input_covariance

        # Update recurrent weight matrix
        prev_M_norm = torch.sum(M, dim=1)
        new_M = M + dM
        new_M = torch.clamp(new_M, min=0)
        new_M_norm = torch.sum(new_M, dim=1) + 1e-12
        target_M_norm = (
            (1 - alpha * dt / tau_M) * prev_M_norm + (alpha * dt / tau_M) * k_I + 1e-12
        )
        new_M = torch.diag(target_M_norm / new_M_norm) @ new_M

        # Update forward weight matrix
        prev_W_norm = torch.sum(torch.abs(W), dim=1)
        new_W = W + dW
        new_W_norm = torch.sum(torch.abs(new_W), dim=1) + 1e-12
        target_W_norm = (
            (1 - zeta * dt / tau_W) * prev_W_norm + (zeta * dt / tau_W) * k_E + 1e-12
        )
        new_W = torch.diag(target_W_norm / new_W_norm) @ new_W

        if tau_k is not False:
            dk_E = (dt / tau_k) * (1 - torch.diag(population_covariance) / sig2)
            k_E = k_E + dk_E

        if wandb_logging and ii % int(log_t / dt) == 0:
            # Move tensors to CPU for logging
            recurrent_update_magnitude = torch.sum(torch.abs(new_M - M)).item() / (
                N_I * N_I * dt
            )
            feedforward_update_magnitude = torch.sum(torch.abs(new_W - W)).item() / (
                N_E * N_I * dt
            )

            wandb.log(
                {
                    "recurrent_update_magnitude": recurrent_update_magnitude,
                    "feedforward_update_magnitude": feedforward_update_magnitude,
                    "time": dt * ii,
                },
                commit=False,
            )

            if tau_k is not False:
                excitatory_mass_update_magnitude = torch.sum(torch.abs(dk_E)).item() / (
                    N_I * dt
                )
                wandb.log(
                    {
                        "excitatory_mass_update_magnitude": excitatory_mass_update_magnitude
                    },
                    commit=False,
                )

            population_outer = (
                X @ input_eigenbasis @ torch.diag(torch.sqrt(input_eigenspectrum))
            )
            pc_covariances = population_outer.T @ population_outer
            pc_variances = torch.diag(pc_covariances)
            pc_allocations = (pc_variances / input_eigenspectrum) / (N_I * sig2)

            # Convert to CPU for logging
            pc_variances_cpu = pc_variances.cpu()
            pc_allocations_cpu = pc_allocations.cpu()

            wandb.log(
                {
                    f"variance_from_pc_{jj}": pc_variances_cpu[jj].item()
                    for jj in range(N_E)
                },
                commit=False,
            )

            wandb.log(
                {
                    f"allocaiton_to_pc_{jj}": pc_allocations_cpu[jj].item()
                    for jj in range(N_E)
                },
                commit=False,
            )

            total_allocation = torch.sum(pc_allocations).item()
            uniform_allocation = (total_allocation / N_E) * torch.ones(
                N_E, device=device
            )
            relative_allocation_error = torch.sum(
                torch.abs(pc_allocations - uniform_allocation) / uniform_allocation
            ).item()
            total_variance = torch.sum(pc_variances).item()
            relative_variance_error = torch.sum(
                torch.abs(pc_variances - input_eigenspectrum) / input_eigenspectrum
            ).item()

            wandb.log(
                {
                    "total_allocation": total_allocation,
                    "relative_allocation_error": relative_allocation_error,
                    "total_variance": total_variance,
                    "relative_variance_error": relative_variance_error,
                },
                commit=False,
            )

            # Eigencircuit metrics
            W_projected = W @ input_eigenbasis
            W_projected_abs = torch.abs(W_projected)
            W_projected_norm = W_projected_abs / torch.sum(
                W_projected_abs, dim=1, keepdim=True
            )
            W_entropy = -torch.sum(
                W_projected_norm * torch.log(W_projected_norm), dim=1
            )
            average_entropy = torch.mean(W_entropy).item()

            wandb.log({"average_attunement_entropy": average_entropy}, commit=True)

        M = new_M
        W = new_W

        if torch.isnan(W).any():
            print("NaNs in the feedforward weight matrix")
            break

    return W, M
