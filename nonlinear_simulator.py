import torch
from torch import Tensor
from numpy import ndarray
import scipy.stats
import numpy as np
from typing import Optional, Callable
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from dataclasses import dataclass

from nonlinearities import NonLinearity, RectifiedQuadratic


@dataclass
class SimulationParameters:
    N_E: int = 10
    N_I: int = 100
    k_I: float = 10.0
    target_rate: float = 1.0
    omega: float = 1.0
    T: float = 120000.0
    dt: float = 0.05
    tau_v: float = 1.0
    tau_u: float = 5.0
    tau_M: float = 50.0
    tau_W: float = 250.0
    tau_k: float = 500.0
    zeta: float = 1.0
    alpha: float = 1.0
    nonlinearity_name: str = "rectified_quadratic"
    nonlinearity: NonLinearity = RectifiedQuadratic()
    dynamics_log_time: float = 10.0
    mode_log_time: float = 50.0
    num_samples: int = 10000
    wandb_logging: bool = False
    log_neuron_rate: bool = False
    random_seed: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float64


@jaxtyped(typechecker=typechecked)
def run_simulation(
    initial_W: Float[torch.Tensor, "N_I N_E"],
    initial_M: Float[torch.Tensor, "N_I N_I"],
    input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
    input_eigenspectrum: Float[torch.Tensor, "N_E"],
    parameters: SimulationParameters,
):
    """
    Simulates non-linear dynamics of a neural network.

    Args:
        initial_W (torch.Tensor): The initial input feedforward weight matrix.
        initial_M (torch.Tensor): The initial recurrent weight matrix.
        input_eigenspectrum (torch.Tensor): The eigenvalues of the input covariance matrix.
        input_eigenbasis (torch.Tensor): The eigenvectors of the input covariance matrix.
        non_linearity (Callable): The non-linearity of the neural network.
        parameters (SimulationParameters): The parameters of the simulation.

    Returns:
        _type_: _description_
    """
    # Unpack from parameters
    N_E = parameters.N_E
    N_I = parameters.N_I
    k_I = parameters.k_I
    target_rate = parameters.target_rate
    T = parameters.T
    dt = parameters.dt
    tau_u = parameters.tau_u
    tau_v = parameters.tau_v
    tau_M = parameters.tau_M
    tau_W = parameters.tau_W
    tau_k = parameters.tau_k
    zeta = parameters.zeta
    alpha = parameters.alpha
    non_linearity = parameters.nonlinearity
    dynamics_log_time = parameters.dynamics_log_time
    mode_log_time = parameters.mode_log_time
    wandb_logging = parameters.wandb_logging
    log_neuron_rate = parameters.log_neuron_rate
    device = parameters.device
    dtype = parameters.dtype

    # === Perform checks on the input ===

    assert (
        mode_log_time % dynamics_log_time == 0
    ), f"mode_log_time must be a multiple of dynamics_log_time for fiddly annoying reasons. Got mode log time {mode_log_time} and dynamics log time {dynamics_log_time}"

    # Verify that the input feedforward weights has the correct shape
    assert initial_W.shape == (
        N_I,
        N_E,
    ), f"Initial feedforward weight matrix has wrong shape. Expected {(N_I, N_E)}, got {initial_W.shape}"
    # Verify that the initial recurrent weights has the correct shape
    assert initial_M.shape == (
        N_I,
        N_I,
    ), f"Initial recurrent weight matrix has wrong shape. Expected {(N_I, N_I)}, got {initial_M.shape}"
    assert torch.allclose(
        input_eigenbasis @ input_eigenbasis.T,
        torch.eye(N_E, device=device, dtype=dtype),
    ), "Input eigenbasis is not orthogonal"
    # Verify that the input eigenspectrum is positive
    assert torch.all(
        input_eigenspectrum >= 0
    ), "Input eigenspectrum has negative entries"
    # Verify that the initial recurrent weight matrix is non-negative
    assert torch.all(
        initial_M >= 0
    ), "Initial recurrent weight matrix has negative entries"

    # === Set up before the simulation ===
    total_number_of_timesteps = int(T / dt)

    W = initial_W.clone()
    M = initial_M.clone()
    covariance_sqrt = input_eigenbasis @ torch.diag(torch.sqrt(input_eigenspectrum))

    # Move relevant matrices to the device
    W = W.to(device)
    M = M.to(device)
    covariance_sqrt.to(device)

    # Initialize k_E, W_norm, and M_norm
    k_E = torch.sum(torch.abs(W), dim=1)
    W_norm = torch.sum(torch.abs(W), dim=1)
    M_norm = torch.sum(M, dim=1)

    noise_amplitude = torch.sqrt(
        torch.tensor(2.0 * dt / tau_u, device=device, dtype=dtype)
    )

    # Initialise u as a random draw from a normal distribution with covariance input_covariance
    u = covariance_sqrt @ torch.randn((N_E, 1), device=device, dtype=dtype)

    # Initialise the activities at steady-state
    r, v = compute_firing_rates(W, M, u, parameters)
    # Initialise a running average firing rate.
    r_bar = r.clone()

    # === Run the simulation ===
    for ii in range(total_number_of_timesteps):
        dB = torch.randn((N_E, 1), device=device, dtype=dtype)
        colored_noise = noise_amplitude * covariance_sqrt @ dB
        u = (1 - dt / tau_u) * u + colored_noise

        # Update the neural activity
        v = v + (dt / tau_v) * (W @ u - M @ r - v)
        r = non_linearity(v)

        # Update the running average of the firing rates:
        r_bar = (1 - dt / tau_W) * r_bar + (dt / tau_W) * r

        # Update the excitatory mass
        if tau_k is not False:
            new_k_E = k_E + (dt / tau_k) * (1 - r.squeeze(-1) / target_rate)
            new_k_E = torch.clamp(new_k_E, min=1e-12)
        else:
            new_k_E = k_E

        # Update the norms
        W_norm = (1 - zeta * dt / tau_W) * W_norm + (zeta * dt / tau_W) * k_E
        M_norm = (1 - alpha * dt / tau_M) * M_norm + (alpha * dt / tau_M) * k_I

        # Define the residual activity
        r_res = r - r_bar

        # Update weight matrices
        new_W = W + (dt / tau_W) * r_res @ u.T
        new_M = M + (dt / tau_M) * r_res @ r_res.T

        # Rectify the recurrent weights
        new_M = torch.clamp(new_M, min=0)

        # Renormalize the weight matrices
        new_W = (
            torch.diag(W_norm / (torch.sum(torch.abs(new_W), dim=1) + 1e-12)) @ new_W
        )
        new_M = torch.diag(M_norm / (torch.sum(new_M, dim=1) + 1e-12)) @ new_M

        # === Mode logging ===
        if wandb_logging and ii % int(mode_log_time / dt) == 0:
            mode_rate_contributions = compute_mode_contributions(
                new_W,
                new_M,
                input_eigenbasis,
                input_eigenspectrum,
                parameters,
            )

            mean_population_mode_contribution = torch.sum(
                mode_rate_contributions, dim=0
            )
            mean_neuron_steady_state_rate = torch.sum(mode_rate_contributions, dim=1)
            allocation_ratio = mean_population_mode_contribution / input_eigenspectrum
            total_population_mode_contribution = torch.sum(
                mean_population_mode_contribution
            )
            normalised_mean_population_mode_contribution = (
                mean_population_mode_contribution / total_population_mode_contribution
            )
            normalised_allocation_ratio = (
                normalised_mean_population_mode_contribution / input_eigenspectrum
            )

            mean_allocation = torch.mean(allocation_ratio)
            # Compute the mean difference to the mean
            allocation_error = torch.mean(
                torch.abs(allocation_ratio - mean_allocation) / mean_allocation
            )

            mean_average_population_steady_state_rate = torch.mean(
                mean_neuron_steady_state_rate
            )
            if log_neuron_rate:
                wandb.log(
                    {
                        f"stimulus_averaged/steady_state/neuron_rate_{jj}": mean_neuron_steady_state_rate[
                            jj
                        ]
                        for jj in range(N_I)
                    },
                    commit=False,
                )

            wandb.log(
                {
                    f"stimulus_averaged/steady_state/population_mode_contribution_{jj}": mean_population_mode_contribution[
                        jj
                    ]
                    for jj in range(N_E)
                },
                commit=False,
            )

            wandb.log(
                {
                    f"stimulus_averaged/steady_state/allocation_ratio_{jj}": allocation_ratio[
                        jj
                    ]
                    for jj in range(N_E)
                },
                commit=False,
            )

            wandb.log(
                {
                    f"stimulus_averaged/steady_state/normalised_allocation_ratio_{jj}": normalised_allocation_ratio[
                        jj
                    ]
                    for jj in range(N_E)
                },
                commit=False,
            )

            wandb.log(
                {
                    "stimulus_averaged/steady_state/mean_allocation": mean_allocation,
                    "stimulus_averaged/steady_state/allocation_error": allocation_error,
                },
                commit=False,
            )

            wandb.log(
                {
                    f"stimulus_averaged/steady_state/average_neuron_rate": mean_average_population_steady_state_rate
                },
                commit=False,
            )

        # === Dynamics logging ===
        if wandb_logging and ii % int(dynamics_log_time / dt) == 0:
            recurrent_update_magnitude = torch.sum(torch.abs(new_M - M)).item() / (
                N_I * N_I * dt
            )
            feedforward_update_magnitude = torch.sum(torch.abs(new_W - W)).item() / (
                N_E * N_I * dt
            )

            wandb.log(
                {
                    "dynamics/recurrent_update_magnitude": recurrent_update_magnitude,
                    "dynamics/feedforward_update_magnitude": feedforward_update_magnitude,
                    "time": dt * ii,
                },
                commit=False,
            )

            if tau_k is not False:
                excitatory_mass_update_magnitude = torch.sum(
                    torch.abs(new_k_E - k_E)
                ).item() / (N_I * dt)
                wandb.log(
                    {
                        "dynamics/excitatory_mass_update_magnitude": excitatory_mass_update_magnitude
                    },
                    commit=False,
                )
            if log_neuron_rate:
                wandb.log(
                    {f"dynamics/neuron_rate_{ii}": r[ii].item() for ii in range(N_I)},
                    commit=False,
                )
                wandb.log(
                    {
                        f"dynamics/excitatory_mass_{ii}": new_k_E[ii].item()
                        for ii in range(N_I)
                    },
                    commit=False,
                )

            wandb.log(
                {
                    "dynamics/average_excitatory_mass": torch.mean(new_k_E).item(),
                    "dynamics/average_neuron_rate": torch.mean(r).item(),
                },
                commit=True,
            )

        # Update the weight matrices
        W = new_W
        M = new_M
        k_E = new_k_E

        if torch.isnan(W).any():
            print("NaNs in the feedforward weight matrix")
            break

    return W, M


@jaxtyped(typechecker=typechecked)
def generate_conditions(parameters: SimulationParameters) -> tuple[
    Float[torch.Tensor, "N_I N_E"],
    Float[torch.Tensor, "N_I N_I"],
    Float[torch.Tensor, "N_E N_E"],
    Float[torch.Tensor, "N_E"],
]:
    # Unpack parameters
    N_E = parameters.N_E
    N_I = parameters.N_I
    k_I = parameters.k_I
    target_rate = parameters.target_rate
    nonlinearity = parameters.nonlinearity
    omega = parameters.omega
    dtype = parameters.dtype
    device = parameters.device
    random_seed = parameters.random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    k_E = (
        N_E
        * torch.sqrt(torch.tensor(omega / N_I))
        * ((k_I * target_rate) + nonlinearity.inverse(target_rate))
        / (torch.sqrt(torch.tensor(target_rate)))
    )

    # Draw an input weight matrix at random
    initial_W = torch.randn(N_I, N_E, device=device, dtype=dtype)

    # Normalise the sum of each row to be k_E
    initial_W = k_E * initial_W / torch.sum(torch.abs(initial_W), dim=1, keepdim=True)

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

    spectrum_multiplier = (target_rate * N_I) / (omega * N_E)

    input_eigenspectrum = (
        2 * torch.sort(torch.rand(N_E, device=device, dtype=dtype), descending=True)[0]
    )
    input_eigenspectrum = spectrum_multiplier * input_eigenspectrum

    return initial_W, initial_M, input_eigenbasis, input_eigenspectrum


@jaxtyped(typechecker=typechecked)
def compute_firing_rates(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    u: Float[torch.Tensor, "N_E numstim"],
    parameters: SimulationParameters,
    threshold=1e-8,
    max_iter=100000,
):
    """
    Compute the firing rates of a neural network given the input and weight matrices.

    Args:
        W (torch.Tensor): The feedforward weight matrix.
        M (torch.Tensor): The recurrent weight matrix.
        u (torch.Tensor): The input to the network.
        non_linearity (Callable): The non-linearity of the neural network.

    Returns:
        torch.Tensor: The firing rates of the network.
    """
    # Unpack from parameters
    dt = 0.1 * parameters.dt
    tau_v = parameters.tau_v
    non_linearity = parameters.nonlinearity

    # Initialise the input h and the voltage v
    h = W @ u
    v = torch.zeros_like(h)
    r = non_linearity(v)
    r_dot = float("inf")
    counter = 0
    # Iterate until the rates have converged
    while r_dot > threshold and counter < max_iter:
        v = v + (dt / tau_v) * (h - M @ r - v)
        r_new = non_linearity(v)
        r_dot = torch.mean(torch.abs(r_new - r)) / dt
        r = r_new
        counter += 1

    if counter == max_iter:
        wandb.alert(
            title="Firing rate computation did not converge",
            text="The firing rate computation did not converge within the maximum number of iterations",
            level=wandb.AlertLevel.WARN,
        )

    return r, v


@jaxtyped(typechecker=typechecked)
def compute_mode_contributions(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
    input_eigenspectrum: Float[torch.Tensor, "N_E"],
    parameters: SimulationParameters,
) -> Float[torch.Tensor, "N_I N_E"]:
    # Unpack from parameters
    device = parameters.device
    dtype = parameters.dtype
    num_modes = parameters.N_E
    num_neurons = parameters.N_I
    num_samples = parameters.num_samples

    assert W.shape == (
        num_neurons,
        num_modes,
    ), f"Expected W to have shape {(num_neurons, num_modes)}, got {W.shape}"

    covariance_sqrt = input_eigenbasis @ torch.diag(torch.sqrt(input_eigenspectrum))
    stimuli = covariance_sqrt @ torch.randn(
        (num_modes, num_samples), device=device, dtype=dtype
    )

    stimuli_norms = torch.norm(stimuli, dim=0, keepdim=True)
    normalised_stimuli = stimuli / stimuli_norms

    mode_stimulus_contributions = (
        torch.matmul(input_eigenbasis.T, normalised_stimuli) ** 2
    )

    # Verify that contributions sum to 1 for each stimulus
    assert torch.allclose(
        torch.sum(mode_stimulus_contributions, dim=0),
        torch.ones(num_samples, device=device, dtype=dtype),
    )

    # Compute firing rates for all stimuli in parallel
    firing_rates, _ = compute_firing_rates(
        W,
        M,
        stimuli,
        parameters=parameters,
        threshold=1e-8,
    )

    mode_rate_contributions = (
        torch.matmul(firing_rates, mode_stimulus_contributions.T) / num_samples
    )
    return mode_rate_contributions
