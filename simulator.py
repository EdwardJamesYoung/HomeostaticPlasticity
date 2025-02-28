import torch
from typing import Tuple
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from input_generation import InputGenerator
from params import SimulationParameters


@jaxtyped(typechecker=typechecked)
def run_simulation(
    initial_W: Float[torch.Tensor, "N_I N_E"],
    initial_M: Float[torch.Tensor, "N_I N_I"],
    input_generator: InputGenerator,
    parameters: SimulationParameters,
):
    """
    Simulates non-linear dynamics of a neural network.

    Args:
        initial_W (torch.Tensor): The initial input feedforward weight matrix.
        initial_M (torch.Tensor): The initial recurrent weight matrix.
        input_eigenspectrum (torch.Tensor): The eigenvalues of the input covariance matrix.
        input_eigenbasis (torch.Tensor): The eigenvectors of the input covariance matrix.
        activation_function (Callable): The non-linearity of the neural network.
        parameters (SimulationParameters): The parameters of the simulation.

    Returns:
        _type_: _description_
    """
    # Unpack from parameters
    N_E = parameters.N_E
    N_I = parameters.N_I
    k_I = parameters.k_I
    target_rate = parameters.target_rate
    target_variance = parameters.target_variance
    variable_input_mass = parameters.variable_input_mass
    T = parameters.T
    dt = parameters.dt
    tau_v = parameters.tau_v
    tau_M = parameters.tau_M
    tau_W = parameters.tau_W
    tau_k = parameters.tau_k
    zeta = parameters.zeta
    alpha = parameters.alpha
    activation_function = parameters.activation_function
    covariance_learning = parameters.covariance_learning
    dynamics_log_time = parameters.dynamics_log_time
    mode_log_time = parameters.mode_log_time
    wandb_logging = parameters.wandb_logging
    device = parameters.device
    dtype = parameters.dtype
    rate_homeostasis = parameters.rate_homeostasis
    variance_homeostasis = parameters.variance_homeostasis

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
    # Verify that the initial recurrent weight matrix is non-negative
    assert torch.all(
        initial_M >= 0
    ), "Initial recurrent weight matrix has negative entries"

    # === Set up before the simulation ===
    total_number_of_timesteps = int(T / dt)

    W = initial_W.clone()
    M = initial_M.clone()

    # Move relevant matrices to the device
    W = W.to(device)
    M = M.to(device)

    # Initialize k_E, W_norm, and M_norm
    k_E = torch.sum(torch.abs(W), dim=1)
    W_norm = torch.sum(torch.abs(W), dim=1)
    M_norm = torch.sum(M, dim=1)

    # Initialise the input, firing rates, and mean-firing rate
    u = input_generator.step()
    r, v = compute_firing_rates(W, M, u, parameters)
    r_bar = r.clone()

    # === Run the simulation ===
    for ii in range(total_number_of_timesteps):

        v = v + (dt / tau_v) * (W @ u - M @ r - v)
        r = activation_function(v)
        r_bar = (1 - dt / tau_W) * r_bar + (dt / tau_W) * r

        # Update the norms of the weight matrices:
        W_norm = (1 - zeta * dt / tau_W) * W_norm + (zeta * dt / tau_W) * k_E
        M_norm = (1 - alpha * dt / tau_M) * M_norm + (alpha * dt / tau_M) * k_I

        # Update the weight matrices:
        if covariance_learning:
            r_res = r - r_bar
        else:
            r_res = r

        new_W = W + (dt / tau_W) * r_res @ u.T
        new_M = M + (dt / tau_M) * r_res @ r_res.T

        # Rectify the recurrent weights
        new_M = torch.clamp(new_M, min=1e-14)

        # Renormalize the weight matrices
        new_W = (
            torch.diag(W_norm / (torch.sum(torch.abs(new_W), dim=1) + 1e-12)) @ new_W
        )
        new_M = torch.diag(M_norm / (torch.sum(new_M, dim=1) + 1e-12)) @ new_M

        # Update the excitatory mass
        if variable_input_mass:

            if rate_homeostasis:
                ratio = r.squeeze(-1) / target_rate
            elif variance_homeostasis:
                ratio = (r - r_bar).squeeze(-1) ** 2 / target_variance
            else:
                ratio = torch.ones_like(r).squeeze(-1)

            new_k_E = k_E + (dt / tau_k) * (1 - ratio)
            new_k_E = torch.clamp(new_k_E, min=1e-12)

        else:
            new_k_E = k_E

        if wandb_logging and ii % int(mode_log_time / dt) == 0:
            mode_log(W=W, M=M, input_generator=input_generator, parameters=parameters)

        if wandb_logging and ii % int(dynamics_log_time / dt) == 0:
            dynamics_log(
                dW=new_W - W,
                dM=new_M - M,
                k_E=k_E,
                new_k_E=new_k_E,
                r=r,
                parameters=parameters,
                iteration_step=ii,
            )

        # Update the weight matrices
        W = new_W
        M = new_M
        k_E = new_k_E
        u = input_generator.step()

        if torch.isnan(W).any():
            print("NaNs in the feedforward weight matrix")
            break
        if torch.isnan(M).any():
            print("NaNs in the recurrent weight matrix")
            break
        if torch.isnan(k_E).any():
            print("NaNs in the excitatory mass")
            break
        if torch.isnan(r).any():
            print("NaNs in the firing rates")
            break

    return W, M


@jaxtyped(typechecker=typechecked)
def mode_log(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    input_generator: InputGenerator,
    parameters: SimulationParameters,
):
    N_E = parameters.N_E
    num_samples = parameters.num_samples
    stimuli, mode_stimuli_contributions = input_generator.stimuli_batch(
        num_samples
    )  # [N_E, num_samples]

    rates, _ = compute_firing_rates(
        W,
        M,
        stimuli,
        parameters=parameters,
        threshold=1e-8,
    )  # [N_I, num_samples]

    mean_rate = torch.mean(rates, dim=(0, 1))  # Scalar
    var_rate = torch.var(rates, dim=1).sum(dim=0)  # Scalar

    mode_rate_contributions, mode_variance_contributions = compute_mode_contributions(
        rates, mode_stimuli_contributions, parameters
    )

    mode_strengths = input_generator.mode_strengths()
    rate_ratio = mode_rate_contributions / mode_strengths
    variance_ratio = mode_variance_contributions / mode_strengths

    mean_rate_ratio = torch.mean(rate_ratio)
    mean_variance_ratio = torch.mean(variance_ratio)

    normalised_rate_ratio = rate_ratio / mean_rate_ratio  # Has mean 1
    normalised_variance_ratio = variance_ratio / mean_variance_ratio  # Has mean 1

    rate_allocation_error = torch.mean(
        torch.abs(rate_ratio - mean_rate_ratio) / torch.abs(mean_rate_ratio)
    )
    variance_allocation_error = torch.mean(
        torch.abs(variance_ratio - mean_variance_ratio) / torch.abs(mean_variance_ratio)
    )

    wandb.log(
        {
            f"steady_state/average_population_rate": mean_rate,
            f"steady_state/average_population_variance": var_rate,
        },
        commit=False,
    )

    wandb.log(
        {
            f"steady_state/mode_rate_contribution_{jj}": mode_rate_contributions[jj]
            for jj in range(N_E)
        },
        commit=False,
    )

    wandb.log(
        {"steady_state/attunement_entropy": input_generator.attunement_entropy(W)},
        commit=False,
    )

    wandb.log(
        {f"steady_state/rate_ratio_{jj}": rate_ratio[jj] for jj in range(N_E)},
        commit=False,
    )
    wandb.log(
        {f"steady_state/variance_ratio_{jj}": variance_ratio[jj] for jj in range(N_E)},
        commit=False,
    )
    wandb.log(
        {
            f"steady_state/normalised_rate_ratio_{jj}": normalised_rate_ratio[jj]
            for jj in range(N_E)
        },
        commit=False,
    )
    wandb.log(
        {
            f"steady_state/normalised_variance_ratio_{jj}": normalised_variance_ratio[
                jj
            ]
            for jj in range(N_E)
        },
        commit=False,
    )
    wandb.log(
        {
            "steady_state/allocation_error/rate": rate_allocation_error,
            "steady_state/allocation_error/variance": variance_allocation_error,
        },
        commit=False,
    )


def dynamics_log(
    dW: Float[torch.Tensor, "N_I N_E"],
    dM: Float[torch.Tensor, "N_I N_I"],
    k_E: Float[torch.Tensor, "N_I"],
    new_k_E: Float[torch.Tensor, "N_I"],
    r: Float[torch.Tensor, "N_I"],
    parameters: SimulationParameters,
    iteration_step: int,
):
    N_E = parameters.N_E
    N_I = parameters.N_I
    dt = parameters.dt

    recurrent_update_magnitude = torch.sum(torch.abs(dM)).item() / (N_I * N_I * dt)
    feedforward_update_magnitude = torch.sum(torch.abs(dW)).item() / (N_E * N_I * dt)
    excitatory_mass_update_magnitude = torch.sum(torch.abs(new_k_E - k_E)).item() / (
        N_I * dt
    )

    wandb.log(
        {
            "dynamics/recurrent_update_magnitude": recurrent_update_magnitude,
            "dynamics/feedforward_update_magnitude": feedforward_update_magnitude,
            "dynamics/excitatory_mass_update_magnitude": excitatory_mass_update_magnitude,
            "dynamics/average_excitatory_mass": torch.mean(new_k_E).item(),
            "dynamics/average_neuron_rate": torch.mean(r).item(),
            "time": dt * iteration_step,
        },
        commit=True,
    )


@jaxtyped(typechecker=typechecked)
def generate_initial_weights(parameters: SimulationParameters) -> tuple[
    Float[torch.Tensor, "N_I N_E"],
    Float[torch.Tensor, "N_I N_I"],
]:
    # Unpack parameters
    N_E = parameters.N_E
    N_I = parameters.N_I
    k_I = parameters.k_I
    target_rate = parameters.target_rate
    target_variance = parameters.target_variance
    activation_function = parameters.activation_function
    omega = parameters.omega
    initial_feedforward_weight_scaling = parameters.initial_feedforward_weight_scaling
    dtype = parameters.dtype
    device = parameters.device
    random_seed = parameters.random_seed

    torch.manual_seed(random_seed)

    if target_rate is not None:
        k_E = (
            N_E
            * torch.sqrt(torch.tensor(omega / N_I))
            * ((k_I * target_rate) + activation_function.inverse(target_rate))
            / (torch.sqrt(torch.tensor(target_rate)))
        )
    elif target_variance is not None:
        k_E = N_E * k_I * torch.sqrt(torch.tensor(omega / N_I))
    else:
        raise ValueError("Must specify either target rate or target variance")

    k_E = initial_feedforward_weight_scaling * k_E

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

    return initial_W, initial_M


@jaxtyped(typechecker=typechecked)
def compute_firing_rates(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    u: Float[torch.Tensor, "N_E num_stimuli"],
    parameters: SimulationParameters,
    threshold=1e-8,
    max_iter=100000,
) -> tuple[
    Float[torch.Tensor, "N_I num_stimuli"], Float[torch.Tensor, "N_I num_stimuli"]
]:
    """
    Compute the firing rates of a neural network given the input and weight matrices.

    Args:
        W (torch.Tensor): The feedforward weight matrix.
        M (torch.Tensor): The recurrent weight matrix.
        u (torch.Tensor): The input to the network.
        activation_function (Callable): The non-linearity of the neural network.

    Returns:
        torch.Tensor: The firing rates of the network.
    """
    # Unpack from parameters
    dt = 0.5 * parameters.dt
    tau_v = parameters.tau_v
    activation_function = parameters.activation_function

    # Initialise the input h and the voltage v
    h = W @ u
    v = torch.zeros_like(h)
    r = activation_function(v)
    r_dot = float("inf")
    counter = 0
    # Iterate until the rates have converged
    while r_dot > threshold and counter < max_iter:
        v = v + (dt / tau_v) * (h - M @ r - v)
        r_new = activation_function(v)
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
    r: Float[torch.Tensor, "N_I num_samples"],
    q: Float[torch.Tensor, "N_E num_samples"],
    parameters: SimulationParameters,
) -> Tuple[Float[torch.Tensor, "N_E"], Float[torch.Tensor, "N_E"]]:

    num_samples = parameters.num_samples

    # Center the data by subtracting means
    r_res = r - r.mean(dim=1, keepdim=True)  # [N_I, num_samples]
    q_res = q - q.mean(dim=1, keepdim=True)  # [N_E, num_samples]

    # [N_I, num_samples] @ [num_samples, N_E] -> [N_I, N_E]
    rq_cov = (r_res @ q_res.T) / (num_samples)  # [N_I, N_E]

    rq_cov_squared = torch.sum(rq_cov**2, dim=0)  # [N_E]
    q_var = torch.diag(q_res @ q_res.T) / (num_samples)  # [N_E]

    mode_variance_contributions = rq_cov_squared / (q_var + 1e-16)  # [N_E]

    q_squared = q**2  # [N_E, num_samples]
    q_squared_sum = q_squared.sum(dim=0, keepdim=True)  # [1, num_samples]
    q_squared_normalised = q_squared / (q_squared_sum + 1e-16)  # [N_E, num_samples]

    mode_rate_contributions = torch.sum(r @ q_squared_normalised.T, dim=0)  # [N_E]

    return mode_rate_contributions, mode_variance_contributions
