import torch
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Optional, Any

from input_generation import InputGenerator
from params import SimulationParameters

from metrics import dynamics_log, curves_log


@jaxtyped(typechecker=typechecked)
def compute_firing_rates(
    W: Float[torch.Tensor, "repeats batch_size N_I N_E"],
    M: Float[torch.Tensor, "repeats batch_size N_I N_I"],
    u: Float[torch.Tensor, "batch_size N_E num_stimuli"],
    parameters: SimulationParameters,
    v_init: Optional[Float[torch.Tensor, "repeats batch_size N_I num_stimuli"]] = None,
) -> tuple[
    Float[torch.Tensor, "repeats batch_size N_I num_stimuli"],
    Float[torch.Tensor, "repeats batch_size N_I num_stimuli"],
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
    dt_v = parameters.dt_v
    tau_v = parameters.tau_v
    activation_function = parameters.activation_function
    threshold = parameters.rate_computation_threshold
    max_iter = parameters.rate_computation_iterations

    # Initialise the input h and the voltage v
    h = torch.einsum("rbie,bes->rbis", W, u)  # [repeats, batch_size, N_I, num_stimuli]
    if v_init is not None:
        v = v_init
    else:
        v = torch.zeros_like(h)
    r = activation_function(v)
    r_dot = float("inf")
    counter = 0
    # Iterate until the rates have converged
    while r_dot > threshold and counter < max_iter:
        inhibitory_term = torch.einsum(
            "rbij,rbjs->rbis", M, r
        )  # [repeats, batch_size, N_I, num_stimuli]
        v = v + (dt_v / tau_v) * (h - inhibitory_term - v)
        r_new = activation_function(v)
        r_dot = torch.mean(torch.abs(r_new - r)) / dt_v
        r = r_new
        counter += 1

    return r, v


@jaxtyped(typechecker=typechecked)
def run_simulation(
    initial_W: Float[torch.Tensor, "repeats N_I N_E"],
    initial_M: Float[torch.Tensor, "repeats N_I N_I"],
    input_generator: InputGenerator,
    parameters: SimulationParameters,
) -> tuple[
    Float[torch.Tensor, "repeats batch_size N_I N_E"],
    Float[torch.Tensor, "repeats batch_size N_I N_I"],
    dict[str, any],
]:
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
    repeats = parameters.repeats
    N_E = parameters.N_E
    N_I = parameters.N_I
    k_I = parameters.k_I
    homeostasis = parameters.homeostasis
    homeostasis_power = parameters.homeostasis_power
    homeostasis_target = parameters.homeostasis_target
    covariance_learning = parameters.covariance_learning
    voltage_learning = parameters.voltage_learning
    T = parameters.T
    dt = parameters.dt
    tau_M = parameters.tau_M
    tau_W = parameters.tau_W
    tau_k = parameters.tau_k
    zeta = parameters.zeta
    gamma = parameters.gamma
    log_time = parameters.log_time
    device = parameters.device

    # === Perform checks on the input ===

    batch_size = input_generator.batch_size
    # Repeat the initial matrices along the batch dimension
    initial_W = initial_W.unsqueeze(1).repeat(
        1, batch_size, 1, 1
    )  # [repeats, batch_size, N_I, N_E]
    initial_M = initial_M.unsqueeze(1).repeat(
        1, batch_size, 1, 1
    )  # [repeats, batch_size, N_I, N_I]

    # Verify that the input feedforward weights has the correct shape
    assert initial_W.shape == (
        repeats,
        batch_size,
        N_I,
        N_E,
    ), f"Initial feedforward weight matrix has wrong shape. Expected {(repeats, batch_size, N_I, N_E)}, got {initial_W.shape}"
    # Verify that the initial recurrent weights has the correct shape
    assert initial_M.shape == (
        repeats,
        batch_size,
        N_I,
        N_I,
    ), f"Initial recurrent weight matrix has wrong shape. Expected {(repeats, batch_size, N_I, N_I)}, got {initial_M.shape}"
    # Verify that the initial weight matrices are non-negative
    assert torch.all(
        initial_M >= 0
    ), "Initial recurrent weight matrix has negative entries"
    assert torch.all(
        initial_W >= 0
    ), "Initial feedforward weight matrix has negative entries"

    # === Set up before the simulation ===
    W = initial_W.clone()
    M = initial_M.clone()

    # Move relevant matrices to the device
    W = W.to(device)
    M = M.to(device)

    # Initialize k_E, W_norm, and M_norm
    k_E = torch.sum(torch.abs(W), dim=-1)  # [repeats, batch_size, N_I]
    W_norm = torch.sum(torch.abs(W), dim=-1)  # [repeats, batch_size, N_I]
    M_norm = torch.sum(M, dim=-1)  # [repeats, batch_size, N_I]

    # Initialise the input, firing rates, and mean-firing rate
    stimuli = input_generator.stimuli_patterns  #  [batch_size, N_E, num_stimuli]
    probabilities = input_generator.stimuli_probabilities  # [batch_size, num_stimuli]
    excitatory_third_factor = (
        input_generator.excitatory_third_factor
    )  # [batch_size, num_stimuli]
    inhibitory_third_factor = (
        input_generator.inhibitory_third_factor
    )  # [batch_size, num_stimuli]

    # Each update step will be dt time steps long
    total_update_steps = int(T / dt)
    W_lr = dt / tau_W
    M_lr = dt / tau_M
    k_lr = dt / tau_k

    # === Initialize metrics tracking ===
    num_log_steps = int(T / log_time) + 1  # +1 for initial step

    # Compute initial metrics
    r, v = compute_firing_rates(W, M, stimuli, parameters, v_init=None)

    initial_log_dict = {}
    initial_log_dict.update(
        dynamics_log(
            W=W,
            dW=torch.zeros_like(W),
            M=M,
            dM=torch.zeros_like(M),
            k_E=k_E,
            dk_E=torch.zeros_like(k_E),
            parameters=parameters,
        )
    )
    initial_log_dict.update(
        curves_log(
            rates=r,
            input_generator=input_generator,
            parameters=parameters,
        )
    )

    # Initialize tracking tensors for all metrics (on CPU)
    metrics_over_time = {}
    for key, value in initial_log_dict.items():
        metric_shape = value.shape
        metrics_over_time[key] = torch.zeros(
            (num_log_steps,) + metric_shape, device="cpu", dtype=parameters.dtype
        )

    # Store initial values
    log_step = 0
    for key, value in initial_log_dict.items():
        metrics_over_time[key][log_step] = value.detach().cpu()

    metrics_over_time["time"] = torch.zeros(
        num_log_steps, device="cpu", dtype=parameters.dtype
    )
    metrics_over_time["time"][log_step] = dt * log_step

    log_step += 1

    for ii in range(total_update_steps):
        r, v = compute_firing_rates(
            W, M, stimuli, parameters, v_init=v
        )  # [repeats, batch, N_I, num_stimuli]

        if voltage_learning:
            learning_variable = v
        else:
            learning_variable = r

        if covariance_learning:
            learning_variable = learning_variable - torch.sum(
                learning_variable * probabilities, dim=-1
            ).unsqueeze(-1)
        else:
            learning_variable = learning_variable

        dW = torch.einsum(
            "rbij,bj,bj,bkj->rbik",
            learning_variable,
            excitatory_third_factor,
            probabilities,
            stimuli,
        )  # [repeats, batch, N_I, N_E]

        dM = torch.einsum(
            "rbij,bj,bj,rbkj->rbik",
            learning_variable,
            inhibitory_third_factor,
            probabilities,
            r,
        )  # [repeats, batch, N_I, N_I]

        # Update the excitatory mass
        if homeostasis:
            homeostatic_quantity = torch.einsum(
                "rbij,bj->rbi", r**homeostasis_power, probabilities
            )
            ratio = homeostatic_quantity / (homeostasis_target**homeostasis_power)

            new_k_E = k_E + k_lr * (1 - ratio)
            new_k_E = torch.clamp(new_k_E, min=1e-14)

        else:
            new_k_E = k_E

        # Update the norms of the weight matrices:
        W_norm = (1 - zeta * W_lr) * W_norm + (
            zeta * W_lr
        ) * new_k_E  # [repeats, batch, N_I]
        M_norm = (1 - gamma * M_lr) * M_norm + (
            gamma * M_lr
        ) * k_I  # [repeats, batch, N_I]

        # Update the weight matrices:
        new_W = W + W_lr * dW  # [repeats, batch, N_I, N_E]
        new_M = M + M_lr * dM  # [repeats, batch, N_I, N_I]

        # Rectify all the weights:
        new_M = torch.clamp(new_M, min=1e-16)  # [repeats, batch, N_I, N_I]
        new_W = torch.clamp(new_W, min=1e-16)  # [repeats, batch, N_I, N_E]

        new_W = torch.einsum(
            "rbi,rbie -> rbie",
            W_norm / (torch.sum(torch.abs(new_W), dim=-1) + 1e-12),
            new_W,
        )  # [repeats, batch, N_I, N_E]
        new_M = torch.einsum(
            "rbi,rbij->rbij", M_norm / (torch.sum(new_M, dim=-1) + 1e-12), new_M
        )  # [repeats, batch, N_I, N_I]

        # === Logging ===
        if (ii * dt) % log_time < dt and log_step < num_log_steps:
            log_dict = {}
            log_dict.update(
                dynamics_log(
                    W=W,
                    dW=new_W - W,
                    M=M,
                    dM=new_M - M,
                    k_E=new_k_E,
                    dk_E=new_k_E - k_E,
                    parameters=parameters,
                )
            )
            log_dict.update(
                curves_log(
                    rates=r,
                    input_generator=input_generator,
                    parameters=parameters,
                )
            )

            # Store metrics
            for key, value in log_dict.items():
                metrics_over_time[key][log_step] = value.detach().cpu()

            metrics_over_time["time"][log_step] = log_time * log_step

            log_step += 1

        # Update the weight matrices
        W = new_W
        M = new_M
        k_E = new_k_E

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

    return W, M, metrics_over_time
