import torch
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from input_generation import InputGenerator, DiscreteGenerator
from params import SimulationParameters
from compute_metrics import (
    compute_population_response_metrics,
    compute_discrepancies,
    dynamics_log,
    compute_firing_rates,
    compute_regressions,
    compute_firing_rates_newton,
    compute_firing_rates_momentum,
)


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

        log_dict = {}

        if wandb_logging and ii % int(mode_log_time / dt) == 0:
            population_response_metrics = compute_population_response_metrics(
                W=W, M=M, input_generator=input_generator, parameters=parameters
            )
            log_dict.update(compute_discrepancies(population_response_metrics))
            # log_dict.update(rate_mode_log(population_response_metrics, parameters))

        if wandb_logging and ii % int(dynamics_log_time / dt) == 0:
            log_dict.update(
                dynamics_log(
                    dW=new_W - W,
                    dM=new_M - M,
                    k_E=k_E,
                    new_k_E=new_k_E,
                    r=r,
                    parameters=parameters,
                    iteration_step=ii,
                )
            )

        if wandb_logging and log_dict:
            wandb.log(log_dict)

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
def deterministic_simulation(
    initial_W: Float[torch.Tensor, "N_I N_E"],
    initial_M: Float[torch.Tensor, "N_I N_I"],
    input_generator: DiscreteGenerator,
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
    homeostasis = parameters.homeostasis
    homeostasis_power = parameters.homeostasis_power
    homeostasis_target = parameters.homeostasis_target
    feedforward_covariance_learning = parameters.feedforward_covariance_learning
    recurrent_covariance_learning = parameters.recurrent_covariance_learning
    feedforward_voltage_learning = parameters.feedforward_voltage_learning
    recurrent_voltage_learning = parameters.recurrent_voltage_learning
    T = parameters.T
    tau_u = parameters.tau_u
    tau_M = parameters.tau_M
    tau_W = parameters.tau_W
    tau_k = parameters.tau_k
    zeta = parameters.zeta
    alpha = parameters.alpha
    dynamics_log_time = parameters.dynamics_log_time
    mode_log_time = parameters.mode_log_time
    wandb_logging = parameters.wandb_logging
    device = parameters.device

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
    stimuli = input_generator.stimuli_patterns  # [N_E, num_latents]
    probabilities = input_generator.stimuli_probabilities  # [num_latents]

    # Each update step will be tau_u time steps long
    total_update_steps = int(T / tau_u)
    W_lr = tau_u / tau_W
    M_lr = tau_u / tau_M
    k_lr = tau_u / tau_k

    v = None
    r = None

    for ii in range(total_update_steps):
        r, v = compute_firing_rates(
            W, M, stimuli, parameters, v_init=v
        )  # [N_I, num_latents]

        if feedforward_voltage_learning:
            feedforward_learning_variable = v
        else:
            feedforward_learning_variable = r

        if feedforward_covariance_learning:
            feedforward_learning_signal = feedforward_learning_variable - torch.sum(
                feedforward_learning_variable * probabilities, dim=1
            ).unsqueeze(1)
        else:
            feedforward_learning_signal = feedforward_learning_variable

        if recurrent_voltage_learning:
            recurrent_learning_variable = v
        else:
            recurrent_learning_variable = r

        if recurrent_covariance_learning:
            recurrent_learning_signal = recurrent_learning_variable - torch.sum(
                recurrent_learning_variable * probabilities, dim=1
            ).unsqueeze(1)
        else:
            recurrent_learning_signal = recurrent_learning_variable

        dW = torch.einsum(
            "ij,j,kj->ik", feedforward_learning_signal, probabilities, stimuli
        )  # [N_I, N_E]

        dM = torch.einsum("ij,j,kj->ik", recurrent_learning_signal, probabilities, r)

        # Update the excitatory mass
        if homeostasis:
            homeostatic_quantity = torch.einsum(
                "ij,j->i", r**homeostasis_power, probabilities
            )
            ratio = homeostatic_quantity / homeostasis_target

            new_k_E = k_E + k_lr * (1 - ratio)
            new_k_E = torch.clamp(new_k_E, min=1e-14)

        else:
            new_k_E = k_E

        # Update the norms of the weight matrices:
        W_norm = (1 - zeta * W_lr) * W_norm + (zeta * W_lr) * new_k_E
        M_norm = (1 - alpha * M_lr) * M_norm + (alpha * M_lr) * k_I

        # Update the weight matrices:
        new_W = W + W_lr * dW  # [N_I, N_E]
        new_M = M + M_lr * dM  # [N_I, N_I]

        # Rectify all the weights:
        new_M = torch.clamp(new_M, min=1e-16)  # [N_I, N_I]
        new_W = torch.clamp(new_W, min=1e-16)  # [N_I, N_E]

        # Renormalize the weight matrices
        new_W = (
            torch.diag(W_norm / (torch.sum(torch.abs(new_W), dim=1) + 1e-12)) @ new_W
        )  # [N_I, N_E]
        new_M = (
            torch.diag(M_norm / (torch.sum(new_M, dim=1) + 1e-12)) @ new_M
        )  # [N_I, N_I]

        # It's logging time!
        log_dict = {}

        if wandb_logging and ii % int(mode_log_time / tau_u) == 0:
            population_response_metrics = compute_population_response_metrics(
                rates=r, input_generator=input_generator, parameters=parameters
            )
            log_dict.update(compute_discrepancies(population_response_metrics))
            log_dict.update(compute_regressions(population_response_metrics))

            # log_dict.update(rate_mode_log(population_response_metrics, parameters))

        if wandb_logging and ii % int(dynamics_log_time / tau_u) == 0:
            log_dict.update(
                dynamics_log(
                    dW=new_W - W,
                    dM=new_M - M,
                    k_E=k_E,
                    new_k_E=new_k_E,
                    r=r,
                    probabilities=probabilities,
                    parameters=parameters,
                    iteration_step=ii,
                )
            )

        if wandb_logging and log_dict:
            wandb.log(log_dict)

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

    return W, M


# @jaxtyped(typechecker=typechecked)
# def var_mode_log(
#     W: Float[torch.Tensor, "N_I N_E"],
#     M: Float[torch.Tensor, "N_I N_I"],
#     input_generator: InputGenerator,
#     parameters: SimulationParameters,
# ) -> dict[str, Any]:
#     N_E = parameters.N_E
#     num_samples = parameters.num_samples
#     stimuli, mode_stimuli_contributions = input_generator.stimuli_batch(
#         num_samples
#     )  # [N_E, num_samples]

#     rates, _ = compute_firing_rates(
#         W,
#         M,
#         stimuli,
#         parameters=parameters,
#         threshold=1e-8,
#     )  # [N_I, num_samples]

#     mean_rate = torch.mean(rates, dim=(0, 1))  # Scalar
#     var_rate = torch.var(rates, dim=1).sum(dim=0)  # Scalar

#     mode_variance_contributions = compute_variance_contributions(
#         rates, mode_stimuli_contributions, parameters
#     )

#     mode_strengths = input_generator.mode_strengths()
#     variance_ratio = mode_variance_contributions / mode_strengths
#     mean_variance_ratio = torch.mean(variance_ratio)
#     normalised_variance_ratio = variance_ratio / mean_variance_ratio

#     variance_allocation_error = torch.mean(
#         torch.abs(variance_ratio - mean_variance_ratio) / torch.abs(mean_variance_ratio)
#     )

#     wandb.log(
#         {
#             f"steady_state/average_population_rate": mean_rate,
#             f"steady_state/average_population_variance": var_rate,
#         },
#         commit=False,
#     )

#     wandb.log(
#         {
#             f"steady_state/mode_rate_contribution_{jj}": mode_rate_contributions[jj]
#             for jj in range(N_E)
#         },
#         commit=False,
#     )

#     wandb.log(
#         {"steady_state/attunement_entropy": input_generator.attunement_entropy(W)},
#         commit=False,
#     )

#     wandb.log(
#         {f"steady_state/rate_ratio_{jj}": rate_ratio[jj] for jj in range(N_E)},
#         commit=False,
#     )
#     wandb.log(
#         {f"steady_state/variance_ratio_{jj}": variance_ratio[jj] for jj in range(N_E)},
#         commit=False,
#     )
#     wandb.log(
#         {
#             f"steady_state/normalised_rate_ratio_{jj}": normalised_rate_ratio[jj]
#             for jj in range(N_E)
#         },
#         commit=False,
#     )
#     wandb.log(
#         {
#             f"steady_state/normalised_variance_ratio_{jj}": normalised_variance_ratio[
#                 jj
#             ]
#             for jj in range(N_E)
#         },
#         commit=False,
#     )
#     wandb.log(
#         {
#             "steady_state/allocation_error/rate": rate_allocation_error,
#             "steady_state/allocation_error/variance": variance_allocation_error,
#         },
#         commit=False,
#     )


@jaxtyped(typechecker=typechecked)
def generate_initial_weights(parameters: SimulationParameters) -> tuple[
    Float[torch.Tensor, "N_I N_E"],
    Float[torch.Tensor, "N_I N_I"],
]:
    # Unpack parameters
    N_E = parameters.N_E
    N_I = parameters.N_I
    k_I = parameters.k_I
    omega = parameters.omega
    initial_feedforward_weight_scaling = parameters.initial_feedforward_weight_scaling
    dtype = parameters.dtype
    device = parameters.device
    random_seed = parameters.random_seed

    torch.manual_seed(random_seed)

    # TODO: fix this shit
    # if homeostasis_type == "rate":
    #     k_E = (
    #         N_E
    #         * torch.sqrt(torch.tensor(omega / N_I))
    #         * (
    #             (k_I * homeostasis_target)
    #             + activation_function.inverse(homeostasis_target)
    #         )
    #         / (torch.sqrt(torch.tensor(homeostasis_target)))
    #     )
    # elif homeostasis_type == "variance" or homeostasis_type == "second_moment":

    k_E = N_E * k_I * torch.sqrt(torch.tensor(omega / N_I))
    k_E = initial_feedforward_weight_scaling * k_E

    # Draw an input weight matrix at random
    initial_W = torch.randn(N_I, N_E, device=device, dtype=dtype)

    # Normalise the sum of each row to be k_E
    initial_W = k_E * initial_W / torch.sum(torch.abs(initial_W), dim=1, keepdim=True)

    # Construct M to be strongly diagonal
    initial_M = torch.rand(N_I, N_I, device=device, dtype=dtype) + (
        N_I / 2
    ) * torch.eye(N_I, device=device, dtype=dtype)
    # Renormalise M
    initial_M = k_I * initial_M / torch.sum(initial_M, dim=1, keepdim=True)

    return initial_W, initial_M


# @jaxtyped(typechecker=typechecked)
# def compute_variance_contributions(
#     r: Float[torch.Tensor, "N_I num_samples"],
#     q: Float[torch.Tensor, "N_E num_samples"],
#     parameters: SimulationParameters,
# ) -> Float[torch.Tensor, "N_E"]:

#     num_samples = parameters.num_samples

#     # Center the data by subtracting means
#     r_res = r - r.mean(dim=1, keepdim=True)  # [N_I, num_samples]
#     q_res = q - q.mean(dim=1, keepdim=True)  # [N_E, num_samples]

#     # [N_I, num_samples] @ [num_samples, N_E] -> [N_I, N_E]
#     rq_cov = (r_res @ q_res.T) / (num_samples)  # [N_I, N_E]

#     rq_cov_squared = torch.sum(rq_cov**2, dim=0)  # [N_E]
#     q_var = torch.diag(q_res @ q_res.T) / (num_samples)  # [N_E]

#     mode_variance_contributions = rq_cov_squared / (q_var + 1e-16)  # [N_E]

#     return mode_variance_contributions
