import torch
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from params import LinearParameters
from metrics import log_variables, spectrum_differences


@jaxtyped(typechecker=typechecked)
def linear_simulation(
    initial_W: Float[torch.Tensor, "batch N_I N_E"],
    initial_M: Float[torch.Tensor, "batch N_I N_I"],
    spectrum: Float[torch.Tensor, "batch N_E"],
    basis: Float[torch.Tensor, "batch N_E N_E"],
    parameters: LinearParameters,
) -> tuple[
    Float[torch.Tensor, "batch N_I N_E"],
    Float[torch.Tensor, "batch N_I N_I"],
    dict[str, torch.Tensor],
]:
    # Unpack from parameters
    batch_size = parameters.batch_size
    N_E = parameters.N_E
    N_I = parameters.N_I
    k_I = parameters.k_I
    homeostasis = parameters.homeostasis
    target_variance = parameters.target_variance
    T = parameters.T
    dt = parameters.dt
    log_time = parameters.log_time
    tau_M = parameters.tau_M
    tau_W = parameters.tau_W
    tau_k = parameters.tau_k
    zeta = parameters.zeta
    alpha = parameters.alpha
    device = parameters.device
    dtype = parameters.dtype

    # === Perform checks on the input ===

    # Verify that the input feedforward weights has the correct shape
    assert initial_W.shape == (
        batch_size,
        N_I,
        N_E,
    ), f"Initial feedforward weight matrix has wrong shape. Expected {(batch_size, N_I, N_E)}, got {initial_W.shape}"
    # Verify that the initial recurrent weights has the correct shape
    assert initial_M.shape == (
        batch_size,
        N_I,
        N_I,
    ), f"Initial recurrent weight matrix has wrong shape. Expected {(batch_size, N_I, N_I)}, got {initial_M.shape}"
    # Verify that the spectrum is all non-negative
    assert torch.all(spectrum >= 0), "Spectrum must be non-negative"
    # Verify taht the basis is an orthogonal matrix
    assert torch.allclose(
        torch.einsum("bij,bik->bjk", basis, basis),
        torch.eye(N_E, device=device, dtype=dtype)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1),
    ), "Basis must be orthogonal"

    # === Set up before the simulation ===
    W = initial_W.clone()
    M = initial_M.clone()

    # Move relevant matrices to the device
    W = W.to(device)
    M = M.to(device)

    # Initialize k_E, W_norm, and M_norm
    k_E = torch.sum(torch.abs(W), dim=-1)  # [batch_size, N_I]
    W_norm = torch.sum(torch.abs(W), dim=-1)  # [batch_size, N_I]
    M_norm = torch.sum(M, dim=-1)  # [batch_size, N_I]

    total_update_steps = int(T / dt)

    covariance_matrix = torch.einsum(
        "bij,bj,bkj->bik", basis, spectrum, basis
    )  # [batch, N_E, N_E]

    # === Initialize metrics tracking ===
    num_log_steps = int(T / log_time) + 1  # +1 for initial step

    # Compute initial metrics to get all keys
    initial_X = torch.linalg.solve(
        torch.eye(N_I, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        + M,
        W,
    )

    initial_log_dict = {}
    initial_log_dict.update(
        log_variables(
            W=W,
            dW=torch.zeros_like(W),
            M=M,
            dM=torch.zeros_like(M),
            k_E=k_E,
            dk_E=torch.zeros_like(k_E),
            parameters=parameters,
            iteration_step=0,
        )
    )
    initial_log_dict.update(
        spectrum_differences(
            X=initial_X,
            spectrum=spectrum,
            basis=basis,
        )
    )

    # Initialize tracking tensors for all metrics (on CPU)
    metrics_over_time = {}
    for key in initial_log_dict.keys():
        metrics_over_time[key] = torch.zeros(num_log_steps, device="cpu", dtype=dtype)

    # Store initial values
    log_step = 0
    for key, value in initial_log_dict.items():
        metrics_over_time[key][log_step] = torch.tensor(
            value, device="cpu", dtype=dtype
        )
    log_step += 1

    # === Perform the simulation ===

    for ii in range(1, total_update_steps + 1):
        # Compute effective input matrix, X = [I + M]^{-1} W
        X = torch.linalg.solve(
            torch.eye(N_I, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            + M,
            W,
        )  # [batch, N_I, N_E]

        # Compute the Hebbian feedforward signal, X C
        dW = torch.einsum("bij,bjk->bik", X, covariance_matrix)  # [batch, N_I, N_E]

        # Compute the recurrent learning signal, X C X^T
        population_covariance = torch.einsum("bij,bkj->bik", dW, X)  # [batch, N_I, N_I]

        # Update the weight matrices:
        new_W = W + (dt / tau_W) * dW  # [batch, N_I, N_E]
        new_M = M + (dt / tau_M) * population_covariance  # [batch, N_I, N_I]

        # Clip the recurrent weight matrix to be non-negative
        new_M = torch.clamp(new_M, min=0.0)

        # Update the norms of the weight matrices:
        W_norm = (1 - zeta * dt / tau_W) * W_norm + (
            zeta * dt / tau_W
        ) * k_E  # [batch, N_I]
        M_norm = (1 - alpha * dt / tau_M) * M_norm + (
            alpha * dt / tau_M
        ) * k_I  # [batch, N_I]

        # Renormalize the weight matrices
        new_W = torch.einsum(
            "bi,bie -> bie",
            W_norm / (torch.sum(torch.abs(new_W), dim=-1) + 1e-12),
            new_W,
        )  # [batch, N_I, N_E]
        new_M = torch.einsum(
            "bi,bij->bij", M_norm / (torch.sum(new_M, dim=-1) + 1e-12), new_M
        )  # [batch, N_I, N_I]

        # Update the excitatory mass
        if homeostasis:
            variances = population_covariance.diagonal(dim1=-2, dim2=-1)  # [batch, N_I]
            ratio = variances / target_variance  # [batch, N_I]

            new_k_E = k_E + (dt / tau_k) * (1 - ratio)
            new_k_E = torch.clamp(new_k_E, min=1e-12)
        else:
            new_k_E = k_E

        if (ii * dt) % log_time < dt and log_step < num_log_steps:

            log_dict = {}
            log_dict.update(
                log_variables(
                    W=W,
                    dW=new_W - W,
                    M=M,
                    dM=new_M - M,
                    k_E=new_k_E,
                    dk_E=new_k_E - k_E,
                    parameters=parameters,
                    iteration_step=ii,
                )
            )
            log_dict.update(
                spectrum_differences(
                    X=X,
                    spectrum=spectrum,
                    basis=basis,
                )
            )

            for key, value in log_dict.items():
                metrics_over_time[key][log_step] = torch.tensor(
                    value, device="cpu", dtype=dtype
                )
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

    return W, M, metrics_over_time
