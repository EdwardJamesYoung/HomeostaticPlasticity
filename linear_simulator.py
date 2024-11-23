import numpy as np
import scipy
from jaxtyping import Float
import wandb


def generate_conditions(
    N_E: int,
    N_I: int,
    k_I: float,
    sig2: float,
    run_number: int 
):
    np.random.seed(run_number)

    # Draw an input weight matrix at random
    initial_W = np.sqrt(sig2)*k_I*np.random.randn(N_I, N_E)

    # Construct M to be diagonally dominant, to ensure that it is invertible
    initial_M = np.random.uniform(low = 0, high = 1, size=(N_I,N_I)) + N_I*np.eye(N_I)
    # Renormalise M so that the input total weight is k_I
    initial_M =  k_I*initial_M/np.sum(initial_M, axis = 1)[:,None]

    input_eigenbasis = scipy.stats.ortho_group.rvs(N_E)
    input_eigenspectrum = np.random.uniform(low = 0, high = 1, size = N_E)
    # Reorder the eigenspectrum in descending order
    input_eigenspectrum = np.sort(input_eigenspectrum)[::-1]

    return initial_W, initial_M, input_eigenbasis, input_eigenspectrum


def run_simulation(
    initial_W: Float[np.ndarray, "N_I N_E"], # The initial feedforward weight matrix
    initial_M: Float[np.ndarray, "N_I N_I"], # The initial recurrent weight matrix
    input_eigenbasis: Float[np.ndarray, "N_E N_E"], # The eigenbasis of the input covariance
    input_eigenspectrum: Float[np.ndarray, "N_E"],
    dt:float = 0.01, # The time step
    T:float = 1000.0, # The total time of simulation
    zeta:float = 1.0, # The time constant of the excitatory decays 
    alpha:float = 1.0, # The time constant of the inhibitory decays
    tau_k:float = 100, # The time constant of the total excitatory input weight
    sig2:float = 0.2, # The target activity variance
    k_I:float = 1.5, # The total inhibitory weight
    tau_M:float = 1.0, # The time constant of the recurrent weight matrix
    tau_W:float = 10.0, # The time constant of the input weight matrix
    log_t:float = 1.0, # The frequency (in seconds) at which to log the state of the simulation
    wandb_logging: bool = False,
):
    N_I, N_E = initial_W.shape

    # Check that the input eigenbasis is orthogonal
    assert np.allclose( input_eigenbasis @ input_eigenbasis.T, np.eye(N_E) )
    
    # Construct the input covariance matrix
    input_covariance = input_eigenbasis @ np.diag(input_eigenspectrum) @ input_eigenbasis.T

    total_number_of_updates = int(T/dt)
    
    assert np.allclose( input_covariance, input_covariance.T ), "The covariance matrix is not symmetric. Got covariance matrix: {covariance}"

    W = initial_W.copy()
    M = initial_M.copy()

    # Initialise k_E 
    k_E = np.sum(np.abs(W), axis = 1)

    for ii in range(total_number_of_updates):

        # print(f"Time step: {i}")
        X = np.linalg.inv(np.eye(N_I) + M)@W
        population_covariance = X@input_covariance@X.T
        
        dM = (dt/tau_M)*population_covariance # Log this 
        dW = (dt/tau_W)*X@input_covariance # Log this

        # Compute the update to the recurrent weight matrix
        prev_M_norm = np.sum(M, axis = 1)
        new_M = M + dM # + noise_coeff*np.random.randn(N_I, N_I)*np.sqrt(dt)
        new_M = np.maximum(0, new_M)
        new_M_norm = np.sum(new_M, axis = 1) + 1e-12
        target_M_norm = (1 - alpha*dt/tau_M)*prev_M_norm + (alpha*dt/tau_M)*k_I + 1e-12
        new_M = np.diag(target_M_norm/new_M_norm)@new_M
        
        # Compute the update to the forward weight matrix
        prev_W_norm = np.sum(np.abs(W), axis = 1)
        new_W = W + dW # + noise_coeff*np.random.randn(N_I, N_E)*np.sqrt(dt)
        new_W_norm = np.sum(np.abs(new_W), axis = 1) + 1e-12
        target_W_norm = (1 - zeta*dt/tau_W)*prev_W_norm + (zeta*dt/tau_W)*k_E + 1e-12
        new_W = np.diag(target_W_norm/new_W_norm)@new_W

        # Update the total available excitatory mass
        dk_E = (dt/tau_k)*( 1 - np.diag(population_covariance)/sig2 )
        k_E = k_E + dk_E

        if wandb_logging and ii % int(log_t/dt) == 0:
            # Compute the magnitude of the recurrent update
            recurrent_update_magnitude = np.sum(np.abs(new_M - M))/(N_I * N_I * dt) 
            # Compute the magnitude of the feedforward update
            feedforward_update_magnitude = np.sum(np.abs(new_W - W))/(N_E * N_I * dt)
            # Compute the magnitude of the change in k_E
            excitatory_mass_update_magnitude = np.sum(np.abs(dk_E))/( N_I * dt)

            wandb.log({ 
                "recurrent_update_magnitude": recurrent_update_magnitude,
                "feedforward_update_magnitude": feedforward_update_magnitude,
                "excitatory_mass_update_magnitude": excitatory_mass_update_magnitude,
                "time":dt*ii,
            },
            commit=False,
            )

            population_outer = X @ input_eigenbasis @ np.diag(np.sqrt(input_eigenspectrum))
            pc_covariances = population_outer.T @ population_outer
            pc_variances = np.diag(pc_covariances)
            pc_allocations = (pc_variances/input_eigenspectrum)/(N_I * sig2)

            # Log the pc variances
            wandb.log(
                {f"variance_from_pc_{jj}":pc_variances[jj] for jj in range(N_E)},
            commit=False,
            )

            # Log the pc allocations
            wandb.log(
                {f"allocaiton_to_pc_{jj}":pc_allocations[jj] for jj in range(N_E)},
            commit=False,
            )

            # Log the relative error between the pc allocations and the uniform allocation
            total_variance = np.sum(pc_allocations)
            uniform_allocation = (total_variance/N_E)*np.ones(N_E)
            relative_error = np.sum(np.abs(pc_allocations - uniform_allocation))/np.sum(np.abs(uniform_allocation))
            wandb.log(
                {"relative_error":relative_error},
            commit=True,
            )

        # Perform the updates
        M = new_M 
        W = new_W

        # Check whether the feedforward weight matrix contains nans
        if np.isnan(W).any():
            print('NaNs in the feedforward weight matrix')
            break
    
    return W, M 





