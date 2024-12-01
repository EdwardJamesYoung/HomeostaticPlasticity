import wandb
import torch
from linear_simulator import run_simulation, generate_conditions
from itertools import product
from typing import Dict, List, Optional
from utils import save_matrix
import gc


def run_grid_experiments(param_grid: Dict[str, List], group_name: Optional[str] = None):
    """
    Run experiments for all combinations in the parameter grid.
    Args:
        param_grid (dict): Dictionary where keys are parameter names and
                          values are lists of parameter values to try.
        group_name (str, optional): Name for the wandb group.
    """
    # Set up GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get all parameter names
    param_names = list(param_grid.keys())
    # Generate all combinations of parameters
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    # Run experiment for each combination
    for parameter_combination in combinations:
        # Create parameter dictionary for this run
        run_params = dict(zip(param_names, parameter_combination))

        # Unpack all the parameters
        N_E = run_params["N_E"]
        N_I = run_params["N_I"]
        dt = run_params["dt"]
        T = run_params["T"]
        zeta = run_params["zeta"]
        alpha = run_params["alpha"]
        tau_k = run_params["tau_k"]
        sig2 = run_params["sig2"]
        k_I = run_params["k_I"]
        tau_M = run_params["tau_M"]
        tau_W = run_params["tau_W"]
        run_number = run_params["run_number"]
        variable_eigenspectrum = run_params["variable_eigenspectrum"]

        # Initialize wandb run
        run = wandb.init(
            project="HomeostaticPlasticity",
            config=run_params,
            group=group_name if group_name is not None else "general",
        )

        try:
            # Generate initial conditions on the specified device
            initial_W, initial_M, input_eigenbasis, input_eigenspectrum = (
                generate_conditions(N_E, N_I, k_I, sig2, run_number, device)
            )

            # Handle fixed eigenspectrum case
            if variable_eigenspectrum is not True:
                input_eigenspectrum = variable_eigenspectrum * torch.ones(
                    N_E, device=device, dtype=input_eigenspectrum.dtype
                )

            # Log the eigenspectrum
            wandb.log(
                {
                    f"eigenvalue_{ii}": val.item()
                    for ii, val in enumerate(input_eigenspectrum)
                },
                commit=False,
            )

            # Compute and log the r value
            time_scale_ratio = tau_M / tau_W
            r = sig2 * N_I / torch.sum(input_eigenspectrum).item()
            r_dominance = r > time_scale_ratio

            wandb.log(
                {
                    "time_scale_ratio": time_scale_ratio,
                    "r": r,
                    "r > time scale ratio": r_dominance,
                },
                commit=False,
            )

            # Store initial matrices as artifacts
            save_matrix(initial_W, "initial_W")
            save_matrix(initial_M, "initial_M")
            save_matrix(input_eigenbasis, "input_eigenbasis")
            save_matrix(input_eigenspectrum, "input_eigenspectrum")

            # Run the simulation
            W, M = run_simulation(
                initial_W=initial_W,
                initial_M=initial_M,
                input_eigenbasis=input_eigenbasis,
                input_eigenspectrum=input_eigenspectrum,
                dt=dt,
                T=T,
                zeta=zeta,
                alpha=alpha,
                tau_k=tau_k,
                sig2=sig2,
                k_I=k_I,
                tau_M=tau_M,
                tau_W=tau_W,
                wandb_logging=True,
            )

            # Save final matrices
            save_matrix(W, "final_W")
            save_matrix(M, "final_M")

        except Exception as e:
            print(f"Error in run with parameters {run_params}: {str(e)}")
            raise e

        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Clear up other memory
            gc.collect()
            # Make sure to close the run
            run.finish()


if __name__ == "__main__":
    # Define the parameter grid
    group_name = "second_experiment"

    param_grid = {
        "N_E": [10],
        "N_I": [500],
        "dt": [0.01],
        "T": [500.0],
        "zeta": [1.0],
        "alpha": [1.0],
        "tau_k": [50, None],
        "sig2": [0.002],
        "k_I": [
            0.25,
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            7.5,
            10.0,
            12.5,
            15.0,
            17.5,
            20.0,
        ],
        "tau_M": [1.0],
        "tau_W": [10.0],
        "run_number": [0, 1, 2, 3],
        "variable_eigenspectrum": [0.5, True],
    }
    wandb.login()

    run_grid_experiments(param_grid, group_name)
