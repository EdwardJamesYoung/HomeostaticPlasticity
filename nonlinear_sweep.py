import wandb
import torch
import importlib
from nonlinear_simulator import (
    SimulationParameters,
    generate_conditions,
    run_simulation,
)
from utils import save_matrix
from itertools import product
from typing import Dict, List, Optional
import gc

from nonlinearities import RectifiedQuadratic, RectifiedLinear, RectifiedCubic

nonlinearity_dict = {
    "rectified_quadratic": RectifiedQuadratic(),
    "rectified_linear": RectifiedLinear(),
    "rectified_cubic": RectifiedCubic(),
}


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

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    # Run experiment for each combination
    for parameter_combination in combinations:
        # Create parameter dictionary for this run
        run_params = dict(zip(param_names, parameter_combination))

        parameters = SimulationParameters(**run_params)

        # Initialize wandb run
        run = wandb.init(
            project="HomeostaticPlasticity",
            config=parameters.__dict__,
            group=group_name if group_name is not None else "general",
        )

        try:
            if "nonlinearity_name" in run_params.keys():
                parameters.nonlinearity = nonlinearity_dict[
                    run_params["nonlinearity_name"]
                ]

            initial_W, initial_M, input_eigenbasis, input_eigenspectrum = (
                generate_conditions(parameters)
            )

            # Store initial matrices as artifacts
            save_matrix(initial_W, "initial_W")
            save_matrix(initial_M, "initial_M")
            save_matrix(input_eigenbasis, "input_eigenbasis")
            save_matrix(input_eigenspectrum, "input_eigenspectrum")

            wandb.run.summary.update(
                {
                    f"eigenvalue_{ii}": input_eigenspectrum[ii].item()
                    for ii in range(parameters.N_E)
                }
            )

            tau_M = parameters.tau_M
            tau_W = parameters.tau_W
            N_I = parameters.N_I
            target_rate = parameters.target_rate

            # Compute and log the r value
            time_scale_ratio = tau_M / tau_W
            omega_emp = target_rate * N_I / torch.sum(input_eigenspectrum).item()
            omega_dominance = True if omega_emp > time_scale_ratio else False

            wandb.run.summary.update(
                {
                    "time_scale_ratio": time_scale_ratio,
                    "omega_empirical": omega_emp,
                    "omega_empirical > time scale ratio": omega_dominance,
                }
            )

            W, M = run_simulation(
                initial_W,
                initial_M,
                input_eigenbasis,
                input_eigenspectrum,
                parameters,
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
    group_name = "long_nonlinear_sweep"

    param_grid = {
        "tau_k": [500.0, False],
        "k_I": [
            5.0,
            10.0,
            15.0,
            20.0,
        ],
        "target_rate": [1],
        "nonlinearity_name": ["rectified_quadratic"],
        "wandb_logging": [True],
        "random_seed": [1],
        "T": 150000.0,
    }
    wandb.login()

    run_grid_experiments(param_grid, group_name)
