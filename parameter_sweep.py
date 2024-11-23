import wandb
from linear_simulator import run_simulation, generate_conditions
from itertools import product
from typing import Dict, List
from utils import save_matrix 

def run_grid_experiments(param_grid: Dict[str, List]):
    """
    Run experiments for all combinations in the parameter grid.
    
    Args:
        param_grid (dict): Dictionary where keys are parameter names and 
                          values are lists of parameter values to try.
    """
    # Get all parameter names
    param_names = list(param_grid.keys())
    
    # Generate all combinations of parameters
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    # Run experiment for each combination
    for parameter_combindation in combinations:
        # Create parameter dictionary for this run
        run_params = dict(zip(param_names, parameter_combindation))

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

        run = wandb.init(
            project="HomeostaticPlasticity",
            config=run_params,
        )

        initial_W, initial_M, input_eigenbasis, input_eigenspectrum = generate_conditions(N_E, N_I, k_I, sig2, run_number)
        
        # Store these matrices as artifacts
        save_matrix(initial_W, "initial_W")
        save_matrix(initial_M, "initial_M")
        save_matrix(input_eigenbasis, "input_eigenbasis")
        save_matrix(input_eigenspectrum, "input_eigenspectrum")
        
        try:
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

            save_matrix(W, "final_W")
            save_matrix(M, "final_M")

        finally:
            # Make sure to close the run
            run.finish()


if __name__ == "__main__":
    # Define the parameter grid
    param_grid = {
        "N_E": [10],
        "N_I": [100],
        "dt": [0.01], 
        "T": [1000.0], 
        "zeta": [1.0],   
        "alpha": [1.0], 
        "tau_k": [100], 
        "sig2": [0.2], 
        "k_I": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 25.0],
        "tau_M": [1.0],  
        "tau_W": [10.0],   
        "run_number": [0, 1, 2, 3, 4]
    }
    wandb.login()
    
    run_grid_experiments(param_grid)
