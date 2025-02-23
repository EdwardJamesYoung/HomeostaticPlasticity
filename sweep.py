import argparse
import yaml
import wandb
import torch
from typing import Dict, List, Optional, Any, Tuple
from itertools import product
import gc
from dataclasses import asdict

from params import SimulationParameters
from simulator import run_simulation, generate_initial_weights
from utils import save_matrix
from input_generation import (
    GaussianGenerator,
    LaplacianGenerator,
    CircularGenerator,
    generate_conditions,
)
from activation_functions import (
    RectifiedQuadratic,
    RectifiedLinear,
    RectifiedCubic,
    Cubic,
    Linear,
)

INPUT_GENERATOR_MAP = {
    "gaussian": GaussianGenerator,
    "laplacian": LaplacianGenerator,
    "circular": CircularGenerator,
}

ACTIVATION_FUNCTION_MAP = {
    "rectified_quadratic": RectifiedQuadratic,
    "rectified_linear": RectifiedLinear,
    "rectified_cubic": RectifiedCubic,
    "cubic": Cubic,
    "linear": Linear,
}


def load_sweep_config(config_path: str) -> Dict:
    """Load sweep configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_keys = ["param_grid", "group_name"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Configuration file must contain '{key}'")

    return config


def expand_input_config(input_config: Dict) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Expands a single input generator configuration into a list of (type, params) tuples,
    handling nested parameter sweeps if present.
    """
    generator_type = input_config["type"]

    if "param_grid" not in input_config:
        # Simple case - no parameter sweep
        return [(generator_type, {})]

    # Get the parameter grid for this generator
    param_grid = input_config["param_grid"]

    # Get all parameter names and values for the sweep
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Generate all combinations of parameter values
    combinations = list(product(*param_values))

    # Create a list of (type, params) tuples for each parameter combination
    expanded_configs = [
        (generator_type, dict(zip(param_names, combination)))
        for combination in combinations
    ]

    return expanded_configs


def expand_all_input_configs(param_grid: Dict) -> Dict:
    """
    Expands all input configurations into a format suitable for parameter sweeping.
    """
    expanded_grid = param_grid.copy()

    if "input_config" in param_grid:
        input_configs = param_grid["input_config"]

        # Expand each input configuration and combine them
        expanded_configs = []
        for config in input_configs:
            expanded_configs.extend(expand_input_config(config))

        expanded_grid["input_config"] = expanded_configs

    return expanded_grid


def create_input_generator(
    generator_type: str,
    generator_params: Dict[str, Any],
    parameters: SimulationParameters,
    input_eigenbasis: Optional[torch.Tensor] = None,
    input_eigenspectrum: Optional[torch.Tensor] = None,
) -> Any:
    """Creates an input generator instance based on type and parameters."""
    generator_class = INPUT_GENERATOR_MAP[generator_type]

    if generator_type in ["gaussian", "laplacian"]:
        if input_eigenbasis is None or input_eigenspectrum is None:
            input_eigenbasis, input_eigenspectrum = generate_conditions(parameters)
        return generator_class(
            parameters=parameters,
            input_eigenbasis=input_eigenbasis,
            input_eigenspectrum=input_eigenspectrum,
        )
    elif generator_type == "circular":
        return generator_class(parameters=parameters, **generator_params)
    else:
        raise ValueError(f"Unknown input generator type: {generator_type}")


def run_grid_experiments(
    param_grid: Dict[str, List], base_params: Dict, group_name: Optional[str] = None
):
    """Run experiments for all combinations in the parameter grid."""
    # Set up GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Expand input generator configurations
    expanded_grid = expand_all_input_configs(param_grid)

    # Get all parameter names and values
    param_names = list(expanded_grid.keys())
    param_values = list(expanded_grid.values())
    combinations = list(product(*param_values))

    print(f"Running {len(combinations)} parameter combinations")

    # Run experiment for each combination
    for parameter_combination in combinations:
        # Create parameter dictionary for this run
        run_params = dict(zip(param_names, parameter_combination))

        # Handle input generator configuration
        if "input_config" in run_params:
            generator_type, generator_params = run_params["input_config"]
            run_params["input_type"] = generator_type
            del run_params["input_config"]

        activation_function_name = run_params.get(
            "activation_function_name", "rectified_quadratic"
        )
        run_params["activation_function"] = ACTIVATION_FUNCTION_MAP[
            activation_function_name
        ]()

        # Merge with base parameters
        full_params = {**base_params, **run_params}

        # Create SimulationParameters instance
        parameters = SimulationParameters(**full_params)

        try:
            # Initialize wandb run
            run = wandb.init(
                project="HomeostaticPlasticity",
                config={
                    **parameters.to_dict(),
                    "input_generator_type": generator_type,
                    "input_generator_params": generator_params,
                },
                group=group_name if group_name is not None else "general",
            )

            # Generate input conditions if needed
            if generator_type in ["gaussian", "laplacian"]:
                input_eigenbasis, input_eigenspectrum = generate_conditions(parameters)
            else:
                input_eigenbasis = input_eigenspectrum = None

            # Create input generator
            input_generator = create_input_generator(
                generator_type,
                generator_params,
                parameters,
                input_eigenbasis,
                input_eigenspectrum,
            )

            if input_eigenbasis is not None:
                # Store matrices as artifacts
                save_matrix(input_eigenbasis, "input_eigenbasis")
                save_matrix(input_eigenspectrum, "input_eigenspectrum")

                wandb.run.summary.update(
                    {
                        f"eigenvalue_{ii}": input_eigenspectrum[ii].item()
                        for ii in range(parameters.N_E)
                    }
                )

            initial_W, initial_M = generate_initial_weights(parameters)

            # Save initial matrices
            save_matrix(initial_W, "initial_W")
            save_matrix(initial_M, "initial_M")

            # Run simulation
            W, M = run_simulation(
                initial_W=initial_W,
                initial_M=initial_M,
                input_generator=input_generator,
                parameters=parameters,
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


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweeps using YAML configuration"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_sweep_config(args.config)

    # Initialize wandb
    wandb.login()

    # Run the sweep
    run_grid_experiments(
        config["param_grid"], config.get("base_params", {}), config["group_name"]
    )


if __name__ == "__main__":
    main()
