import yaml
import torch
import os
import sys
from itertools import product
from pathlib import Path
from typing import Dict, Any, List
import json

from params import LinearParameters
from simulator import linear_simulation
from initialisation import generate_initial_weights, generate_conditions


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")


def merge_configs(
    base_config: Dict[str, Any], experiment_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge base config with experiment-specific config."""
    merged = base_config.copy()

    # Override base params if they exist in experiment config
    if "base_params" in experiment_config:
        merged.update(experiment_config["base_params"])

    return merged.get("base_params", {})


def generate_param_combinations(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from param_grid."""
    if not param_grid:
        return [{}]  # Single empty combination if no param_grid

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = []
    for combination in product(*values):
        combinations.append(dict(zip(keys, combination)))

    return combinations


def params_to_filename(params: Dict[str, Any]) -> str:
    """Convert parameter dictionary to filename."""
    if not params:
        return "single_run"

    # Create filename from key-value pairs
    filename_parts = []
    for key, value in sorted(params.items()):
        # Handle different value types
        if isinstance(value, bool):
            filename_parts.append(f"{key}_{str(value)}")
        elif isinstance(value, (int, float)):
            filename_parts.append(f"{key}_{value}")
        else:
            filename_parts.append(f"{key}_{str(value)}")

    return "_".join(filename_parts)


def run_experiment(experiment_name: str) -> bool:
    """Run a single experiment. Returns True if successful, False otherwise."""
    try:
        print(f"Starting experiment: {experiment_name}")

        # Load configurations
        base_config = load_config("configs/base.yaml")
        experiment_config = load_config(f"configs/{experiment_name}.yaml")

        # Merge configurations
        merged_params = merge_configs(base_config, experiment_config)

        # Generate parameter combinations
        param_grid = experiment_config.get("param_grid", {})
        param_combinations = generate_param_combinations(param_grid)

        print(f"Running {len(param_combinations)} parameter combinations")

        # Create results directory
        results_dir = Path(f"results/{experiment_name}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Store metadata
        metadata = {
            "experiment_name": experiment_name,
            "base_config": base_config,
            "experiment_config": experiment_config,
            "param_combinations": param_combinations,
        }

        with open(results_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Run simulations for each parameter combination
        successful_runs = 0
        for i, param_combo in enumerate(param_combinations):
            try:
                print(
                    f"  Running combination {i+1}/{len(param_combinations)}: {param_combo}"
                )

                # Create final parameters
                final_params = merged_params.copy()
                final_params.update(param_combo)

                # Create LinearParameters object
                parameters = LinearParameters(**final_params)

                # Generate initial conditions
                initial_W, initial_M = generate_initial_weights(parameters)
                spectrum, basis = generate_conditions(parameters)

                # Run simulation
                final_W, final_M, metrics_over_time = linear_simulation(
                    initial_W=initial_W,
                    initial_M=initial_M,
                    spectrum=spectrum,
                    basis=basis,
                    parameters=parameters,
                )

                print(f"    ✓ Simulation completed for {param_combo}. Saving results.")

                # Save results
                filename = params_to_filename(param_combo) + ".pt"
                filepath = results_dir / filename
                torch.save(metrics_over_time, filepath)

                successful_runs += 1
                print(f"    ✓ Saved results to {filepath}")

            except Exception as e:
                print(f"    ✗ Error in parameter combination {param_combo}: {e}")
                continue

        print(
            f"Experiment {experiment_name} completed: {successful_runs}/{len(param_combinations)} runs successful"
        )
        return successful_runs > 0

    except Exception as e:
        print(f"✗ Failed to run experiment {experiment_name}: {e}")
        return False


def main():
    """Main function to run experiments from command line."""
    if len(sys.argv) < 2:
        print("Usage: python run_experiments.py <experiment1> [experiment2] ...")
        print("Available experiments:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.yaml"):
                if config_file.name != "base.yaml":
                    print(f"  {config_file.stem}")
        sys.exit(1)

    experiment_names = sys.argv[1:]

    print(f"Running {len(experiment_names)} experiments: {', '.join(experiment_names)}")

    successful_experiments = 0
    for experiment_name in experiment_names:
        success = run_experiment(experiment_name)
        if success:
            successful_experiments += 1

    print(
        f"\nCompleted: {successful_experiments}/{len(experiment_names)} experiments successful"
    )


if __name__ == "__main__":
    main()
