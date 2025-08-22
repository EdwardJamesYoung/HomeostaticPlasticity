import yaml
import torch
import os
import sys
from itertools import product
from pathlib import Path
from typing import Dict, Any, List, Union
import json
import copy

from params import SimulationParameters
from simulator import run_simulation
from initialisation import generate_initial_weights
from input_generation import (
    CircularInputGenerator,
    TorusInputGenerator,
    DistributionConfig1D,
    DistributionConfig2D,
    create_config_grid_1d,
    create_config_grid_2d,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")


def deep_merge_configs(
    base_config: Dict[str, Any], experiment_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge experiment config into base config."""

    def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(base_config, experiment_config)


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


def create_distribution_configs(
    input_params: Dict[str, Any],
) -> Dict[str, Union[DistributionConfig1D, DistributionConfig2D]]:
    """Create DistributionConfig objects from input_params, expanding grids where specified."""

    # Determine if we're working with 1D or 2D based on input_generator_name
    input_generator_name = input_params.get(
        "input_generator_name", "circular_input_generator"
    )
    is_2d = "torus" in input_generator_name.lower()

    config_names = [
        "probability_config",
        "density_config",
        "gain_config",
        "width_config",
        "excitatory_third_factor_config",
        "inhibitory_third_factor_config",
    ]

    # Handle legacy "probability" naming
    if "probability" in input_params and "probability_config" not in input_params:
        input_params["probability_config"] = input_params["probability"]

    configs = {}

    for config_name in config_names:
        if config_name not in input_params:
            # Create default config
            if is_2d:
                configs[config_name] = DistributionConfig2D()
            else:
                configs[config_name] = DistributionConfig1D()
            continue

        config_data = input_params[config_name]

        # Check if this config has lists (indicating a grid)
        has_grid = any(isinstance(v, list) for v in config_data.values())

        if has_grid:
            # Extract lists and create grid
            if is_2d:
                # For 2D, location should be a list of [x, y] pairs
                mixing_parameters = config_data.get("mixing_parameter", [0.0])
                concentrations = config_data.get("concentration", [1.0])
                locations = config_data.get("location", [[0.0, 0.0]])

                configs[config_name] = create_config_grid_2d(
                    mixing_parameters=mixing_parameters,
                    concentrations=concentrations,
                    locations=locations,
                )
            else:
                # For 1D
                mixing_parameters = config_data.get("mixing_parameter", [0.0])
                concentrations = config_data.get("concentration", [1.0])
                locations = config_data.get("location", [0.0])

                configs[config_name] = create_config_grid_1d(
                    mixing_parameters=mixing_parameters,
                    concentrations=concentrations,
                    locations=locations,
                )
        else:
            # Single values, create single config
            if is_2d:
                location = config_data.get("location", [0.0, 0.0])
                if not isinstance(location, list):
                    location = [location, location]

                configs[config_name] = DistributionConfig2D(
                    mixing_parameter=torch.tensor(
                        [[config_data.get("mixing_parameter", 0.0)]]
                    ),
                    concentration=torch.tensor(
                        [[config_data.get("concentration", 1.0)]]
                    ),
                    location=torch.tensor([location]),
                )
            else:
                configs[config_name] = DistributionConfig1D(
                    mixing_parameter=torch.tensor(
                        [[config_data.get("mixing_parameter", 0.0)]]
                    ),
                    concentration=torch.tensor(
                        [[config_data.get("concentration", 1.0)]]
                    ),
                    location=torch.tensor([[config_data.get("location", 0.0)]]),
                )

    return configs


def create_input_generator(
    merged_config: Dict[str, Any],
) -> Union[CircularInputGenerator, TorusInputGenerator]:
    """Create appropriate input generator based on config."""

    # Create SimulationParameters
    base_params = merged_config.get("base_params", {})
    parameters = SimulationParameters(**base_params)

    # Create distribution configs
    input_params = merged_config.get("input_params", {})
    distribution_configs = create_distribution_configs(input_params)

    # Determine generator type
    input_generator_name = input_params.get(
        "input_generator_name", "circular_input_generator"
    )

    if "torus" in input_generator_name.lower():
        return TorusInputGenerator(parameters=parameters, **distribution_configs)
    else:
        return CircularInputGenerator(parameters=parameters, **distribution_configs)


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


def save_distribution_configs(
    distribution_configs: Dict[str, Union[DistributionConfig1D, DistributionConfig2D]],
) -> Dict[str, Dict[str, Any]]:
    """Convert DistributionConfig objects to serializable format."""
    serializable_configs = {}

    for name, config in distribution_configs.items():
        config_dict = {
            "mixing_parameter": config.mixing_parameter.detach().cpu(),
            "concentration": config.concentration.detach().cpu(),
            "location": config.location.detach().cpu(),
            "batch_size": config.batch_size,
            "device": str(config.device),
            "type": "2D" if isinstance(config, DistributionConfig2D) else "1D",
        }
        serializable_configs[name] = config_dict

    return serializable_configs


def extract_base_experiment_name(experiment_path: str) -> str:
    """Extract the base experiment name from either a regular name or splintered config path."""
    if "/" in experiment_path or experiment_path.endswith(".yaml"):
        # This is a path to a config file
        path = Path(experiment_path)

        # Check if this is a splintered config
        if "splintered_configs" in path.parts:
            # Extract from path like: splintered_configs/test_splintered/test_001.yaml
            splintered_dir = None
            for part in path.parts:
                if part.endswith("_splintered"):
                    splintered_dir = part
                    break

            if splintered_dir:
                # Remove '_splintered' suffix to get original experiment name
                return splintered_dir[:-10]  # Remove '_splintered'
            else:
                # Fallback to config filename
                return path.stem
        else:
            # Regular config file, use filename
            return path.stem
    else:
        # Just an experiment name
        return experiment_path


def create_metadata_filename(param_combo: Dict[str, Any]) -> str:
    """Create a unique metadata filename based on parameter combination."""
    if not param_combo:
        return "metadata.json"

    # Create a descriptive name from parameters
    param_parts = []
    for key, value in sorted(param_combo.items()):
        param_parts.append(f"{key}_{value}")

    return f"metadata_{'_'.join(param_parts)}.json"


def run_experiment(experiment_path: str) -> bool:
    """Run a single experiment. Returns True if successful, False otherwise."""
    try:
        # Extract base experiment name for consistent results directory
        base_experiment_name = extract_base_experiment_name(experiment_path)

        # Determine if this is a full path or just an experiment name
        if "/" in experiment_path or experiment_path.endswith(".yaml"):
            # Full path provided
            experiment_config_path = experiment_path
            # Extract experiment name for display
            display_name = Path(experiment_path).stem
            # For splintered configs, look for base.yaml in the original configs directory
            base_config_path = "configs/base.yaml"
        else:
            # Experiment name provided (legacy behavior)
            display_name = experiment_path
            experiment_config_path = f"configs/{experiment_path}.yaml"
            base_config_path = "configs/base.yaml"

        print(f"Starting experiment: {display_name}")
        print(f"Results will be saved to: results/{base_experiment_name}/")

        # Load configurations
        base_config = load_config(base_config_path)
        experiment_config = load_config(experiment_config_path)

        # Deep merge configurations
        merged_config = deep_merge_configs(base_config, experiment_config)

        # Generate parameter combinations from param_grid only
        param_grid = experiment_config.get("param_grid", {})
        param_combinations = generate_param_combinations(param_grid)

        print(f"Running {len(param_combinations)} parameter combinations")

        # Create results directory using base experiment name
        results_dir = Path(f"results/{base_experiment_name}")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Run simulations for each parameter combination
        successful_runs = 0
        for i, param_combo in enumerate(param_combinations):
            try:
                print(
                    f"  Running combination {i+1}/{len(param_combinations)}: {param_combo}"
                )

                # Create final config for this parameter combination
                final_config = copy.deepcopy(merged_config)
                final_config["base_params"].update(param_combo)

                # Create input generator (this handles the batch_size from input_params grid)
                input_generator = create_input_generator(final_config)

                # Create SimulationParameters
                parameters = SimulationParameters(**final_config["base_params"])

                # Generate initial conditions
                initial_W, initial_M = generate_initial_weights(parameters)

                print(
                    f"    Running simulation with batch_size={input_generator.batch_size}"
                )

                # Run simulation
                final_W, final_M, metrics_over_time = run_simulation(
                    initial_W=initial_W,
                    initial_M=initial_M,
                    input_generator=input_generator,
                    parameters=parameters,
                )

                print(f"    ✓ Simulation completed for {param_combo}. Saving results.")

                # Save the .pt file with parameter-based filename
                pt_filename = params_to_filename(param_combo) + ".pt"
                pt_filepath = results_dir / pt_filename

                # Prepare results to save (same as before)
                results = {
                    "final_W": final_W.detach().cpu(),
                    "final_M": final_M.detach().cpu(),
                    "metrics_over_time": {
                        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in metrics_over_time.items()
                    },
                    "parameters": parameters.to_dict(),
                    "distribution_configs": save_distribution_configs(
                        {
                            "probability_config": input_generator.probability_config,
                            "density_config": input_generator.density_config,
                            "gain_config": input_generator.gain_config,
                            "width_config": input_generator.width_config,
                            "excitatory_third_factor_config": input_generator.excitatory_third_factor_config,
                            "inhibitory_third_factor_config": input_generator.inhibitory_third_factor_config,
                        }
                    ),
                    "input_generator_info": {
                        "type": type(input_generator).__name__,
                        "batch_size": input_generator.batch_size,
                        "stimuli_locations_shape": input_generator.stimuli_locations.shape,
                        "neuron_locations_shape": input_generator.neuron_locations.shape,
                    },
                }

                torch.save(results, pt_filepath)

                # Save separate metadata file for this parameter combination
                metadata_filename = create_metadata_filename(param_combo)
                metadata_filepath = results_dir / metadata_filename

                metadata = {
                    "experiment_name": base_experiment_name,
                    "display_name": display_name,
                    "experiment_config_path": experiment_config_path,
                    "base_config": base_config,
                    "experiment_config": experiment_config,
                    "param_combination": param_combo,
                    "pt_filename": pt_filename,
                }

                with open(metadata_filepath, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                successful_runs += 1
                print(f"    ✓ Saved results to {pt_filepath}")
                print(f"    ✓ Saved metadata to {metadata_filepath}")

            except Exception as e:
                print(f"    ✗ Error in parameter combination {param_combo}: {e}")
                continue

        print(
            f"Experiment {display_name} completed: {successful_runs}/{len(param_combinations)} runs successful"
        )
        print(f"All results saved to: {results_dir}")
        return successful_runs > 0

    except Exception as e:
        print(f"✗ Failed to run experiment {experiment_path}: {e}")
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
