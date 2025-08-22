#!/usr/bin/env python3

import argparse
import yaml
import os
import shutil
from pathlib import Path
from itertools import product
from typing import Dict, Any, List
import copy


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")


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


def create_single_config(
    base_config: Dict[str, Any], param_combo: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a config with a single parameter combination."""
    config = copy.deepcopy(base_config)

    # Replace param_grid with single combination
    single_param_grid = {}
    for key, value in param_combo.items():
        single_param_grid[key] = [value]  # Wrap in list for consistency

    config["param_grid"] = single_param_grid
    return config


def create_slurm_script(
    template_path: str, output_path: str, substitutions: Dict[str, str]
):
    """Create a slurm script from template with substitutions."""
    with open(template_path, "r") as f:
        template_content = f.read()

    for placeholder, value in substitutions.items():
        template_content = template_content.replace(placeholder, value)

    with open(output_path, "w") as f:
        f.write(template_content)

    # Make executable
    os.chmod(output_path, 0o755)


def format_param_combo(param_combo: Dict[str, Any]) -> str:
    """Format parameter combination for display."""
    if not param_combo:
        return "default"

    parts = []
    for key, value in param_combo.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Splinter a config file into individual experiment configs"
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without actually creating files",
    )
    args = parser.parse_args()

    # Load the config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)

    # Extract necessary information
    if "group_name" not in config:
        raise ValueError("Config file must have a 'group_name' field")

    group_name = config["group_name"]
    param_grid = config.get("param_grid", {})

    # Generate parameter combinations
    param_combinations = generate_param_combinations(param_grid)

    print(f"Found {len(param_combinations)} parameter combination(s):")
    for i, combo in enumerate(param_combinations, 1):
        print(f"  {i:3d}: {format_param_combo(combo)}")

    if len(param_combinations) <= 1:
        print(
            f"\nOnly {len(param_combinations)} parameter combination(s) found. No need to splinter."
        )
        return

    # Create output directory
    config_name = config_path.stem  # Get filename without extension
    output_dir = Path(f"splintered_configs/{config_name}_splintered")

    if args.dry_run:
        print(f"\n[DRY RUN] Would create directory: {output_dir}")
        print(f"[DRY RUN] Would create {len(param_combinations)} config files:")
        for i in range(len(param_combinations)):
            config_filename = f"{group_name}_{i+1:03d}.yaml"
            print(f"  {output_dir / config_filename}")
        print(
            f"[DRY RUN] Would create SLURM script: {output_dir / f'{group_name}_submit_all.sh'}"
        )
        return

    if output_dir.exists():
        print(f"\nDirectory {output_dir} already exists. Removing...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    print(f"\nCreated directory: {output_dir}")

    # Create individual configs
    config_files = []
    for i, param_combo in enumerate(param_combinations, 1):
        # Create single config
        single_config = create_single_config(config, param_combo)

        # Write to file
        config_filename = f"{group_name}_{i:03d}.yaml"
        config_filepath = output_dir / config_filename

        with open(config_filepath, "w") as f:
            yaml.dump(single_config, f, default_flow_style=False, sort_keys=False)

        config_files.append(config_filename)
        print(f"  Created {config_filepath}")

    # Check if slurm template exists
    template_path = Path("slurm_template.sh")
    if not template_path.exists():
        raise FileNotFoundError(f"SLURM template not found: {template_path}")

    # Create slurm script
    slurm_script_path = output_dir / f"{group_name}_submit_all.sh"

    substitutions = {
        "__JOB_NAME__": group_name,
        "__ARRAY_RANGE__": f"1-{len(param_combinations)}",
        "__CONFIG_PREFIX__": f"{group_name}_",
        "__CONFIG_DIR__": f"splintered_configs/{config_name}_splintered",
    }

    create_slurm_script(template_path, slurm_script_path, substitutions)
    print(f"  Created {slurm_script_path}")

    print(f"\n✓ Successfully splintered {len(param_combinations)} experiments")
    print(f"✓ To submit jobs: sbatch {slurm_script_path}")
    print(f"✓ Job array will run indices 1-{len(param_combinations)}")


if __name__ == "__main__":
    main()
