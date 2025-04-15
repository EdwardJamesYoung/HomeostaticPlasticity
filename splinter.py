import argparse
import yaml
import os
from itertools import product
import copy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a configuration file into individual configurations for each parameter combination."
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the input configuration file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output directory"
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def expand_input_config(input_configs):
    """
    Expands the input_config list into all possible combinations.
    Each entry in input_configs can have its own param_grid.
    """
    all_expanded = []

    for config in input_configs:
        generator_type = config["type"]

        if "param_grid" not in config:
            # Simple case - no parameter sweep
            all_expanded.append({"type": generator_type})
            continue

        # Get the parameter grid for this generator
        param_grid = config["param_grid"]

        # Get all parameter names and values for the sweep
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Generate all combinations of parameter values
        combinations = list(product(*param_values))

        # Create a dictionary for each parameter combination
        for combination in combinations:
            expanded_config = {"type": generator_type}
            expanded_config.update(dict(zip(param_names, combination)))
            all_expanded.append(expanded_config)

    return all_expanded


def generate_all_combinations(config):
    """
    Generates all possible combinations of parameters from the param_grid.
    """
    param_grid = copy.deepcopy(config["param_grid"])

    # Extract and process input_config separately
    input_configs = param_grid.pop("input_config", [])
    expanded_inputs = expand_input_config(input_configs)

    # Get all parameter names and values for regular parameters
    regular_param_names = list(param_grid.keys())
    regular_param_values = list(param_grid.values())

    # Generate all combinations of regular parameters
    regular_combinations = list(product(*regular_param_values))

    # Final result: all combinations of regular params and input_config
    all_combinations = []

    for regular_combo in regular_combinations:
        regular_params = dict(zip(regular_param_names, regular_combo))

        for input_config in expanded_inputs:
            # Combine regular parameters and input_config
            combined_params = copy.deepcopy(regular_params)
            combined_params["input_config"] = input_config
            all_combinations.append(combined_params)

    return all_combinations


class QuotedString(str):
    """String wrapper class to mark strings that should be quoted in YAML."""

    pass


def setup_yaml():
    """
    Configure YAML dumper to enhance readability and quote only string values,
    not keys or other elements.
    """

    def quoted_presenter(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

    def regular_str_presenter(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(QuotedString, quoted_presenter, Dumper=yaml.SafeDumper)
    yaml.add_representer(str, regular_str_presenter, Dumper=yaml.SafeDumper)

    def represent_none(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:null", "null")

    def represent_list(dumper, data):
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=False
        )

    yaml.add_representer(type(None), represent_none, Dumper=yaml.SafeDumper)
    yaml.add_representer(list, represent_list, Dumper=yaml.SafeDumper)


def quote_string_values(data):
    """
    Recursively process a data structure and wrap string values (not keys) with QuotedString
    to ensure they will be quoted in the YAML output.
    """
    if isinstance(data, dict):
        return {k: quote_string_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [quote_string_values(item) for item in data]
    elif isinstance(data, str) and data.lower() not in (
        "true",
        "false",
        "null",
        "yes",
        "no",
        "on",
        "off",
    ):
        # Only wrap strings that aren't special YAML values
        return QuotedString(data)
    else:
        return data


def generate_configs(config, output_dir):
    """
    Generates individual configuration files for each parameter combination.
    """
    group_name = config["group_name"]
    base_params = config.get("base_params", {})

    # Generate all parameter combinations
    all_combinations = generate_all_combinations(config)

    for i, combination in enumerate(all_combinations, 1):
        # Create a new configuration with the original group_name
        new_config = {
            "group_name": group_name,
            "base_params": copy.deepcopy(base_params),
            "param_grid": {},  # Always include param_grid section
        }

        # Add regular parameters to base_params
        for key, value in combination.items():
            if key != "input_config":
                new_config["base_params"][key] = value

        # Handle input_config properly
        if "input_config" in combination:
            input_config = combination[
                "input_config"
            ]  # This is a dict like {"type": "circular", "mixing_parameter": 0.1, ...}

            # Create a new input config structure with proper param_grid
            new_input_config = {"type": input_config["type"], "param_grid": {}}

            # Extract parameters (excluding type) and make single-item lists
            for param_key, param_value in input_config.items():
                if param_key != "type":
                    new_input_config["param_grid"][param_key] = [param_value]

            # Add to the main param_grid
            new_config["param_grid"]["input_config"] = [new_input_config]

        # Process string values to be quoted
        new_config = quote_string_values(new_config)

        # Generate a filename for this configuration
        filename = f"{group_name}_config_{i}.yaml"
        file_path = os.path.join(output_dir, filename)

        # Save the configuration with nice formatting
        with open(file_path, "w") as f:
            yaml.dump(new_config, f, default_flow_style=False, Dumper=yaml.SafeDumper)

    print(f"Generated {len(all_combinations)} configuration files.")


def main():
    args = parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load the input configuration
    config = load_config(args.config)

    # Configure YAML dumper for better output formatting
    setup_yaml()

    # Generate individual configurations
    generate_configs(config, args.output)

    print(f"Split configurations saved to {args.output}")


if __name__ == "__main__":
    main()
