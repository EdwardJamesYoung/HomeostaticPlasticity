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

    Returns:
        int: The number of configuration files generated
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

    num_configs = len(all_combinations)
    print(f"Generated {num_configs} configuration files.")

    return num_configs


def main():
    args = parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load the input configuration
    config = load_config(args.config)

    # Configure YAML dumper for better output formatting
    setup_yaml()

    # Generate individual configurations
    num_configs = generate_configs(config, args.output)

    # Generate the SLURM batch script
    script_name = generate_slurm_script(config["group_name"], args.output, num_configs)

    print(f"Split configurations saved to {args.output}")
    print(f"SLURM batch script saved to {args.output}/{script_name}")


def generate_slurm_script(group_name, output_dir, num_configs):
    """
    Generates a SLURM batch script for running all the created configurations.

    Args:
        group_name (str): The group name from the configuration
        output_dir (str): Directory where configs were saved and where to save the SLURM script
        num_configs (int): Number of configurations generated

    Returns:
        str: The name of the generated script file
    """
    # Get the template file path - it should be in the same directory as this script
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "slurm_template.sh"
    )

    # If the template doesn't exist, create a default one
    if not os.path.exists(template_path):
        create_default_slurm_template(template_path)

    # Read the template
    with open(template_path, "r") as f:
        template_content = f.read()

    # Replace placeholders in the template
    script_content = template_content.replace("__JOB_NAME__", group_name)
    script_content = script_content.replace("__ARRAY_RANGE__", f"1-{num_configs}")
    script_content = script_content.replace(
        "__CONFIG_PREFIX__", f"{group_name}_config_"
    )

    # Use a relative path instead of absolute path
    # Since the SLURM script will be in the same directory as the config files,
    # we can just use the current directory
    script_content = script_content.replace("__CONFIG_PATH__", ".")

    # Create capitalized filename based on experiment name
    script_name = f"{group_name.upper()}_RUN_JOBS.sh"
    script_path = os.path.join(output_dir, script_name)

    # Write the SLURM script to the output directory
    with open(script_path, "w") as f:
        f.write(script_content)

    # Make the script executable
    os.chmod(script_path, 0o755)  # rwxr-xr-x

    # Return the script name
    return script_name


def create_default_slurm_template(template_path):
    """
    Creates a default SLURM template if one doesn't exist.
    """
    default_template = """#!/bin/bash
#!
#! SLURM job script for Ampere GPU Nodes
#!
#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################
#!
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH --job-name=__JOB_NAME__
#!
#! Which project should be charged:
#SBATCH --account=TERHANI-SL3-GPU
#!
#! Which partition should be used:
#SBATCH --partition ampere
#!
#! How many whole nodes should be allocated?
#SBATCH --nodes=1 
#! How many GPUs should be allocated per node? 
#SBATCH --gres=gpu:1
#!
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! How many tasks per node:
#SBATCH --ntasks-per-node=1
#!
#! How much wallclock time will be required? (format: hh:mm:ss)
#SBATCH --time=3:00:00 
#!
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=END,FAIL 
#!
#! Create an array of jobs (e.g. for parameter sweeps):
#SBATCH --array=__ARRAY_RANGE__
#!
#! Uncomment this to prevent the job from being requeued:
#SBATCH --no-requeue
#!
#! sbatch directives end here
#! ############################################################
#!
#! Number of nodes and tasks per node allocated by SLURM:
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\\([0-9][0-9]*\\).*$/\\1/')
#!
#! ############################################################
#!
#! Optionally modify the environment seen by the application
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
module load miniconda/3                    # python/3.10 cuda/11.7
source ~/.bashrc  # Required for conda things
conda deactivate
conda activate RL
#!
#! Work directory:
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory in which sbatch is run.
#!
#! Are you using OpenMP? If so increase this:
export OMP_NUM_THREADS=1
#!
#! Number of MPI tasks:
np=$[${numnodes}*${mpi_tasks_per_node}]
#!
CONFIG_NAME="__CONFIG_PREFIX__${SLURM_ARRAY_TASK_ID}.yaml"
PATH_TO_CONFIG="__CONFIG_PATH__/${CONFIG_NAME}"
#!
#! Command to run:
CMD="python sweep.py -c $PATH_TO_CONFIG"
#!
###############################################################
### You should not have to change anything below this line ####
###############################################################
#!
cd $workdir
echo -e "Changed directory to `pwd`.\\n"
#!
JOBID=$SLURM_JOB_ID
#!
echo -e "JobID: $JOBID\\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
#!
if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\\nNodes allocated:\\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\\..*$//g'`
fi
#!
echo -e "\\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
#!
echo -e "\\nExecuting command:\\n==================\\n$CMD\\n"
#!
eval $CMD  # This tells shell to run the command CMD
"""

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(template_path), exist_ok=True)

    # Write the default template
    with open(template_path, "w") as f:
        f.write(default_template)

    # Make the template executable
    os.chmod(template_path, 0o644)  # rw-r--r--


if __name__ == "__main__":
    main()
