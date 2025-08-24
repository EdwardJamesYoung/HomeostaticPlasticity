import yaml
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from matplotlib import pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.append("..")

from input_generation import (
    CircularInputGenerator,
    TorusInputGenerator,
    DistributionConfig1D,
    DistributionConfig2D,
)
from params import SimulationParameters


def load_style() -> Dict[str, Any]:
    """Load styling configuration from style.yaml."""
    style_path = Path("style.yaml")
    try:
        with open(style_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Style file not found: {style_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing style file: {e}")


def load_experiment_data(
    experiment_name: str,
) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    """
    Load experiment data and metadata.

    Returns:
        data_dict: Dictionary mapping parameter combination names to results
        metadata_dict: Dictionary mapping parameter combination names to their metadata
    """
    results_dir = Path(f"../results/{experiment_name}")

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load all .pt files and their corresponding metadata
    data_dict = {}
    metadata_dict = {}

    for pt_file in results_dir.glob("*.pt"):
        # Extract parameter combination name (filename without extension)
        param_name = pt_file.stem

        # Load results
        try:
            results = torch.load(pt_file, map_location="cpu", weights_only=False)
            data_dict[param_name] = results
        except Exception as e:
            print(f"Warning: Could not load {pt_file}: {e}")
            continue

        # Load corresponding metadata file
        metadata_path = results_dir / f"metadata_{param_name}.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                metadata_dict[param_name] = metadata
            except Exception as e:
                print(f"Warning: Could not load metadata for {param_name}: {e}")
        else:
            print(f"Warning: No metadata file found for {param_name}")

    if not data_dict:
        raise ValueError(f"No valid data files found in {results_dir}")

    return data_dict, metadata_dict


def reconstruct_distribution_config(
    saved_config: Dict[str, Any],
) -> Union[DistributionConfig1D, DistributionConfig2D]:
    """
    Reconstruct a DistributionConfig object from saved configuration data.

    Args:
        saved_config: Dictionary containing saved config data

    Returns:
        Reconstructed DistributionConfig object
    """
    config_type = saved_config["type"]

    if config_type == "1D":
        return DistributionConfig1D(
            mixing_parameter=saved_config["mixing_parameter"],
            concentration=saved_config["concentration"],
            location=saved_config["location"],
        )
    elif config_type == "2D":
        return DistributionConfig2D(
            mixing_parameter=saved_config["mixing_parameter"],
            concentration=saved_config["concentration"],
            location=saved_config["location"],
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def create_input_generator_from_results(
    results: Dict[str, Any],
) -> Union[CircularInputGenerator, TorusInputGenerator]:
    """
    Recreate an InputGenerator from saved results.

    Args:
        results: Results dictionary containing parameters and distribution configs

    Returns:
        Reconstructed InputGenerator
    """
    # Create parameters
    parameters = SimulationParameters(**results["parameters"])

    # Reconstruct distribution configs
    distribution_configs = {}
    for config_name, saved_config in results["distribution_configs"].items():
        distribution_configs[config_name] = reconstruct_distribution_config(
            saved_config
        )

    # Determine generator type
    generator_type = results["input_generator_info"]["type"]

    if generator_type == "CircularInputGenerator":
        return CircularInputGenerator(parameters=parameters, **distribution_configs)
    elif generator_type == "TorusInputGenerator":
        return TorusInputGenerator(parameters=parameters, **distribution_configs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def extract_final_curves(
    metrics_over_time: Dict[str, torch.Tensor], curve_names: list[str]
) -> Dict[str, torch.Tensor]:
    """
    Extract final time point data for specified curves.

    Args:
        metrics_over_time: Dictionary of metrics over time
        curve_names: List of curve names to extract (e.g., ["curves/density", "curves/gains"])

    Returns:
        Dictionary mapping curve names to their final values
    """
    final_curves = {}

    for curve_name in curve_names:
        if curve_name in metrics_over_time:
            # Get final time point (last index along first dimension)
            final_curves[curve_name] = metrics_over_time[curve_name][-1]
        else:
            print(f"Warning: {curve_name} not found in metrics_over_time")

    return final_curves


def extract_final_tuning_curves(
    metrics_over_time: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Extract final tuning curves for all neurons.

    Args:
        metrics_over_time: Dictionary of metrics over time

    Returns:
        Tensor of shape [N_I, num_stimuli] containing final tuning curves
    """
    if "rates" not in metrics_over_time:
        raise ValueError("rates not found in metrics_over_time")

    # rates has shape [time_steps, repeats, batch_size, N_I, num_stimuli]
    # We want final time, first repeat, first batch: [N_I, num_stimuli]
    rates = metrics_over_time["rates"]
    final_rates = rates[-1, 0, 0, :, :]  # [N_I, num_stimuli]
    return final_rates


def plot_stimulus_curves(
    ax,
    stimulus_locations: torch.Tensor,
    curves_dict: Dict[str, Tuple[torch.Tensor, str, str]],
    individual_curves: Optional[torch.Tensor] = None,
    individual_color: str = "grey",
    individual_alpha: float = 0.7,
):
    """
    Plot multiple curves against stimulus locations on the same axis.

    Args:
        ax: Matplotlib axis
        stimulus_locations: Stimulus positions, shape [num_stimuli] or [num_stimuli, 1]
        curves_dict: Dict mapping curve names to (data, color, label) tuples
        individual_curves: Optional tensor of individual curves [N_curves, num_stimuli]
        individual_color: Color for individual curves
        individual_alpha: Alpha for individual curves
    """
    # Ensure stimulus_locations is 1D
    if stimulus_locations.dim() > 1:
        stimulus_locations = stimulus_locations.squeeze()

    # Plot individual curves first (so they appear behind main curves)
    if individual_curves is not None:
        for i in range(individual_curves.shape[0]):
            ax.plot(
                stimulus_locations.cpu().numpy(),
                individual_curves[i].cpu().numpy(),
                color=individual_color,
                alpha=individual_alpha,
                linewidth=0.5,
                zorder=1,
            )

    # Plot main curves
    for curve_name, (data, color, label) in curves_dict.items():
        # Handle different data shapes - might be [batch, num_stimuli] or [num_stimuli]
        data = data.squeeze()

        if data.dim() > 1:
            # Take first batch element if batched
            plot_data = data[0].cpu().numpy()
        else:
            plot_data = data.cpu().numpy()

        ax.plot(
            stimulus_locations.cpu().numpy(),
            plot_data,
            color=color,
            label=label,
            linewidth=1.5,
            zorder=2,
        )
