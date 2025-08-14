import yaml
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from matplotlib import pyplot as plt


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
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], torch.Tensor]:
    """
    Load experiment data and metadata.

    Returns:
        data_dict: Dictionary mapping parameter combination names to metrics_over_time
        metadata: Experiment metadata
        time_values: Time values for the experiment
    """
    results_dir = Path(f"../results/{experiment_name}")

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load metadata
    metadata_path = results_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Extract timing parameters from metadata
    base_config = metadata["base_config"]["base_params"]
    experiment_config = metadata["experiment_config"]

    # Merge configs to get final parameters
    final_params = base_config.copy()
    if "base_params" in experiment_config:
        final_params.update(experiment_config["base_params"])

    T = final_params.get("T", 120000.0)
    log_time = final_params.get("log_time", 20.0)

    # Calculate number of log steps and time values
    num_log_steps = int(T / log_time) + 1
    time_values = torch.linspace(0, T, num_log_steps)

    # Load all .pt files in the results directory
    data_dict = {}
    for pt_file in results_dir.glob("*.pt"):
        if pt_file.name != "metadata.json":  # Skip metadata
            # Extract parameter combination name (filename without extension)
            param_name = pt_file.stem

            # Load metrics
            try:
                metrics_over_time = torch.load(pt_file, map_location="cpu")
                data_dict[param_name] = metrics_over_time
            except Exception as e:
                print(f"Warning: Could not load {pt_file}: {e}")
                continue

    if not data_dict:
        raise ValueError(f"No valid data files found in {results_dir}")

    return data_dict, metadata, time_values


def plot_quantile_line(
    ax,
    time_values: torch.Tensor,
    metrics: Dict[str, torch.Tensor],
    metric_base: str,
    color_scheme: Dict[str, str],
    alphas: Dict[str, float],
    line_width: float,
    label: str,
):
    """
    Plot a line with quantile bands.

    Args:
        ax: Matplotlib axis
        time_values: Time values for x-axis
        metrics: Dictionary of all metrics
        metric_base: Base name for the metric (e.g., "mode_diff/variances_vs_spectrum")
        color_scheme: Dictionary with 'light', 'normal', 'dark' colors
        alphas: Dictionary with alpha values
        line_width: Width for the median line
        label: Label for the legend
    """
    # Extract quantile data
    q10 = metrics[f"{metric_base}_q10"].cpu()
    q25 = metrics[f"{metric_base}_q25"].cpu()
    q50 = metrics[f"{metric_base}_q50"].cpu()
    q75 = metrics[f"{metric_base}_q75"].cpu()
    q90 = metrics[f"{metric_base}_q90"].cpu()

    # Plot quantile bands
    ax.fill_between(
        time_values, q10, q25, alpha=alphas["light"], color=color_scheme["light"]
    )
    ax.fill_between(
        time_values, q75, q90, alpha=alphas["light"], color=color_scheme["light"]
    )
    ax.fill_between(
        time_values, q25, q75, alpha=alphas["normal"], color=color_scheme["normal"]
    )

    # Plot median line
    ax.plot(
        time_values, q50, color=color_scheme["dark"], linewidth=line_width, label=label
    )


def get_final_metrics(
    data_dict: Dict[str, Dict[str, torch.Tensor]],
    homeostasis_value: bool,
    metric_names: list[str],
    k_I_values: Optional[list[float]] = None,
    tau_W_values: Optional[list[float]] = None,
    N_E_values: Optional[list[int]] = None,
) -> Dict[str, list[float]]:

    results: Dict[str, list[float]] = {metric: [] for metric in metric_names}

    if k_I_values is None and tau_W_values is None and N_E_values is None:
        filenames = [f"homeostasis_{homeostasis_value}"]
    elif k_I_values is not None:
        filenames = [f"homeostasis_{homeostasis_value}_k_I_{k_I}" for k_I in k_I_values]
    elif tau_W_values is not None:
        filenames = [
            f"homeostasis_{homeostasis_value}_tau_W_{tau_W}" for tau_W in tau_W_values
        ]
    elif N_E_values is not None:
        filenames = [f"N_E_{N_E}_homeostasis_{homeostasis_value}" for N_E in N_E_values]

    for filename in filenames:
        if filename in data_dict:
            metrics = data_dict[filename]

            for metric in metric_names:
                if metric in metrics:
                    # Get final value (last time point)
                    final_value = metrics[metric][-1].item()
                    results[metric].append(final_value)
                else:
                    print(f"Warning: Metric {metric} not found in {filename}")
                    results[metric].append(np.nan)
        else:
            print(f"Warning: Data file {filename} not found")
            for metric in metric_names:
                results[metric].append(np.nan)

    return results


def create_boxplot_data(final_metrics, metric_base):
    """
    Create boxplot data from final metrics for different quantiles.

    Args:
        final_metrics: dict with metrics for each k_I value
        metric_base: base name for the quantile metrics

    Returns:
        dict: 'median', 'q25', 'q75', 'q10', 'q90' -> lists of values
    """
    quantiles = ["q10", "q25", "q50", "q75", "q90"]
    boxplot_data = {}

    for q in quantiles:
        metric_name = f"{metric_base}_{q}"
        if metric_name in final_metrics:
            boxplot_data[q] = final_metrics[metric_name]
        else:
            print(f"Warning: {metric_name} not found")
            boxplot_data[q] = [np.nan] * len(
                final_metrics[list(final_metrics.keys())[0]]
            )

    return boxplot_data


def plot_matplotlib_boxplot(
    ax, x_positions, boxplot_data, color_scheme, alphas, widths=0.2
):
    """Plot box-and-whisker plot using matplotlib's built-in boxplot."""

    # Prepare data for matplotlib boxplot
    # Each column is one box plot
    plot_data = []
    for i in range(len(x_positions)):
        # For each k_I value, create a "distribution" from quantiles
        # We can approximate this by creating multiple points at each quantile
        box_values = [
            boxplot_data["q10"][i],  # 10th percentile
            boxplot_data["q25"][i],  # 25th percentile
            boxplot_data["q50"][i],  # Median
            boxplot_data["q75"][i],  # 75th percentile
            boxplot_data["q90"][i],  # 90th percentile
        ]
        plot_data.append(box_values)

    # Create boxplot
    bp = ax.boxplot(
        plot_data,
        positions=x_positions,
        widths=widths,
        patch_artist=True,
        boxprops=dict(
            facecolor=color_scheme["normal"],
            alpha=alphas["normal"],
            edgecolor=color_scheme["normal"],
        ),
        whiskerprops=dict(color=color_scheme["light"]),
        capprops=dict(color=color_scheme["light"]),
        medianprops=dict(color=color_scheme["dark"]),
    )

    return bp


def plot_custom_boxplot(
    ax, x_positions, boxplot_data, color_scheme, alphas, line_width
):
    """Plot custom box-and-whisker plot with specified colors."""

    medians = boxplot_data["q50"]
    q25 = boxplot_data["q25"]
    q75 = boxplot_data["q75"]
    q10 = boxplot_data["q10"]
    q90 = boxplot_data["q90"]

    for i, x in enumerate(x_positions):
        # Skip if any values are NaN
        if any(np.isnan([medians[i], q25[i], q75[i], q10[i], q90[i]])):
            continue

        # Box (25th to 75th percentile)
        box_height = q75[i] - q25[i]
        box = plt.Rectangle(
            (x - 0.1, q25[i]),
            0.2,
            box_height,
            facecolor=color_scheme["normal"],
            edgecolor=color_scheme["normal"],
            alpha=alphas["normal"],
        )
        ax.add_patch(box)

        # Median line
        ax.plot(
            [x - 0.1, x + 0.1],
            [medians[i], medians[i]],
            color=color_scheme["dark"],
            linewidth=line_width,
        )

        # Whiskers
        # Lower whisker (25th to 10th percentile)
        ax.plot(
            [x, x], [q25[i], q10[i]], color=color_scheme["light"], linewidth=line_width
        )
        ax.plot(
            [x - 0.05, x + 0.05],
            [q10[i], q10[i]],
            color=color_scheme["light"],
            linewidth=line_width,
        )

        # Upper whisker (75th to 90th percentile)
        ax.plot(
            [x, x], [q75[i], q90[i]], color=color_scheme["light"], linewidth=line_width
        )
        ax.plot(
            [x - 0.05, x + 0.05],
            [q90[i], q90[i]],
            color=color_scheme["light"],
            linewidth=line_width,
        )
