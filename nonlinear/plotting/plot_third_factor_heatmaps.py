import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append("..")

from plotting_utils import (
    load_style,
    load_experiment_data,
    create_input_generator_from_results,
    create_custom_colormap,
    find_center_neuron_index,
    find_target_batch_index_2d,
)

from metrics import curves_log
from params import SimulationParameters


def plot_third_factor_heatmaps():
    """Create third factor heatmap comparison with 2x2 grid."""

    # Load style
    style = load_style()

    # Load data from both experiments
    excitatory_data_dict, excitatory_metadata_dict = load_experiment_data(
        "excitatory_third_factor_learning"
    )
    inhibitory_data_dict, inhibitory_metadata_dict = load_experiment_data(
        "inhibitory_third_factor_learning"
    )

    # Extract style parameters
    fig_width = style["sizes"]["figure_width"]
    ratio = style["sizes"]["ratio"]

    # Calculate panel dimensions (2x2 grid)
    panel_width = fig_width / 2
    panel_height = panel_width * ratio
    fig_height = 2 * panel_height

    # Color schemes and custom colormaps
    blue_colors = style["colours"]["blue"]
    purple_colors = style["colours"]["purple"]

    blue_cmap = create_custom_colormap(blue_colors, "blue_custom")
    purple_cmap = create_custom_colormap(purple_colors, "purple_custom")

    # Font sizes
    title_size = style["fonts"]["title_size"]
    axis_size = style["fonts"]["axis_size"]
    legend_size = style["fonts"]["legend_size"]
    ticks_size = style["fonts"]["ticks_size"]
    label_size = style["fonts"]["label_size"]

    # Label positions
    label_xpos = style["label"]["xpos"]
    label_ypos = style["label"]["ypos"]

    # Create figure with 2x2 panels
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
        2,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.1, "hspace": 0.15},
    )

    # Find homeostasis=True results for both experiments
    excitatory_key = None
    inhibitory_key = None

    for key in excitatory_data_dict.keys():
        if "homeostasis_True" in key:
            excitatory_key = key
            break

    for key in inhibitory_data_dict.keys():
        if "homeostasis_True" in key:
            inhibitory_key = key
            break

    if excitatory_key is None or inhibitory_key is None:
        raise ValueError("Could not find homeostasis=True results for both experiments")

    # Load results and create input generators
    excitatory_results = excitatory_data_dict[excitatory_key]
    inhibitory_results = inhibitory_data_dict[inhibitory_key]

    excitatory_results["parameters"]["device"] = torch.device("cpu")
    inhibitory_results["parameters"]["device"] = torch.device("cpu")

    excitatory_input_generator = create_input_generator_from_results(excitatory_results)
    inhibitory_input_generator = create_input_generator_from_results(inhibitory_results)

    # Find batch indices for target conditions
    try:
        excitatory_batch_idx = find_target_batch_index_2d(excitatory_input_generator)
        inhibitory_batch_idx = find_target_batch_index_2d(inhibitory_input_generator)
    except ValueError:
        # Fallback: assume the target conditions are in the first batch
        print(
            "Warning: Could not find exact batch indices, using first batch (index 0)"
        )
        excitatory_batch_idx = 0
        inhibitory_batch_idx = 0

    print(f"Using batch index {excitatory_batch_idx} for excitatory third factor")
    print(f"Using batch index {inhibitory_batch_idx} for inhibitory third factor")

    # Grid dimensions (assume 20x20 for 400 stimuli/neurons)
    grid_size = int(np.sqrt(400))  # Should be 20

    # Extract data for all panels
    # Panel A: Stimuli locations + center neuron tuning curve
    stimuli_locations = excitatory_input_generator.stimuli_locations  # [400, 2]
    neuron_locations = excitatory_input_generator.neuron_locations[
        excitatory_batch_idx
    ]  # [400, 2]

    # Find center neuron
    center_neuron_idx = find_center_neuron_index(neuron_locations)

    # Get center neuron's tuning curve (first repeat, target batch)
    excitatory_rates = excitatory_results["metrics_over_time"]["rates"][
        -1, 0, excitatory_batch_idx, :, :
    ]  # [N_I, num_stimuli]
    center_tuning_curve = excitatory_rates[center_neuron_idx, :]  # [400]

    # Reshape for heatmap
    center_tuning_2d = center_tuning_curve.reshape(grid_size, grid_size)
    stimuli_x = stimuli_locations[:, 0].reshape(grid_size, grid_size)
    stimuli_y = stimuli_locations[:, 1].reshape(grid_size, grid_size)

    # Panel B: Excitatory third factor
    excitatory_third_factor = excitatory_input_generator.excitatory_third_factor[
        excitatory_batch_idx
    ]  # [400]
    excitatory_third_factor_2d = excitatory_third_factor.reshape(grid_size, grid_size)

    # Panel C: Inhibitory density from excitatory experiment
    excitatory_metrics = excitatory_results["metrics_over_time"]
    # excitatory_density = excitatory_metrics["curves/density"][
    #     -1, 0, excitatory_batch_idx, :
    # ]  # [400]

    curves = curves_log(
        rates=excitatory_results["metrics_over_time"]["rates"][-1],
        input_generator=excitatory_input_generator,
        parameters=SimulationParameters(**excitatory_results["parameters"]),
    )
    excitatory_density = curves["curves/density"][0, excitatory_batch_idx, :]
    excitatory_density_2d = excitatory_density.reshape(grid_size, grid_size)

    # Panel D: Inhibitory density from inhibitory experiment
    inhibitory_metrics = inhibitory_results["metrics_over_time"]
    inhibitory_density = inhibitory_metrics["curves/density"][
        -1, 0, inhibitory_batch_idx, :
    ]  # [400]
    inhibitory_density_2d = inhibitory_density.reshape(grid_size, grid_size)

    # Create extent for proper axis scaling (-π to π)
    extent = [-np.pi, np.pi, -np.pi, np.pi]

    # Panel A: Tuning curve heatmap + stimuli dots
    im_a = ax_a.imshow(
        center_tuning_2d.cpu().numpy(), cmap="plasma", extent=extent, origin="lower"
    )
    ax_a.scatter(
        stimuli_x.cpu().numpy().flatten(),
        stimuli_y.cpu().numpy().flatten(),
        s=1,
        c="white",
        alpha=0.7,
    )

    # Panel B: Excitatory third factor heatmap
    im_b = ax_b.imshow(
        excitatory_third_factor_2d.cpu().numpy(),
        cmap=purple_cmap,
        extent=extent,
        origin="lower",
    )

    # Panel C: Inhibitory density from excitatory experiment
    im_c = ax_c.imshow(
        excitatory_density_2d.cpu().numpy(),
        cmap=blue_cmap,
        extent=extent,
        origin="lower",
    )

    # Panel D: Inhibitory density from inhibitory experiment
    im_d = ax_d.imshow(
        inhibitory_density_2d.cpu().numpy(),
        cmap=blue_cmap,
        extent=extent,
        origin="lower",
    )

    # Style all panels
    panels = [ax_a, ax_b, ax_c, ax_d]
    panel_labels = ["A", "B", "C", "D"]
    images = [im_a, im_b, im_c, im_d]

    for i, (ax, label, im) in enumerate(zip(panels, panel_labels, images)):
        # Remove axis labels and ticks as requested
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Panel label
        ax.text(
            label_xpos,
            label_ypos,
            label,
            transform=ax.transAxes,
            fontsize=label_size,
            fontweight="bold",
            va="top",
            ha="left",
            color="white",  # White text to show up on dark backgrounds
        )

    # Create figures directory and save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "third_factor_heatmaps.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_third_factor_heatmaps()
