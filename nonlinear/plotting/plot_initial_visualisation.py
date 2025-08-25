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
    extract_final_curves,
    extract_final_tuning_curves,
    plot_stimulus_curves,
)


def plot_initial_visualisation():
    """Create initial visualisation plot comparing homeostasis conditions."""

    # Load style and data
    style = load_style()
    data_dict, metadata_dict = load_experiment_data("initial_visualisation")

    # Extract style parameters
    fig_width = style["sizes"]["figure_width"]
    ratio = style["sizes"]["ratio"]

    # Calculate panel dimensions (two panels side by side)
    panel_width = fig_width / 2
    panel_height = panel_width * ratio
    fig_height = panel_height

    # Color schemes
    red_colors = style["colours"]["red"]
    blue_colors = style["colours"]["blue"]
    green_colors = style["colours"]["green"]
    grey_colors = style["colours"]["grey"]

    # Font sizes
    title_size = style["fonts"]["title_size"]
    axis_size = style["fonts"]["axis_size"]
    legend_size = style["fonts"]["legend_size"]
    ticks_size = style["fonts"]["ticks_size"]
    label_size = style["fonts"]["label_size"]

    # Label positions
    label_xpos = style["label"]["xpos"]
    label_ypos = style["label"]["ypos"]

    # Load titles
    homeostasis_on_title = "Rate Homeostasis"
    homeostasis_off_title = style["titles"]["homeostasis_off"]

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fig_width, fig_height), gridspec_kw={"wspace": 0.1}
    )

    # Find the two parameter combinations
    homeostasis_false_key = None
    homeostasis_true_key = None

    for key in data_dict.keys():
        if "homeostasis_False" in key:
            homeostasis_false_key = key
        elif "homeostasis_True" in key:
            homeostasis_true_key = key

    if homeostasis_false_key is None or homeostasis_true_key is None:
        raise ValueError("Could not find both homeostasis conditions in data")

    # Process each panel
    panels = [
        (ax1, homeostasis_false_key, homeostasis_off_title, "A"),
        (ax2, homeostasis_true_key, homeostasis_on_title, "B"),
    ]

    for ax, data_key, title, panel_label in panels:
        # Load results for this condition
        results = data_dict[data_key]
        metrics_over_time = results["metrics_over_time"]

        # Recreate input generator to get probability distribution and stimulus locations
        input_generator = create_input_generator_from_results(results)
        stimulus_locations = input_generator.stimuli_locations  # [num_stimuli, 1]
        probability_dist = (
            input_generator.stimuli_probabilities
        )  # [batch_size, num_stimuli]

        # Extract final curves
        final_curves = extract_final_curves(
            metrics_over_time, ["curves/density", "curves/gains"]
        )

        # Extract final tuning curves
        final_tuning_curves = extract_final_tuning_curves(metrics_over_time)

        prob_data = probability_dist.squeeze()
        density_data = final_curves["curves/density"].squeeze()
        gains_data = final_curves["curves/gains"].squeeze()

        prob_normalised = prob_data / prob_data.mean()
        density_normalised = density_data / density_data.mean()
        gains_normalised = gains_data / gains_data.mean()

        scaled_tuning_curves = final_tuning_curves / gains_data.mean()

        # Prepare curves for plotting
        curves_to_plot = {
            "probability": (prob_normalised, red_colors["normal"], r"$p(s)$"),
            "density": (density_normalised, blue_colors["normal"], r"$d_I(s)$"),
            "gains": (gains_normalised, green_colors["normal"], r"$g_I(s)$"),
        }

        # Plot all curves
        plot_stimulus_curves(
            ax=ax,
            stimulus_locations=stimulus_locations,
            curves_dict=curves_to_plot,
            individual_curves=scaled_tuning_curves,
            individual_color=grey_colors["normal"],
            individual_alpha=0.7,
        )

        # Style the axis
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel(r"Input stimulus, $s$", fontsize=axis_size)

        if panel_label == "B":
            ax.set_yticklabels([])
            ax.tick_params(left=False)

        # Set x-axis limits
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, 2.25)

        # Set x-axis ticks
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        # Tick sizes
        ax.tick_params(axis="both", which="major", labelsize=ticks_size)

        # Grid
        if style["elements"]["grid"]:
            ax.grid(True, alpha=0.3)

        # Legend (only on right panel)
        if panel_label == "B" and style["elements"]["legend"]:
            ax.legend(fontsize=legend_size, loc="best")

        # Panel label
        ax.text(
            label_xpos,
            label_ypos,
            panel_label,
            transform=ax.transAxes,
            fontsize=label_size,
            fontweight="bold",
            va="top",
            ha="left",
        )

    # Create figures directory and save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "initial_visualisation.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_initial_visualisation()
