import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append("..")

from plotting_utils import load_style, load_experiment_data, plot_quantile_line


def plot_density_matching():
    """Create density matching plot with homeostasis comparison."""

    # Load style and data
    style = load_style()
    data_dict, metadata, time_values = load_experiment_data("proportional_allocation")

    # Extract style parameters
    fig_width = style["sizes"]["figure_width"]
    ratio = style["sizes"]["ratio"]

    # Calculate panel dimensions (two panels side by side)
    panel_width = fig_width / 2
    panel_height = panel_width * ratio
    fig_height = panel_height

    # Color schemes
    blue_colors = style["colours"]["blue"]
    red_colors = style["colours"]["red"]
    green_colors = style["colours"]["green"]

    # Alpha values
    alphas = {
        "light": style["colours"]["light_alpha"],
        "normal": style["colours"]["normal_alpha"],
        "dark": style["colours"]["dark_alpha"],
    }

    # Line width
    line_width = style["lines"]["normal_line_width"]

    # Font sizes
    title_size = style["fonts"]["title_size"]
    axis_size = style["fonts"]["axis_size"]
    legend_size = style["fonts"]["legend_size"]
    ticks_size = style["fonts"]["ticks_size"]
    label_size = style["fonts"]["label_size"]

    # Label positions
    label_xpos = style["label"]["xpos"]
    label_ypos = style["label"]["ypos"]

    # Load in the titles
    homeostasis_on_title = style["titles"]["homeostasis_on"]
    homeostasis_off_title = style["titles"]["homeostasis_off"]

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fig_width, fig_height), gridspec_kw={"wspace": 0.1}
    )

    # Left panel: homeostasis_False
    if "homeostasis_False" in data_dict:
        metrics_false = data_dict["homeostasis_False"]

        plot_quantile_line(
            ax1,
            time_values,
            metrics_false,
            "mode_diff/spectrum_vs_allocations",
            blue_colors,
            alphas,
            line_width,
            "Spectrum",
        )

        plot_quantile_line(
            ax1,
            time_values,
            metrics_false,
            "mode_diff/allocations_vs_uniform",
            red_colors,
            alphas,
            line_width,
            "Uniform",
        )

        plot_quantile_line(
            ax1,
            time_values,
            metrics_false,
            "mode_diff/allocations_vs_one_hot",
            green_colors,
            alphas,
            line_width,
            "One-hot",
        )

        ax1.set_title(homeostasis_off_title, fontsize=title_size)
    else:
        print("Warning: homeostasis_False data not found")

    # Right panel: homeostasis_True
    if "homeostasis_True" in data_dict:
        metrics_true = data_dict["homeostasis_True"]

        plot_quantile_line(
            ax2,
            time_values,
            metrics_true,
            "mode_diff/spectrum_vs_allocations",
            blue_colors,
            alphas,
            line_width,
            "Spectrum",
        )

        plot_quantile_line(
            ax2,
            time_values,
            metrics_true,
            "mode_diff/allocations_vs_uniform",
            red_colors,
            alphas,
            line_width,
            "Uniform",
        )

        plot_quantile_line(
            ax2,
            time_values,
            metrics_true,
            "mode_diff/allocations_vs_one_hot",
            green_colors,
            alphas,
            line_width,
            "One-hot",
        )

        ax2.set_title(homeostasis_on_title, fontsize=title_size)
    else:
        print("Warning: homeostasis_True data not found")

    # In the styling section, modify the loop to handle right panel differently:
    for i, ax in enumerate([ax1, ax2]):
        ax.set_xlabel("Time", fontsize=axis_size)
        if i == 0:  # Left panel only
            ax.set_ylabel("TV Distance", fontsize=axis_size)
        else:  # Right panel - remove y-axis label and ticks
            ax.set_yticklabels([])
            ax.tick_params(left=False)

        ax.tick_params(axis="both", which="major", labelsize=ticks_size)

        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, time_values.max())

        if style["elements"]["grid"]:
            ax.grid(True, alpha=0.3)

    # Only show legend on right panel to avoid duplication
    if style["elements"]["legend"]:
        ax2.legend(fontsize=legend_size, loc="best")

    # Add panel labels
    ax1.text(
        label_xpos,
        label_ypos,
        "A",
        transform=ax1.transAxes,
        fontsize=label_size,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax2.text(
        label_xpos,
        label_ypos,
        "B",
        transform=ax2.transAxes,
        fontsize=label_size,
        fontweight="bold",
        va="top",
        ha="left",
    )

    plt.tight_layout()

    # Create figures directory and save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "density_matching.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_density_matching()
