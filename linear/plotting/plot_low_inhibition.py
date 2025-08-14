import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append("..")

from plotting_utils import load_style, load_experiment_data, plot_quantile_line


def plot_low_inhibition():
    """Create low inhibition plot with time series for k_I=0.25."""

    # Load style and data
    style = load_style()
    data_dict, metadata, time_values = load_experiment_data("inhibition_dominance")

    # Extract style parameters
    fig_width = style["sizes"]["figure_width"]
    ratio = style["sizes"]["ratio"]

    # Calculate panel dimensions (2x2 grid)
    panel_width = fig_width / 2
    panel_height = panel_width * ratio
    fig_height = 2 * panel_height

    # Color schemes
    blue_colors = style["colours"]["blue"]
    red_colors = style["colours"]["red"]
    green_colors = style["colours"]["green"]
    purple_colors = style["colours"]["purple"]

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
    ticks_size = style["fonts"]["ticks_size"]
    label_size = style["fonts"]["label_size"]
    legend_size = style["fonts"]["legend_size"]

    # Label positions
    label_xpos = style["label"]["xpos"]
    label_ypos = style["label"]["ypos"]

    # Load titles
    homeostasis_on_title = style["titles"]["homeostasis_on"]
    homeostasis_off_title = style["titles"]["homeostasis_off"]

    # Create figure with 2x2 panels
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
        2,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.1, "hspace": 0.15},
    )

    # Target k_I value
    k_I_target = 0.25
    filename_false = f"homeostasis_False_k_I_{k_I_target}"
    filename_true = f"homeostasis_True_k_I_{k_I_target}"

    # Panel A: TV distances, k_I=0.25, No Homeostasis
    if filename_false in data_dict:
        metrics_false = data_dict[filename_false]

        plot_quantile_line(
            ax_a,
            time_values,
            metrics_false,
            "mode_diff/variances_vs_spectrum",
            blue_colors,
            alphas,
            line_width,
            "Spectrum",
        )

        plot_quantile_line(
            ax_a,
            time_values,
            metrics_false,
            "mode_diff/variances_vs_uniform",
            red_colors,
            alphas,
            line_width,
            "Uniform",
        )

        plot_quantile_line(
            ax_a,
            time_values,
            metrics_false,
            "mode_diff/variances_vs_one_hot",
            green_colors,
            alphas,
            line_width,
            "One-hot",
        )

        ax_a.set_title(homeostasis_off_title, fontsize=title_size)
    else:
        print(f"Warning: {filename_false} data not found")

    # Panel B: TV distances, k_I=0.25, With Homeostasis
    if filename_true in data_dict:
        metrics_true = data_dict[filename_true]

        plot_quantile_line(
            ax_b,
            time_values,
            metrics_true,
            "mode_diff/variances_vs_spectrum",
            blue_colors,
            alphas,
            line_width,
            "Spectrum",
        )

        plot_quantile_line(
            ax_b,
            time_values,
            metrics_true,
            "mode_diff/variances_vs_uniform",
            red_colors,
            alphas,
            line_width,
            "Uniform",
        )

        plot_quantile_line(
            ax_b,
            time_values,
            metrics_true,
            "mode_diff/variances_vs_one_hot",
            green_colors,
            alphas,
            line_width,
            "One-hot",
        )

        ax_b.set_title(homeostasis_on_title, fontsize=title_size)
    else:
        print(f"Warning: {filename_true} data not found")

    # Panel C: Neuron perplexities, k_I=0.25, No Homeostasis
    if filename_false in data_dict:
        metrics_false = data_dict[filename_false]

        plot_quantile_line(
            ax_c,
            time_values,
            metrics_false,
            "mode_diff/neuron_perplexities",
            purple_colors,
            alphas,
            line_width,
            "Perplexity",
        )

    # Panel D: Neuron perplexities, k_I=0.25, With Homeostasis
    if filename_true in data_dict:
        metrics_true = data_dict[filename_true]

        plot_quantile_line(
            ax_d,
            time_values,
            metrics_true,
            "mode_diff/neuron_perplexities",
            purple_colors,
            alphas,
            line_width,
            "Perplexity",
        )

    # Style all panels
    panels = [ax_a, ax_b, ax_c, ax_d]
    panel_labels = ["A", "B", "C", "D"]

    for i, (ax, label) in enumerate(zip(panels, panel_labels)):
        # Y-axis labels (only left panels)
        if i % 2 == 0:  # Left panels (A, C)
            if i < 2:  # Top row (A)
                ax.set_ylabel("TV Distance", fontsize=axis_size)
            else:  # Bottom row (C)
                ax.set_ylabel("Average Perplexity", fontsize=axis_size)
        else:  # Right panels (B, D) - remove y-axis labels and ticks
            ax.set_yticklabels([])
            ax.tick_params(left=False)

        # X-axis labels (only bottom panels)
        if i >= 2:  # Bottom panels (C, D)
            ax.set_xlabel("Time", fontsize=axis_size)
        else:  # Top panels (A, B) - remove x-axis labels
            ax.set_xticklabels([])
            ax.tick_params(bottom=False)

        # Set axis limits
        ax.set_xlim(0, time_values.max())
        if i < 2:  # Top row - TV Distance
            ax.set_ylim(0, 1.0)
        # Bottom row - let perplexity auto-scale
        else:
            ax.set_ylim(1, 6)

        # Tick sizes
        ax.tick_params(axis="both", which="major", labelsize=ticks_size)

        # Grid
        if style["elements"]["grid"]:
            ax.grid(True, alpha=0.3)

        # Legend (only on panel B)
        if i == 1 and style["elements"]["legend"]:  # Panel B
            ax.legend(fontsize=legend_size, loc="best")

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
        )

    plt.tight_layout()

    # Create figures directory and save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "low_inhibition.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_low_inhibition()
