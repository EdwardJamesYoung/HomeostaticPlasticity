import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append("..")

from plotting_utils import (
    load_style,
    load_experiment_data,
    get_final_metrics,
    create_boxplot_data,
    plot_matplotlib_boxplot,
    plot_quantile_line,
)


def extract_k_I_values(experiment_config):
    """Extract k_I values from experiment configuration."""
    param_grid = experiment_config.get("param_grid", {})
    k_I_values = param_grid.get("k_I", [])
    print(f"Extracted k_I values: {k_I_values}")
    return sorted(k_I_values)


def plot_inhibition_dominance():
    """Create inhibition dominance plot with 2x2 grid of panels."""

    # Load style and data
    style = load_style()
    data_dict, metadata, time_values = load_experiment_data("inhibition_dominance")

    # Extract k_I values from experiment config
    experiment_config = metadata["experiment_config"]
    k_I_values = extract_k_I_values(experiment_config)

    if not k_I_values:
        raise ValueError("No k_I values found in experiment configuration")

    print(f"Found k_I values: {k_I_values}")

    # Extract style parameters
    fig_width = style["sizes"]["figure_width"]
    ratio = style["sizes"]["ratio"]

    # Calculate panel dimensions (2x2 grid)
    panel_width = fig_width / 2
    panel_height = panel_width * ratio
    fig_height = 2 * panel_height

    # Color schemes
    blue_colors = style["colours"]["blue"]
    purple_colors = style["colours"]["purple"]
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
    ticks_size = style["fonts"]["ticks_size"]
    label_size = style["fonts"]["label_size"]
    legend_size = style["fonts"]["legend_size"]

    # Label positions
    label_xpos = style["label"]["xpos"]
    label_ypos = style["label"]["ypos"]

    # Load titles
    homeostasis_on_title = style["titles"]["homeostasis_on"]
    homeostasis_off_title = style["titles"]["homeostasis_off"]

    # Replace the subplot creation with:
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.15, wspace=0.1)
    fig = plt.figure(figsize=(fig_width, fig_height * 1.5))

    # Create subplots manually
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Add extra space before the bottom row
    gs_bottom = gridspec.GridSpec(1, 2, top=0.30, hspace=0.15, wspace=0.1)
    ax_e = fig.add_subplot(gs_bottom[0, 0])
    ax_f = fig.add_subplot(gs_bottom[0, 1])

    # Define metrics to extract
    tv_metrics = [f"mode_diff/variances_vs_spectrum_q{q}" for q in [10, 25, 50, 75, 90]]
    perplexity_metrics = [
        f"mode_diff/neuron_perplexities_q{q}" for q in [10, 25, 50, 75, 90]
    ]

    all_metrics = tv_metrics + perplexity_metrics

    # Extract final metrics for both homeostasis conditions
    final_metrics_false = get_final_metrics(data_dict, k_I_values, "False", all_metrics)
    final_metrics_true = get_final_metrics(data_dict, k_I_values, "True", all_metrics)

    # Create boxplot data
    tv_data_false = create_boxplot_data(
        final_metrics_false, "mode_diff/variances_vs_spectrum"
    )
    tv_data_true = create_boxplot_data(
        final_metrics_true, "mode_diff/variances_vs_spectrum"
    )
    perp_data_false = create_boxplot_data(
        final_metrics_false, "mode_diff/neuron_perplexities"
    )
    perp_data_true = create_boxplot_data(
        final_metrics_true, "mode_diff/neuron_perplexities"
    )

    # Log-scale x positions
    x_positions = np.log2(k_I_values)

    # Panel A: TV Distance, No Homeostasis
    plot_matplotlib_boxplot(ax_a, x_positions, tv_data_false, blue_colors, alphas)
    ax_a.set_title(homeostasis_off_title, fontsize=title_size)

    # Panel B: TV Distance, With Homeostasis
    plot_matplotlib_boxplot(ax_b, x_positions, tv_data_true, blue_colors, alphas)
    ax_b.set_title(homeostasis_on_title, fontsize=title_size)

    # Panel C: Perplexity, No Homeostasis
    plot_matplotlib_boxplot(ax_c, x_positions, perp_data_false, purple_colors, alphas)

    # Panel D: Perplexity, With Homeostasis
    plot_matplotlib_boxplot(ax_d, x_positions, perp_data_true, purple_colors, alphas)

    k_I_target = 0.25
    filename_false = f"homeostasis_False_k_I_{k_I_target}"
    filename_true = f"homeostasis_True_k_I_{k_I_target}"

    if filename_false in data_dict:
        metrics_false_025 = data_dict[filename_false]

        plot_quantile_line(
            ax_e,
            time_values,
            metrics_false_025,
            "mode_diff/variances_vs_spectrum",
            blue_colors,
            alphas,
            line_width,
            "Spectrum",
        )

        plot_quantile_line(
            ax_e,
            time_values,
            metrics_false_025,
            "mode_diff/variances_vs_uniform",
            red_colors,
            alphas,
            line_width,
            "Uniform",
        )

        plot_quantile_line(
            ax_e,
            time_values,
            metrics_false_025,
            "mode_diff/variances_vs_one_hot",
            green_colors,
            alphas,
            line_width,
            "One-hot",
        )

    # Panel F: Time series, k_I=0.25, With Homeostasis
    if filename_true in data_dict:
        metrics_true_025 = data_dict[filename_true]

        plot_quantile_line(
            ax_f,
            time_values,
            metrics_true_025,
            "mode_diff/variances_vs_spectrum",
            blue_colors,
            alphas,
            line_width,
            "Spectrum",
        )

        plot_quantile_line(
            ax_f,
            time_values,
            metrics_true_025,
            "mode_diff/variances_vs_uniform",
            red_colors,
            alphas,
            line_width,
            "Uniform",
        )

        plot_quantile_line(
            ax_f,
            time_values,
            metrics_true_025,
            "mode_diff/variances_vs_one_hot",
            green_colors,
            alphas,
            line_width,
            "One-hot",
        )

    # Style all panels
    panels = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]
    panel_labels = ["A", "B", "C", "D", "E", "F"]

    # Update the styling loop to handle the new panels:
    for i, (ax, label) in enumerate(zip(panels, panel_labels)):
        # Handle different panel types
        if i < 4:  # Panels A, B, C, D (box plots)
            # Y-axis labels (only left panels)
            if i % 2 == 0:  # Left panels (A, C)
                if i < 2:  # Top row (A)
                    ax.set_ylabel("TV Distance", fontsize=axis_size)
                else:  # Bottom row (C)
                    ax.set_ylabel("Average Perplexity", fontsize=axis_size)
            else:  # Right panels (B, D) - remove y-axis labels and ticks
                ax.set_yticklabels([])
                ax.tick_params(left=False)

            # X-axis setup for box plots
            ax.set_xlim(min(x_positions) - 0.2, max(x_positions) + 0.2)

            # X-axis labels (panels C, D, E, F get labels)
            if i >= 2:  # Panels C, D, E, F
                if i < 4:  # Box plots (C, D)
                    ax.set_xlabel("$k_I$", fontsize=axis_size)
                    ax.set_xticks(x_positions)  # Use log positions
                    ax.set_xticklabels(
                        [str(k) for k in k_I_values]
                    )  # But show original values
                # Time series x-axis handling is done in the else block above
            else:  # Top panels (A, B) - remove x-axis labels
                ax.set_xticks(x_positions)  # Still set tick positions
                ax.set_xticklabels([])
                ax.tick_params(bottom=False)

        else:  # Panels E, F (time series)
            # Y-axis labels (only left panel)
            if i % 2 == 0:  # Left panel (E)
                ax.set_ylabel("TV Distance", fontsize=axis_size)
            else:  # Right panel (F) - remove y-axis labels and ticks
                ax.set_yticklabels([])
                ax.tick_params(left=False)

            # X-axis for time series (always show since they're bottom row)
            ax.set_xlabel("Time", fontsize=axis_size)
            ax.set_xlim(0, time_values.max())
            ax.set_ylim(0, 1.0)

            # Legend (only on panel F)
            if i == 5 and style["elements"]["legend"]:  # Panel F
                ax.legend(fontsize=legend_size, loc="best")

        # Tick sizes
        ax.tick_params(axis="both", which="major", labelsize=ticks_size)

        # Grid
        if style["elements"]["grid"]:
            ax.grid(True, alpha=0.3)

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

    # Create figures directory and save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "inhibition_dominance.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_inhibition_dominance()
