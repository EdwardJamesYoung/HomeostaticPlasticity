import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append("..")

from plotting_utils import load_style, load_experiment_data, plot_quantile_line


def plot_update_magnitudes():
    """Create update magnitudes plot comparing proportional allocation and low."""

    # Load style
    style = load_style()

    # Load data from both experiments
    prop_data_dict, prop_metadata, prop_time_values = load_experiment_data(
        "proportional_allocation"
    )
    time_scale_data_dict, time_scale_metadata, time_scale_time_values = (
        load_experiment_data("time_scale_ratio")
    )

    # Extract style parameters
    fig_width = style["sizes"]["figure_width"]
    ratio = style["sizes"]["ratio"]

    # Calculate panel dimensions (2x2 grid)
    panel_width = fig_width / 2
    panel_height = panel_width * ratio
    fig_height = 2 * panel_height

    # Color schemes
    green_colors = style["colours"]["green"]
    red_colors = style["colours"]["red"]

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

    alphas = {
        "light": style["colours"]["light_alpha"],
        "normal": style["colours"]["normal_alpha"],
        "dark": style["colours"]["dark_alpha"],
    }

    # Create figure with 2x2 panels
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
        2,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.1, "hspace": 0.15},
    )

    # Define the datasets we need
    # Proportional allocation data
    prop_false = "homeostasis_False"  # Excitatory mass homeostasis
    prop_true = "homeostasis_True"  # Variance homeostasis

    # time_scaleition dominance data
    time_scale_false = "homeostasis_False_tau_W_2.0"  # Excitatory mass homeostasis
    time_scale_true = "homeostasis_True_tau_W_2.0"  # Variance homeostasis

    # Panel A: Feedforward update magnitude, Excitatory Mass Homeostasis
    if prop_false in prop_data_dict:
        plot_quantile_line(
            ax_a,
            prop_time_values,
            metrics=prop_data_dict[prop_false],
            metric_base="dynamics/feedforward_update_magnitude",
            color_scheme=green_colors,
            alphas=alphas,
            line_width=line_width,
            label="Proportional Allocation",
        )

    if time_scale_false in time_scale_data_dict:
        plot_quantile_line(
            ax_a,
            time_scale_time_values,
            metrics=time_scale_data_dict[time_scale_false],
            metric_base="dynamics/feedforward_update_magnitude",
            color_scheme=red_colors,
            alphas=alphas,
            line_width=line_width,
            label="Low time_scaleition",
        )

    ax_a.set_title(homeostasis_off_title, fontsize=title_size)

    # Panel B: Feedforward update magnitude, Variance Homeostasis
    if prop_true in prop_data_dict:
        plot_quantile_line(
            ax_b,
            prop_time_values,
            metrics=prop_data_dict[prop_true],
            metric_base="dynamics/feedforward_update_magnitude",
            color_scheme=green_colors,
            alphas=alphas,
            line_width=line_width,
            label="Proportional Allocation",
        )

    if time_scale_true in time_scale_data_dict:
        plot_quantile_line(
            ax_b,
            time_scale_time_values,
            metrics=time_scale_data_dict[time_scale_true],
            metric_base="dynamics/feedforward_update_magnitude",
            color_scheme=red_colors,
            alphas=alphas,
            line_width=line_width,
            label="Low time_scaleition",
        )

        ax_b.set_title(homeostasis_on_title, fontsize=title_size)

    # Panel C: Recurrent update magnitude, Excitatory Mass Homeostasis
    if prop_false in prop_data_dict:
        plot_quantile_line(
            ax_c,
            prop_time_values,
            metrics=prop_data_dict[prop_false],
            metric_base="dynamics/recurrent_update_magnitude",
            color_scheme=green_colors,
            alphas=alphas,
            line_width=line_width,
            label="Proportional Allocation",
        )

    if time_scale_false in time_scale_data_dict:
        plot_quantile_line(
            ax_c,
            time_scale_time_values,
            metrics=time_scale_data_dict[time_scale_false],
            metric_base="dynamics/recurrent_update_magnitude",
            color_scheme=red_colors,
            alphas=alphas,
            line_width=line_width,
            label="Low time_scaleition",
        )

    if prop_true in prop_data_dict:
        plot_quantile_line(
            ax_d,
            prop_time_values,
            metrics=prop_data_dict[prop_true],
            metric_base="dynamics/recurrent_update_magnitude",
            color_scheme=green_colors,
            alphas=alphas,
            line_width=line_width,
            label="Proportional Allocation",
        )

    if time_scale_true in time_scale_data_dict:
        plot_quantile_line(
            ax_d,
            time_scale_time_values,
            metrics=time_scale_data_dict[time_scale_true],
            metric_base="dynamics/recurrent_update_magnitude",
            color_scheme=red_colors,
            alphas=alphas,
            line_width=line_width,
            label="Low time_scaleition",
        )

    # Style all panels
    panels = [ax_a, ax_b, ax_c, ax_d]
    panel_labels = ["A", "B", "C", "D"]

    for i, (ax, label) in enumerate(zip(panels, panel_labels)):
        # Y-axis labels (all panels keep y-axis labels and ticks)
        if i % 2 == 0:  # Left panels (A, C)
            ax.set_ylabel("Average Update Rate", fontsize=axis_size)

        # X-axis labels (only bottom panels)
        if i >= 2:  # Bottom panels (C, D)
            ax.set_xlabel("Time", fontsize=axis_size)

        # Set axis limits
        # Use the longer time series for x-axis limit
        max_time = max(prop_time_values.max(), time_scale_time_values.max())
        ax.set_xlim(0, max_time)

        # Tick sizes
        ax.tick_params(axis="both", which="major", labelsize=ticks_size)

        # Grid
        if style["elements"]["grid"]:
            ax.grid(True, alpha=0.3)

        # Legend (only on top-right panel)
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

    output_path = figures_dir / "update_magnitudes.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_update_magnitudes()
