import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append("..")

from plotting_utils import load_style, load_experiment_data


def plot_update_magnitudes():
    """Create update magnitudes plot comparing proportional allocation and low inhibition."""

    # Load style
    style = load_style()

    # Load data from both experiments
    prop_data_dict, prop_metadata, prop_time_values = load_experiment_data(
        "proportional_allocation"
    )
    inhib_data_dict, inhib_metadata, inhib_time_values = load_experiment_data(
        "inhibition_dominance"
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

    # Inhibition dominance data (k_I = 0.25)
    inhib_false = "homeostasis_False_k_I_0.25"  # Excitatory mass homeostasis
    inhib_true = "homeostasis_True_k_I_0.25"  # Variance homeostasis

    excitatory_scale = 10 * 100
    inhibitory_scale = 100 * 100

    # Panel A: Feedforward update magnitude, Excitatory Mass Homeostasis
    if prop_false in prop_data_dict:
        ax_a.plot(
            prop_time_values,
            (excitatory_scale / 25)
            * prop_data_dict[prop_false]["dynamics/feedforward_update_magnitude"].cpu(),
            color=green_colors["normal"],
            linewidth=line_width,
            label="Proportional Allocation",
        )

    if inhib_false in inhib_data_dict:
        ax_a.plot(
            inhib_time_values,
            (excitatory_scale / 0.25)
            * inhib_data_dict[inhib_false][
                "dynamics/feedforward_update_magnitude"
            ].cpu(),
            color=red_colors["normal"],
            linewidth=line_width,
            label="Low Inhibition",
        )

    ax_a.set_title(homeostasis_off_title, fontsize=title_size)

    # Panel B: Feedforward update magnitude, Variance Homeostasis
    if prop_true in prop_data_dict:
        ax_b.plot(
            prop_time_values,
            (excitatory_scale / 25)
            * prop_data_dict[prop_true]["dynamics/feedforward_update_magnitude"].cpu(),
            color=green_colors["normal"],
            linewidth=line_width,
            label="Proportional Allocation",
        )

    if inhib_true in inhib_data_dict:
        ax_b.plot(
            inhib_time_values,
            (excitatory_scale / 0.25)
            * inhib_data_dict[inhib_true][
                "dynamics/feedforward_update_magnitude"
            ].cpu(),
            color=red_colors["normal"],
            linewidth=line_width,
            label="Low Inhibition",
        )

    ax_b.set_title(homeostasis_on_title, fontsize=title_size)

    # Panel C: Recurrent update magnitude, Excitatory Mass Homeostasis
    if prop_false in prop_data_dict:
        ax_c.plot(
            prop_time_values,
            (inhibitory_scale / 25)
            * prop_data_dict[prop_false]["dynamics/recurrent_update_magnitude"].cpu(),
            color=green_colors["normal"],
            linewidth=line_width,
            label="Proportional Allocation",
        )

    if inhib_false in inhib_data_dict:
        ax_c.plot(
            inhib_time_values,
            (inhibitory_scale / 0.25)
            * inhib_data_dict[inhib_false]["dynamics/recurrent_update_magnitude"].cpu(),
            color=red_colors["normal"],
            linewidth=line_width,
            label="Low Inhibition",
        )

    # Panel D: Recurrent update magnitude, Variance Homeostasis
    if prop_true in prop_data_dict:
        ax_d.plot(
            prop_time_values,
            (inhibitory_scale / 25)
            * prop_data_dict[prop_true]["dynamics/recurrent_update_magnitude"].cpu(),
            color=green_colors["normal"],
            linewidth=line_width,
            label="Proportional Allocation",
        )

    if inhib_true in inhib_data_dict:
        ax_d.plot(
            inhib_time_values,
            (inhibitory_scale / 0.25)
            * inhib_data_dict[inhib_true]["dynamics/recurrent_update_magnitude"].cpu(),
            color=red_colors["normal"],
            linewidth=line_width,
            label="Low Inhibition",
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
            ax.set_ylim(0, 1.0)
        else:  # Top panels (A, B) - remove x-axis labels
            ax.set_xticklabels([])
            ax.tick_params(bottom=False)
            ax.set_ylim(0, 0.2)

        # Set axis limits
        # Use the longer time series for x-axis limit
        max_time = max(prop_time_values.max(), inhib_time_values.max())
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
