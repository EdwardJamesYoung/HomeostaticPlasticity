import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append("..")

from plotting_utils import (
    load_style,
    load_power_law_analysis_data,
    create_boxplot_data_from_tensors,
    plot_custom_boxplot,
)


def extract_tau_W_values_from_analysis(analysis_dict):
    """Extract and sort tau_W values from analysis file names."""
    tau_W_values = []

    for param_name in analysis_dict.keys():
        # Extract tau_W value from parameter name like "T_2000.0_tau_W_0.125_tau_k_16.0_tau_product_4.0"
        parts = param_name.split("_")
        for i, part in enumerate(parts):
            if part == "tau" and i + 1 < len(parts) and parts[i + 1] == "W":
                if i + 2 < len(parts):
                    try:
                        tau_W = float(parts[i + 2])
                        tau_W_values.append(tau_W)
                        break
                    except ValueError:
                        continue

    return sorted(list(set(tau_W_values)))  # Remove duplicates and sort


def plot_time_scale_ratio_powerlaw():
    """Create time scale ratio power law analysis plot with 2x2 grid."""

    # Load style and data
    style = load_style()
    analysis_dict = load_power_law_analysis_data("time_scale_ratio")

    # Extract tau_W values
    tau_W_values = extract_tau_W_values_from_analysis(analysis_dict)
    print(f"Found tau_W values: {tau_W_values}")

    # Extract style parameters
    fig_width = style["sizes"]["figure_width"]
    ratio = style["sizes"]["ratio"]

    # Calculate panel dimensions (2x2 grid)
    panel_width = fig_width / 2
    panel_height = panel_width * ratio
    fig_height = 2 * panel_height

    # Color schemes
    blue_colors = style["colours"]["blue"]
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

    # Label positions
    label_xpos = style["label"]["xpos"]
    label_ypos = style["label"]["ypos"]

    # Create figure with 2x2 panels
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
        2,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.17, "hspace": 0.17},
    )

    # Prepare data for each tau_W value
    d_I_p_gamma_data = []
    g_I_p_gamma_data = []
    d_I_p_l1_data = []
    g_I_p_l1_data = []

    for tau_W in tau_W_values:
        # Find the analysis file for this tau_W value
        param_name = f"T_2000.0_tau_W_{tau_W}_tau_k_16.0_tau_product_4.0"

        if param_name in analysis_dict:
            results = analysis_dict[param_name]

            # Extract data for d_I vs p
            if "d_I_vs_p" in results:
                d_I_p_gamma_data.append(results["d_I_vs_p"]["gammas"])
                d_I_p_l1_data.append(results["d_I_vs_p"]["l1_distances"])
            else:
                print(f"Warning: d_I_vs_p not found for {param_name}")
                d_I_p_gamma_data.append(torch.tensor([]))
                d_I_p_l1_data.append(torch.tensor([]))

            # Extract data for g_I vs p
            if "g_I_vs_p" in results:
                g_I_p_gamma_data.append(results["g_I_vs_p"]["gammas"])
                g_I_p_l1_data.append(results["g_I_vs_p"]["l1_distances"])
            else:
                print(f"Warning: g_I_vs_p not found for {param_name}")
                g_I_p_gamma_data.append(torch.tensor([]))
                g_I_p_l1_data.append(torch.tensor([]))
        else:
            print(f"Warning: No analysis data found for tau_W = {tau_W}")
            # Add empty tensors for missing data
            d_I_p_gamma_data.append(torch.tensor([]))
            g_I_p_gamma_data.append(torch.tensor([]))
            d_I_p_l1_data.append(torch.tensor([]))
            g_I_p_l1_data.append(torch.tensor([]))

    # Create boxplot data for each tau_W value
    d_I_gamma_boxplots = []
    g_I_gamma_boxplots = []
    d_I_l1_boxplots = []
    g_I_l1_boxplots = []

    for i in range(len(tau_W_values)):
        if d_I_p_gamma_data[i].numel() > 0:
            d_I_gamma_boxplots.append(
                create_boxplot_data_from_tensors([d_I_p_gamma_data[i]])
            )
            d_I_l1_boxplots.append(create_boxplot_data_from_tensors([d_I_p_l1_data[i]]))
        else:
            # Create empty boxplot data for missing values
            empty_data = {"q10": 0, "q25": 0, "q50": 0, "q75": 0, "q90": 0}
            d_I_gamma_boxplots.append(empty_data)
            d_I_l1_boxplots.append(empty_data)

        if g_I_p_gamma_data[i].numel() > 0:
            g_I_gamma_boxplots.append(
                create_boxplot_data_from_tensors([g_I_p_gamma_data[i]])
            )
            g_I_l1_boxplots.append(create_boxplot_data_from_tensors([g_I_p_l1_data[i]]))
        else:
            empty_data = {"q10": 0, "q25": 0, "q50": 0, "q75": 0, "q90": 0}
            g_I_gamma_boxplots.append(empty_data)
            g_I_l1_boxplots.append(empty_data)

    # Log-scale x positions
    x_positions = np.log2(tau_W_values)

    ax_a.set_title(r"$d_I \propto p^\gamma$")

    # Panel A: d_I vs p gamma values
    plot_custom_boxplot(
        ax_a,
        x_positions,
        d_I_gamma_boxplots,
        blue_colors,
        alphas,
        line_width,
        widths=0.2,
    )
    ax_a.set_ylim(0, 2.5)
    ax_a.axhline(y=2, color="black", linestyle="--", linewidth=1)

    ax_b.set_title(r"$g_I \propto p^\gamma$")

    # Panel B: g_I vs p gamma values
    plot_custom_boxplot(
        ax_b,
        x_positions,
        g_I_gamma_boxplots,
        green_colors,
        alphas,
        line_width,
        widths=0.2,
    )

    ax_b.set_ylim(-1.5, 0.5)
    ax_b.axhline(y=-1, color="black", linestyle="--", linewidth=1)

    # Panel C: d_I vs p l1 distances
    plot_custom_boxplot(
        ax_c, x_positions, d_I_l1_boxplots, blue_colors, alphas, line_width, widths=0.2
    )

    # Panel D: g_I vs p l1 distances
    plot_custom_boxplot(
        ax_d, x_positions, g_I_l1_boxplots, green_colors, alphas, line_width, widths=0.2
    )

    # Style all panels
    panels = [ax_a, ax_b, ax_c, ax_d]
    panel_labels = ["A", "B", "C", "D"]

    for i, (ax, label) in enumerate(zip(panels, panel_labels)):
        # Y-axis labels
        if i < 2:  # Top row (gamma)
            if i % 2 == 0:  # Left panel
                ax.set_ylabel(r"Power law exponent, $\gamma$", fontsize=axis_size)
        else:  # Bottom row (l1 distance)
            if i % 2 == 0:  # Left panel
                ax.set_ylabel(r"$\ell_1$ distance", fontsize=axis_size)

        # X-axis setup
        ax.set_xlim(min(x_positions) - 0.2, max(x_positions) + 0.2)

        # X-axis labels (only bottom panels)
        if i >= 2:  # Bottom panels (C, D)
            ax.set_xlabel(r"$\tau_W$", fontsize=axis_size)
            ax.set_ylim(0, 0.1)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(tau) for tau in tau_W_values])
        else:  # Top panels (A, B) - remove x-axis labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels([])
            ax.tick_params(bottom=False)

        # Tick sizes - enable ticks for both left and right panels
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

    output_path = figures_dir / "time_scale_ratio_powerlaw.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_time_scale_ratio_powerlaw()
