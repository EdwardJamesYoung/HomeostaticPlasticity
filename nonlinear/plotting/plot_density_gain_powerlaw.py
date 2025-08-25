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
    extract_homeostasis_power_values_from_analysis,
    create_boxplot_data_from_tensors,
    plot_custom_boxplot,
    compute_linear_regression,
)


def plot_density_gain_powerlaw():
    """Create density/gain power law analysis plot with homeostasis power scaling."""

    # Load style and data
    style = load_style()
    analysis_dict = load_power_law_analysis_data("density_gain_power_law")

    # Extract homeostasis_power values for rectified_linear activation function
    homeostasis_power_values = extract_homeostasis_power_values_from_analysis(
        analysis_dict
    )
    # homeostasis_power_values.remove(2.0)
    print(f"Found homeostasis_power values: {homeostasis_power_values}")

    # Calculate 1/homeostasis_power for x-axis
    inv_homeostasis_power_values = [1.0 / hp for hp in homeostasis_power_values]

    # Extract style parameters
    fig_width = style["sizes"]["figure_width"]
    ratio = style["sizes"]["ratio"]

    # Calculate panel dimensions (1x2 grid)
    panel_width = fig_width / 2
    panel_height = panel_width * ratio
    fig_height = panel_height

    # Color schemes
    blue_colors = style["colours"]["blue"]
    green_colors = style["colours"]["green"]
    grey_colors = style["colours"]["grey"]

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

    # Create figure with 1x2 panels
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(fig_width, fig_height), gridspec_kw={"wspace": 0.2}
    )

    # Prepare data for each homeostasis_power value
    d_I_p_gamma_data = []
    g_I_p_gamma_data = []

    for hp in homeostasis_power_values:
        # Find the analysis file for this homeostasis_power value
        param_name = f"activation_function_name_rectified_linear_homeostasis_power_{hp}"

        if param_name in analysis_dict:
            results = analysis_dict[param_name]

            # Extract data for d_I vs p
            if "d_I_vs_p" in results:
                d_I_p_gamma_data.append(results["d_I_vs_p"]["gammas"])
            else:
                print(f"Warning: d_I_vs_p not found for {param_name}")
                d_I_p_gamma_data.append(torch.tensor([]))

            # Extract data for g_I vs p
            if "g_I_vs_p" in results:
                g_I_p_gamma_data.append(results["g_I_vs_p"]["gammas"])
            else:
                print(f"Warning: g_I_vs_p not found for {param_name}")
                g_I_p_gamma_data.append(torch.tensor([]))
        else:
            print(f"Warning: No analysis data found for homeostasis_power = {hp}")
            # Add empty tensors for missing data
            d_I_p_gamma_data.append(torch.tensor([]))
            g_I_p_gamma_data.append(torch.tensor([]))

    # Create boxplot data for each homeostasis_power value
    d_I_gamma_boxplots = []
    g_I_gamma_boxplots = []

    # Also collect median values for linear regression
    d_I_medians = []
    g_I_medians = []

    for i in range(len(homeostasis_power_values)):
        if d_I_p_gamma_data[i].numel() > 0:
            boxplot_data = create_boxplot_data_from_tensors([d_I_p_gamma_data[i]])
            d_I_gamma_boxplots.append(boxplot_data)
            d_I_medians.append(boxplot_data["q50"])
        else:
            # Create empty boxplot data for missing values
            empty_data = {"q10": 0, "q25": 0, "q50": 0, "q75": 0, "q90": 0}
            d_I_gamma_boxplots.append(empty_data)
            d_I_medians.append(0)

        if g_I_p_gamma_data[i].numel() > 0:
            boxplot_data = create_boxplot_data_from_tensors([g_I_p_gamma_data[i]])
            g_I_gamma_boxplots.append(boxplot_data)
            g_I_medians.append(boxplot_data["q50"])
        else:
            empty_data = {"q10": 0, "q25": 0, "q50": 0, "q75": 0, "q90": 0}
            g_I_gamma_boxplots.append(empty_data)
            g_I_medians.append(0)

    # Compute linear regressions
    d_I_slope, d_I_intercept, d_I_r2 = compute_linear_regression(
        inv_homeostasis_power_values, d_I_medians
    )
    g_I_slope, g_I_intercept, g_I_r2 = compute_linear_regression(
        inv_homeostasis_power_values, g_I_medians
    )

    # X-axis positions (linear spacing)
    x_positions = inv_homeostasis_power_values

    # Create regression lines
    x_line = np.linspace(0, max(x_positions), 100)
    d_I_y_line = d_I_slope * x_line + d_I_intercept
    g_I_y_line = g_I_slope * x_line + g_I_intercept

    # Panel A: d_I vs p gamma values (blue)
    # Plot regression line first (behind boxplots)
    ax_a.plot(
        x_line,
        d_I_y_line,
        color=grey_colors["normal"],
        linewidth=line_width,
        linestyle="-",
        alpha=1.0,
        zorder=1,
    )
    plot_custom_boxplot(
        ax_a,
        x_positions,
        d_I_gamma_boxplots,
        blue_colors,
        alphas,
        line_width,
        widths=0.1,
    )

    # Panel B: g_I vs p gamma values (green)
    # Plot regression line first (behind boxplots)
    ax_b.plot(
        x_line,
        g_I_y_line,
        color=grey_colors["normal"],
        linewidth=line_width,
        linestyle="-",
        alpha=1.0,
        zorder=1,
    )
    plot_custom_boxplot(
        ax_b,
        x_positions,
        g_I_gamma_boxplots,
        green_colors,
        alphas,
        line_width,
        widths=0.1,
    )

    # Style both panels
    panels = [ax_a, ax_b]
    panel_labels = ["A", "B"]
    r_squared_values = [d_I_r2, g_I_r2]
    slope_values = [d_I_slope, g_I_slope]

    ax_a.set_ylim(-0.5, 2.5)
    ax_b.set_ylim(-1.0, 0.5)

    for i, (ax, label, r2_val, slope_val) in enumerate(
        zip(panels, panel_labels, r_squared_values, slope_values)
    ):
        # Y-axis label (only left panel)
        if i == 0:
            ax.set_ylabel(r"$\gamma$", fontsize=axis_size)

        # X-axis setup
        ax.set_xlim(0, max(x_positions) + 0.05)
        ax.set_xlabel(r"$1/\eta$", fontsize=axis_size)

        # Tick sizes
        ax.tick_params(axis="both", which="major", labelsize=ticks_size)

        # Grid
        if style["elements"]["grid"]:
            ax.grid(True, alpha=0.3)

        # Panel label (outside, top-left)
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

        # R² and slope values (inside panel, top-left)
        ax.text(
            0.05,
            0.95,
            f"$R^2 = {r2_val:.2f}, \\beta = {slope_val:.2f}$",
            transform=ax.transAxes,
            fontsize=axis_size,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Create figures directory and save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "density_gain_powerlaw.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")
    print(f"d_I vs p: R² = {d_I_r2:.3f}, β = {d_I_slope:.3f}")
    print(f"g_I vs p: R² = {g_I_r2:.3f}, β = {g_I_slope:.3f}")


if __name__ == "__main__":
    plot_density_gain_powerlaw()
