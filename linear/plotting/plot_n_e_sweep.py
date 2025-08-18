import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Add parent directory to path for imports
sys.path.append("..")

from plotting_utils import (
    load_style,
    load_experiment_data,
    get_final_metrics,
    create_boxplot_data,
    plot_matplotlib_boxplot,
)


def extract_N_E_values(experiment_config):
    """Extract N_E values from experiment configuration."""
    param_grid = experiment_config.get("param_grid", {})
    N_E_values = param_grid.get("N_E", [])
    print(f"Extracted N_E values: {N_E_values}")
    return sorted(N_E_values)


def compute_linear_regression(x_values, y_values):
    """
    Compute linear regression and return slope, intercept, and R².

    Args:
        x_values: Independent variable (N_E values)
        y_values: Dependent variable (median perplexity values)

    Returns:
        slope, intercept, r_squared
    """
    # Reshape for sklearn
    X = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values)

    # Fit linear regression
    reg = LinearRegression()
    reg.fit(X, y)

    # Predict values for R² calculation
    y_pred = reg.predict(X)
    r_squared = r2_score(y, y_pred)

    return reg.coef_[0], reg.intercept_, r_squared


def plot_N_E_sweep():
    """Create N_E sweep plot with 1x2 grid of panels and linear regression."""

    # Load style and data
    style = load_style()
    data_dict, metadata, time_values = load_experiment_data("mode_number")

    # Extract N_E values from experiment config
    experiment_config = metadata["experiment_config"]
    N_E_values = extract_N_E_values(experiment_config)

    if not N_E_values:
        raise ValueError("No N_E values found in experiment configuration")

    print(f"Found N_E values: {N_E_values}")

    # Extract style parameters
    fig_width = style["sizes"]["figure_width"]
    ratio = style["sizes"]["ratio"]

    # Calculate panel dimensions (1x2 grid)
    panel_width = fig_width / 2
    panel_height = panel_width * ratio
    fig_height = panel_height

    # Color schemes
    purple_colors = style["colours"]["purple"]
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

    # Load titles
    homeostasis_on_title = style["titles"]["homeostasis_on"]
    homeostasis_off_title = style["titles"]["homeostasis_off"]

    # Create figure with 1x2 panels
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.1, "hspace": 0.15},
    )

    # Define metrics to extract
    perplexity_metrics = [
        f"statistics/neuron_perplexities_q{q}" for q in [10, 25, 50, 75, 90]
    ]

    all_metrics = perplexity_metrics

    # Extract final metrics for both homeostasis conditions
    final_metrics_false = get_final_metrics(
        data_dict=data_dict,
        homeostasis_value=False,
        metric_names=all_metrics,
        N_E_values=N_E_values,
    )
    final_metrics_true = get_final_metrics(
        data_dict=data_dict,
        homeostasis_value=True,
        metric_names=all_metrics,
        N_E_values=N_E_values,
    )

    perp_data_false = create_boxplot_data(
        final_metrics_false, "statistics/neuron_perplexities"
    )
    perp_data_true = create_boxplot_data(
        final_metrics_true, "statistics/neuron_perplexities"
    )

    # Compute linear regressions first
    median_false = perp_data_false["q50"]
    slope_false, intercept_false, r2_false = compute_linear_regression(
        N_E_values, median_false
    )

    median_true = perp_data_true["q50"]
    slope_true, intercept_true, r2_true = compute_linear_regression(
        N_E_values, median_true
    )

    # Create regression line
    x_line = np.linspace(min(N_E_values), max(N_E_values), 100)
    y_line_false = slope_false * x_line + intercept_false
    y_line_true = slope_true * x_line + intercept_true

    # Plot regression lines FIRST (so they appear behind)
    ax_a.plot(
        x_line,
        y_line_false,
        color=grey_colors["normal"],
        linewidth=line_width,
        linestyle="-",
        alpha=1.0,
        zorder=1,
    )
    ax_b.plot(
        x_line,
        y_line_true,
        color=grey_colors["normal"],
        linewidth=line_width,
        linestyle="-",
        alpha=1.0,
        zorder=1,
    )

    plot_matplotlib_boxplot(
        ax_a, N_E_values, perp_data_false, purple_colors, alphas, widths=2
    )
    plot_matplotlib_boxplot(
        ax_b, N_E_values, perp_data_true, purple_colors, alphas, widths=2
    )

    # Set titles
    ax_a.set_title(homeostasis_off_title, fontsize=title_size)
    ax_b.set_title(homeostasis_on_title, fontsize=title_size)

    # Style all panels
    panels = [ax_a, ax_b]
    panel_labels = ["A", "B"]
    r_squared_values = [r2_false, r2_true]
    slope_values = [slope_false, slope_true]

    # And in the loop, update the zip to include slopes:
    for i, (ax, label, r2_val, slope_val) in enumerate(
        zip(panels, panel_labels, r_squared_values, slope_values)
    ):
        if i % 2 == 0:
            ax.set_ylabel("Average Perplexity", fontsize=axis_size)
        else:
            ax.set_yticklabels([])
            ax.tick_params(left=False)

        ax.set_ylim(0, 20)

        # X-axis setup for box plots
        ax.set_xlim(min(N_E_values) - 2.5, max(N_E_values) + 2.5)

        ax.set_xlabel("$N_E$", fontsize=axis_size)
        ax.set_xticks(N_E_values)
        ax.set_xticklabels([str(k) for k in N_E_values])

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
    plt.tight_layout()

    # Create figures directory and save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "N_E_sweep.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")
    print(f"Panel A R² = {r2_false:.3f}, Panel B R² = {r2_true:.3f}")


if __name__ == "__main__":
    plot_N_E_sweep()
