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
    plot_stimulus_curves,
)


def find_target_batch_index(
    input_generator, target_mixing_parameter=0.4, target_concentration=1.0
):
    """
    Find the batch index corresponding to the target mixing parameter and concentration.

    Args:
        input_generator: The input generator object
        target_mixing_parameter: Target mixing parameter value
        target_concentration: Target concentration value

    Returns:
        batch_idx: Index of the matching batch element
    """
    # Check the appropriate config based on experiment type
    # For input_density_modulation, check density_config
    # For input_gain_modulation, check gain_config

    # Try density_config first
    if input_generator.density_config.batch_size > 1:
        config = input_generator.density_config
        mixing_params = config.mixing_parameter.squeeze()
        concentrations = config.concentration.squeeze()

        for i in range(len(mixing_params)):
            if (
                abs(mixing_params[i].item() - target_mixing_parameter) < 1e-6
                and abs(concentrations[i].item() - target_concentration) < 1e-6
            ):
                return i

    # Try gain_config if density_config didn't work
    elif input_generator.gain_config.batch_size > 1:
        config = input_generator.gain_config
        mixing_params = config.mixing_parameter.squeeze()
        concentrations = config.concentration.squeeze()

        for i in range(len(mixing_params)):
            if (
                abs(mixing_params[i].item() - target_mixing_parameter) < 1e-6
                and abs(concentrations[i].item() - target_concentration) < 1e-6
            ):
                return i

    raise ValueError(
        f"Could not find batch index for mixing_parameter={target_mixing_parameter}, concentration={target_concentration}"
    )


def normalise_curves(density, gains, individual_curves):
    density_normalised = density / density.mean()
    gain_normalisation_factor = gains.mean()
    gains_normalised = gains / gain_normalisation_factor
    individual_curves_normalised = individual_curves / gain_normalisation_factor

    return density_normalised, gains_normalised, individual_curves_normalised


def plot_input_modulation_comparison():
    """Create input modulation comparison plot with 2x2 grid."""

    # Load style
    style = load_style()

    # Load data from both experiments
    density_data_dict, density_metadata_dict = load_experiment_data(
        "input_density_modulation"
    )
    gain_data_dict, gain_metadata_dict = load_experiment_data("input_gain_modulation")

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

    # Create figure with 2x2 panels
    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
        2,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"wspace": 0.1, "hspace": 0.15},
    )

    # Find homeostasis=True results for both experiments
    density_key = None
    gain_key = None

    for key in density_data_dict.keys():
        if "homeostasis_True" in key:
            density_key = key
            break

    for key in gain_data_dict.keys():
        if "homeostasis_True" in key:
            gain_key = key
            break

    if density_key is None or gain_key is None:
        raise ValueError("Could not find homeostasis=True results for both experiments")

    # Load results and create input generators
    density_results = density_data_dict[density_key]
    gain_results = gain_data_dict[gain_key]

    density_input_generator = create_input_generator_from_results(density_results)
    gain_input_generator = create_input_generator_from_results(gain_results)

    # Find batch indices for target conditions
    density_batch_idx = find_target_batch_index(density_input_generator)
    gain_batch_idx = find_target_batch_index(gain_input_generator)

    print(f"Using batch index {density_batch_idx} for density modulation")
    print(f"Using batch index {gain_batch_idx} for gain modulation")

    # Extract stimulus locations (same for all)
    stimulus_locations = density_input_generator.stimuli_locations

    # Panel A: Input density modulation - Excitatory (Input patterns)
    density_input_patterns = density_input_generator.stimuli_patterns[
        density_batch_idx
    ]  # [N_E, num_stimuli]
    density_input_density = density_input_generator.input_density[
        density_batch_idx
    ]  # [num_stimuli]
    density_input_gains = density_input_generator.input_gains[
        density_batch_idx
    ]  # [num_stimuli]

    density_input_density, density_input_gains, density_input_patterns = (
        normalise_curves(
            density_input_density, density_input_gains, density_input_patterns
        )
    )

    curves_a = {
        "input_density": (
            density_input_density,
            blue_colors["normal"],
            "Excitatory density, $d_E$",
        ),
        "input_gains": (
            density_input_gains,
            green_colors["normal"],
            "Excitatory gains, $g_E$",
        ),
    }

    plot_stimulus_curves(
        ax=ax_a,
        stimulus_locations=stimulus_locations,
        curves_dict=curves_a,
        individual_curves=density_input_patterns / 10.0,
        individual_color=grey_colors["normal"],
        individual_alpha=0.7,
    )

    # Panel B: Input density modulation - Inhibitory (Neural responses)
    density_metrics = density_results["metrics_over_time"]
    density_neural_rates = density_metrics["rates"][
        -1, 0, density_batch_idx, :, :
    ]  # [N_I, num_stimuli]
    density_neural_density = density_metrics["curves/density"][
        -1, 0, density_batch_idx, :
    ]  # [num_stimuli]
    density_neural_gains = density_metrics["curves/gains"][
        -1, 0, density_batch_idx, :
    ]  # [num_stimuli]

    density_neural_density, density_neural_gains, density_neural_rates = (
        normalise_curves(
            density_neural_density, density_neural_gains, density_neural_rates
        )
    )

    curves_b = {
        "neural_density": (
            density_neural_density,
            blue_colors["normal"],
            "Inhibitory density, $d_I$",
        ),
        "neural_gains": (
            density_neural_gains,
            green_colors["normal"],
            "Inhibitory gains, $g_I$",
        ),
    }

    plot_stimulus_curves(
        ax=ax_b,
        stimulus_locations=stimulus_locations,
        curves_dict=curves_b,
        individual_curves=density_neural_rates,
        individual_color=grey_colors["normal"],
        individual_alpha=0.7,
    )

    # Panel C: Input gain modulation - Excitatory (Input patterns)
    gain_input_patterns = gain_input_generator.stimuli_patterns[
        gain_batch_idx
    ]  # [N_E, num_stimuli]
    gain_input_density = gain_input_generator.input_density[
        gain_batch_idx
    ]  # [num_stimuli]
    gain_input_gains = gain_input_generator.input_gains[gain_batch_idx]  # [num_stimuli]

    gain_input_density, gain_input_gains, gain_input_patterns = normalise_curves(
        gain_input_density, gain_input_gains, gain_input_patterns
    )

    curves_c = {
        "input_density": (gain_input_density, blue_colors["normal"], "Input density"),
        "input_gains": (gain_input_gains, green_colors["normal"], "Input gains"),
    }

    plot_stimulus_curves(
        ax=ax_c,
        stimulus_locations=stimulus_locations,
        curves_dict=curves_c,
        individual_curves=gain_input_patterns / 10.0,
        individual_color=grey_colors["normal"],
        individual_alpha=0.5,
    )

    # Panel D: Input gain modulation - Inhibitory (Neural responses)
    gain_metrics = gain_results["metrics_over_time"]
    gain_neural_rates = gain_metrics["rates"][
        -1, 0, gain_batch_idx, :, :
    ]  # [N_I, num_stimuli]
    gain_neural_density = gain_metrics["curves/density"][
        -1, 0, gain_batch_idx, :
    ]  # [num_stimuli]
    gain_neural_gains = gain_metrics["curves/gains"][
        -1, 0, gain_batch_idx, :
    ]  # [num_stimuli]

    gain_neural_density, gain_neural_gains, gain_neural_rates = normalise_curves(
        gain_neural_density, gain_neural_gains, gain_neural_rates
    )

    curves_d = {
        "neural_density": (
            gain_neural_density,
            blue_colors["normal"],
            "Neural density",
        ),
        "neural_gains": (gain_neural_gains, green_colors["normal"], "Neural gains"),
    }

    plot_stimulus_curves(
        ax=ax_d,
        stimulus_locations=stimulus_locations,
        curves_dict=curves_d,
        individual_curves=gain_neural_rates,
        individual_color=grey_colors["normal"],
        individual_alpha=0.7,
    )

    # Style all panels
    panels = [ax_a, ax_b, ax_c, ax_d]
    panel_labels = ["A", "B", "C", "D"]

    for i, (ax, label) in enumerate(zip(panels, panel_labels)):
        # X-axis labels (only bottom panels)
        if i >= 2:  # Bottom panels (C, D)
            ax.set_xlabel(r"Input stimulus, $s$", fontsize=axis_size)
        else:  # Top panels (A, B) - remove x-axis labels
            ax.set_xticklabels([])
            ax.tick_params(bottom=False)

        ax.set_ylim([0, 2.5])
        ax.set_yticklabels([])

        # Set x-axis limits and ticks
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        if i >= 2:  # Only bottom panels get x-tick labels
            ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

        # Tick sizes
        ax.tick_params(axis="both", which="major", labelsize=ticks_size)

        # Grid
        if style["elements"]["grid"]:
            ax.grid(True, alpha=0.3)

        # Legend (panels A and B only)
        if i < 2 and style["elements"]["legend"]:
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

    ax_a.set_title("Excitatory Profile", fontsize=title_size)
    ax_b.set_title("Inhibitory Profile", fontsize=title_size)

    # Create figures directory and save
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    output_path = figures_dir / "input_modulation_comparison.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    plot_input_modulation_comparison()
