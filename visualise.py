import torch
import numpy as np
from typing import Any
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from input_generation import CircularGenerator
from simulator import SimulationParameters

import matplotlib.pyplot as plt


def visualise_weights(
    W: Float[torch.Tensor, "N_I N_E"], M: Float[torch.Tensor, "N_I N_I"]
):
    r"""Takes a feedforward weight matrix and visualises the weights as curves over the input space."""

    # Create a new figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].set_title("Feedforward weights")
    axs[0].set_xlabel("Input space")
    axs[0].set_ylabel("Weight value")

    # Loop over inhibitory neurons N_I
    for i_idx in range(W.shape[0]):
        # Plot the weights of the i_idx-th inhibitory neuron
        axs[0].plot(W[i_idx, :].detach().cpu().numpy())

    # Find the argmax stimulus for each neuron
    argmax_stimuli = torch.argmax(W, dim=1).detach().cpu().numpy()  # [N_I]

    # Sort neurons by their argmax stimuli
    sorted_indices = np.argsort(argmax_stimuli)

    # Reshuffle M according to the sorted ordering
    M_sorted = M[sorted_indices, :][:, sorted_indices].detach().cpu().numpy()

    # Plot the recurrent weights
    axs[1].set_title("Recurrent weights (sorted by preferred input)")
    axs[1].set_xlabel("Inhibitory neuron index (sorted)")
    axs[1].set_ylabel("Inhibitory neuron index (sorted)")
    im = axs[1].imshow(M_sorted, cmap="viridis")
    fig.colorbar(im, ax=axs[1])

    plt.show()


@jaxtyped(typechecker=typechecked)
def visualise_tuning_curves(
    population_response_metrics: dict[str, Any],
    parameters: SimulationParameters,
):
    r"""Takes a feedforward weight matrix and a recurrent weight matrix and visualises the tuning curves of the neurons."""

    # Extract metrics from the population_metrics dictionary
    stimulus_space = population_response_metrics["stimulus_space"]
    stimuli_probabilities = population_response_metrics["stimuli_probabilities"]
    normalised_total_rate = population_response_metrics["normalised_total_rate"]
    normalised_average_input_activation = population_response_metrics[
        "normalised_average_input_activation"
    ]
    rates = population_response_metrics["rates"]
    argmax_stimuli = population_response_metrics["argmax_stimuli"]
    normalised_max_rates = population_response_metrics["normalised_max_rates"]
    smoothed_max_rates = population_response_metrics["smoothed_max_rates"]
    preferred_stimulus_density = population_response_metrics[
        "preferred_stimulus_density"
    ]
    density_times_gain = population_response_metrics["density_times_gain"]
    average_rates = population_response_metrics["average_rates"]
    pattern_overlaps = population_response_metrics["pattern_overlaps"]
    generalised_density = population_response_metrics["generalised_density"]  # [N_E]
    generalised_gain = population_response_metrics["generalised_gain"]  # [N_E]

    # Create a two panel figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # In the first panel, show the stimuli probabilities as a function of the stimulus
    axs[0].plot(stimulus_space, stimuli_probabilities, label="Stimuli probabilities")
    axs[0].plot(stimulus_space, normalised_total_rate, label="(Normalised) total rate")
    axs[0].plot(
        np.linspace(
            stimulus_space.min(),
            stimulus_space.max(),
            parameters.N_E,
        ),
        normalised_average_input_activation,
        label="(Normalised) average input activation",
    )

    # Plot the squared stimulus probabilities
    axs[0].plot(
        stimulus_space,
        stimuli_probabilities**2 / (stimuli_probabilities**2).sum(),
        label="Squared stimuli probabilities",
        color="orange",
    )

    mean_pattern_overlap = pattern_overlaps.mean(axis=0)
    normalised_mean_pattern_overlap = (
        (parameters.N_E / parameters.num_latents)
        * mean_pattern_overlap
        / mean_pattern_overlap.sum()
    )
    # normalised_generalised_density = (
    #     (parameters.N_E / parameters.num_latents)
    #     * generalised_density
    #     / generalised_density.sum()
    # )
    # normalised_generalised_gain = (
    #     (parameters.N_E / parameters.num_latents)
    #     * generalised_gain
    #     / generalised_gain.sum()
    # )
    normalised_generalised_density = generalised_density / generalised_density.sum()
    normalised_generalised_gain = generalised_gain / generalised_gain.sum()

    axs[0].plot(
        np.linspace(
            stimulus_space.min(),
            stimulus_space.max(),
            parameters.N_E,
        ),
        normalised_mean_pattern_overlap,
        label="(Normalised) mean pattern overlap",
        color="black",
    )

    # axs[0].plot(
    #     # np.linspace(
    #     #     stimulus_space.min(),
    #     #     stimulus_space.max(),
    #     #     parameters.N_E,
    #     # ),
    #     stimulus_space,
    #     normalised_generalised_density,
    #     label="(Generalised) density",
    #     color="cyan",
    # )
    # axs[0].plot(
    #     # np.linspace(
    #     #     stimulus_space.min(),
    #     #     stimulus_space.max(),
    #     #     parameters.N_E,
    #     # ),
    #     stimulus_space,
    #     normalised_generalised_gain,
    #     label="(Generalised) gain",
    #     color="magenta",
    # )

    axs[0].scatter(
        argmax_stimuli,
        normalised_max_rates,
        color="purple",
        zorder=3,
        label="(Normalised) max response",
    )
    axs[0].plot(
        stimulus_space,
        smoothed_max_rates,
        color="purple",
        alpha=0.7,
        zorder=2,
        label="(Normalised) smoothed max response curve",
    )

    axs[0].plot(
        stimulus_space,
        preferred_stimulus_density,
        color="red",
        label="Density of preferred orientations",
    )

    # axs[0].plot(stimulus_space, density_times_gain, color="orange", label="new thing")

    axs[0].legend()
    axs[0].set_xlabel("Stimulus")
    axs[0].set_xlim(stimulus_space.min(), stimulus_space.max())
    axs[0].set_title("Input distribution and network response")

    # In the second panel, show the tuning curves of the neurons
    # Get the default color cycle from matplotlib
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for i_idx in range(rates.shape[0]):
        # Use the same color for both the line and its corresponding scatter point
        (line,) = axs[1].plot(
            stimulus_space, rates[i_idx, :], color=colors[i_idx % len(colors)]
        )
        axs[1].scatter(
            argmax_stimuli[i_idx],
            average_rates[i_idx],
            color=line.get_color(),
            zorder=3,
        )

    axs[1].set_title("Tuning curves")
    axs[1].set_xlabel("Stimulus")
    axs[1].set_ylabel("Firing rate")
    axs[1].set_ylim(min(rates.min(), 0), 1.1 * rates.max())
    axs[1].set_xlim(stimulus_space.min(), stimulus_space.max())

    # Show the plot
    plt.show()


def visualise_input_patterns(input_generator: CircularGenerator):
    r"""Takes an input generator and visualises the input patterns."""

    # Get the stimuli patterns
    stimuli_patterns = (
        input_generator.stimuli_patterns.detach().cpu().numpy()
    )  # [N_E, num_latents]
    neuron_positions = (
        input_generator.neuron_positions.detach().cpu().numpy()
    )  # [N_E, 1]

    # Create a figure
    fig = plt.figure()
    plt.title("Input patterns")

    # Loop over the stimuli patterns
    for i_idx in range(stimuli_patterns.shape[1]):
        # Plot the i_idx-th stimulus pattern
        plt.plot(neuron_positions, stimuli_patterns[:, i_idx])

    # Set the axis limits
    plt.ylim(0, 1.1 * stimuli_patterns.max())
    plt.xlim(neuron_positions.min(), neuron_positions.max())

    # Show the plot
    plt.show()


def visualise_pattern_overlaps(
    population_response_metrics: dict[str, Any],
    input_generator: CircularGenerator,
):
    pattern_overlaps = population_response_metrics["pattern_overlaps"]  # [N_I, N_E]
    neuron_positions = (
        input_generator.neuron_positions.detach().cpu().numpy()
    )  # [N_E, 1]

    # Create a figure
    fig = plt.figure()
    plt.title("Pattern overlaps")

    # Loop over the inhibitory neurons
    for i_idx in range(pattern_overlaps.shape[0]):
        # Plot the i_idx-th pattern overlap
        plt.plot(neuron_positions, pattern_overlaps[i_idx, :])

    # Plot the average pattern overlap in bold
    plt.plot(
        neuron_positions,
        pattern_overlaps.mean(axis=0),
        color="black",
        linewidth=2,
    )

    # Set the axis limits
    plt.ylim(0, 1.1 * pattern_overlaps.max())
    plt.xlim(neuron_positions.min(), neuron_positions.max())

    # Show the plot
    plt.show()
