import torch
import numpy as np
from typing import Any
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from input_generation import CircularGenerator
from simulator import SimulationParameters

import matplotlib.pyplot as plt


def visualise_feedforward_weights(W: Float[torch.Tensor, "N_I N_E"]):
    r"""Takes a feedforward weight matrix and visualises the weights as curves over the input space."""

    # Create a new figure
    fig = plt.figure()
    plt.title("Feedforward weights")
    plt.xlabel("Input space")
    plt.ylabel("Weight value")

    # Loop over inhibitory neurons N_I
    for i_idx in range(W.shape[0]):
        # Plot the weights of the i_idx-th inhibitory neuron
        plt.plot(W[i_idx, :].detach().cpu().numpy())

    # Show the plot
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

    axs[0].plot(stimulus_space, density_times_gain, color="orange", label="new thing")

    axs[0].legend()
    axs[0].set_xlabel("Stimulus")
    axs[0].set_xlim(stimulus_space.min(), stimulus_space.max())
    axs[0].set_title("Input distribution and network response")

    # In the second panel, show the tuning curves of the neurons
    for i_idx in range(rates.shape[0]):
        axs[1].plot(stimulus_space, rates[i_idx, :])
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
