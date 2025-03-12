import torch
import numpy as np
from typing import Tuple, Any
import wandb
from scipy import stats
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from input_generation import CircularGenerator
from simulator import compute_firing_rates
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


def visualise_tuning_curves(
    W: Float[torch.Tensor, "N_I N_E"],
    M: Float[torch.Tensor, "N_I N_I"],
    input_generator: CircularGenerator,
    parameters: SimulationParameters,
):
    r"""Takes a feedforward weight matrix and a recurrent weight matrix and visualises the tuning curves of the neurons."""

    N_E = parameters.N_E
    num_latents = parameters.num_latents
    N_I = parameters.N_I

    stimuli_probabilities = input_generator.stimuli_probabilities  # [num_latents]
    stimuli_patterns = input_generator.stimuli_patterns  # [N_E, num_latents]
    average_input_activation = stimuli_patterns @ stimuli_probabilities.to(
        dtype=stimuli_patterns.dtype
    )  # [N_E]
    stimuli_probabilities = stimuli_probabilities.detach().cpu().numpy()
    normalised_average_input_activation = (N_E / num_latents) * (
        average_input_activation / average_input_activation.sum()
    ).detach().cpu().numpy()
    rates, _ = compute_firing_rates(
        W,
        M,
        stimuli_patterns,
        parameters=parameters,
        threshold=1e-8,
    )  # [N_I, num_latents]
    rates = rates.detach().cpu().numpy()  # [N_I, num_latents]
    argmax_rates = rates.argmax(axis=1)  # [N_I] (indices of max response)
    stimulus_space = input_generator.stimuli_positions.squeeze()  # [num_latents]
    argmax_stimuli = stimulus_space[
        argmax_rates
    ].flatten()  # [N_I] (stimuli of max response)
    max_rates = rates.max(axis=1)  # [N_I]

    inverse_max_rates = 1 / max_rates
    normalised_max_rates = (N_I / num_latents) * max_rates / max_rates.sum()  # [N_I]
    smoothed_max_rates = circular_smooth_values(
        argmax_stimuli, normalised_max_rates, stimulus_space, bw=0.25
    )  # [num_latents]

    total_rate = rates.sum(axis=0)  # [num_latents]
    normalised_total_rate = total_rate / total_rate.sum()  # [num_latents]

    argmax_stimuli = (
        argmax_stimuli.detach().cpu().numpy()
    )  # [N_I] (stimuli of max response)

    stimulus_space = (
        input_generator.stimuli_positions.detach().cpu().numpy()
    )  # [num_latents, 1]

    # Create a two panel figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    # In the first panel, show the stimuli probabilities as a function of the stimulus
    axs[0].plot(stimulus_space, stimuli_probabilities, label="Stimuli probabilities")
    axs[0].plot(stimulus_space, normalised_total_rate, label="(Normalised) total rate")
    axs[0].plot(
        np.linspace(
            stimulus_space.min(),
            stimulus_space.max(),
            N_E,
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

    bw = 0.25 * (N_I ** (-0.2))

    kde = circular_kde(argmax_stimuli, stimulus_space, bw=bw)
    kde = kde / kde.sum()
    axs[0].plot(
        stimulus_space, kde, color="red", label="Density of preferred orientations"
    )

    axs[0].legend()
    axs[0].set_xlabel("Stimulus")
    # axs[0].set_ylim(0, 1.1 * stimuli_probabilities.max())
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


def circular_kde(argmax_rates, stimulus_space, bw=0.25):
    extended_data = np.concatenate(
        [
            argmax_rates - 2 * np.pi,
            argmax_rates,
            argmax_rates + 2 * np.pi,
        ]
    )

    # Compute KDE on extended data
    kde = stats.gaussian_kde(extended_data, bw_method=bw)
    y = kde(stimulus_space.squeeze())

    return y


def circular_smooth_values(
    argmax_stimuli: Float[torch.Tensor, "N_I"],
    max_rates: Float[torch.Tensor, "N_I"],
    stimulus_space: Float[torch.Tensor, "num_latents"],
    bw=0.25,
) -> Float[torch.Tensor, "num_latents"]:

    diff = torch.abs(
        stimulus_space.unsqueeze(0) - argmax_stimuli.unsqueeze(1)
    )  # [N_I, num_latents]
    diff = diff.cpu().numpy()
    diff = np.minimum(diff, 2 * np.pi - diff)  # [N_I, num_latents]
    gaussians = np.exp(-0.5 * (diff / bw) ** 2) / (
        bw * np.sqrt(2 * np.pi)
    )  # [N_I, num_latents]

    weights = gaussians / gaussians.sum(axis=0)  # [N_I, num_latents]

    y = max_rates @ weights  # [num_latents]

    y = (y / y.sum()).squeeze()

    return y


def visualise_input_patterns(input_generator: CircularGenerator):
    r"""Takes an input generator and visualises the input patterns."""

    # Get the stimuli patterns
    stimuli_patterns = (
        input_generator.stimuli_patterns.detach().cpu().numpy()
    )  # [N_E, num_latents]
    stimulus_space = (
        input_generator.stimuli_positions.detach().cpu().numpy()
    )  # [num_latents, 1]

    # Create a figure
    fig = plt.figure()
    plt.title("Input patterns")

    # Loop over the stimuli patterns
    for i_idx in range(stimuli_patterns.shape[0]):
        # Plot the i_idx-th stimulus pattern
        plt.plot(stimulus_space, stimuli_patterns[i_idx, :])

    # Set the axis limits
    plt.ylim(0, 1.1 * stimuli_patterns.max())
    plt.xlim(stimulus_space.min(), stimulus_space.max())

    # Show the plot
    plt.show()
