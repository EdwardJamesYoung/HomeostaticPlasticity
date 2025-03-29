import torch
import scipy.stats
import numpy as np
from abc import ABC, abstractmethod
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Dict, Any, Tuple
from params import SimulationParameters


EPSILON = 1e-12


class InputGenerator(ABC):

    def __init__(self, parameters: SimulationParameters):
        self.parameters = parameters
        self.N_E = parameters.N_E
        self.num_latents = parameters.num_latents
        self.tau_u = parameters.tau_u
        self.dt = parameters.dt
        self.device = parameters.device
        self.dtype = parameters.dtype

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def step(self) -> Float[torch.Tensor, "{self.N_E} 1"]:
        pass

    @abstractmethod
    def stimuli_batch(self, num_stimuli: int) -> Tuple[
        Float[torch.Tensor, "{self.N_E} num_stimuli"],
        Float[torch.Tensor, "{self.num_latents} num_stimuli"],
    ]:
        """
        This outputs first the stimuli, and then the mode contributions for each stimulus.
        """
        pass

    @abstractmethod
    def __dict__(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def attunement_entropy(self, W: Float[torch.Tensor, "N_I {self.N_E}"]) -> float:
        pass


class PiecewiseConstantGenerator(InputGenerator, ABC):

    def __init__(self, parameters: SimulationParameters):
        super().__init__(parameters=parameters)
        self.steps_remaining = 0

    @jaxtyped(typechecker=typechecked)
    def step(self) -> Float[torch.Tensor, "{self.N_E} 1"]:
        if self.steps_remaining == 0:
            self.current_input, _ = self.stimuli_batch(1)
            self.steps_remaining = int(self.tau_u / self.dt)
        else:
            self.steps_remaining -= 1

        return self.current_input


class OUProcessGenerator(InputGenerator, ABC):

    def __init__(
        self,
        parameters: SimulationParameters,
    ):
        super().__init__(parameters=parameters)

        self.noise_amplitude = torch.sqrt(
            torch.tensor(
                2.0 * self.dt / self.tau_u, device=self.device, dtype=self.dtype
            )
        )
        self.latents = torch.randn(
            (self.num_latents, 1), device=self.device, dtype=self.dtype
        )

    @abstractmethod
    def latents_to_input(
        self, latents: Float[torch.Tensor, "{self.num_latents} 1"]
    ) -> Float[torch.Tensor, "{self.N_E} 1"]:
        pass

    @jaxtyped(typechecker=typechecked)
    def step(self) -> Float[torch.Tensor, "{self.N_E} 1"]:
        dB = torch.randn((self.num_latents, 1), device=self.device, dtype=self.dtype)
        self.latents = (
            1 - self.dt / self.tau_u
        ) * self.latents + self.noise_amplitude * dB
        return self.latents_to_input(self.latents)


class EigenbasisInputGenerator(ABC):

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "{self.N_E} {self.num_latents}"],
        input_eigenspectrum: Float[torch.Tensor, "{self.num_latents}"],
    ):
        self.input_eigenbasis = input_eigenbasis.to(
            device=parameters.device, dtype=parameters.dtype
        )
        self.input_eigenspectrum = input_eigenspectrum.to(
            device=parameters.device, dtype=parameters.dtype
        )
        # Validate eigenbasis and eigenspectrum
        assert torch.allclose(
            input_eigenbasis.T @ input_eigenbasis,
            torch.eye(
                self.num_latents, device=parameters.device, dtype=parameters.dtype
            ),
            atol=1e-6,
        ), "Each column of the input eigenbasis must be orthogonal"
        assert torch.all(
            input_eigenspectrum >= 0
        ), "Input eigenspectrum has negative entries"

    @jaxtyped(typechecker=typechecked)
    def mode_strengths(self) -> Float[torch.Tensor, "{self.num_latents}"]:
        return self.input_eigenspectrum

    @jaxtyped(typechecker=typechecked)
    def attunement_entropy(
        self, W: Float[torch.Tensor, "N_I {self.num_latents}"]
    ) -> float:
        W_projected = W @ self.input_eigenbasis  # [N_I, num_latents]
        W_projected_abs = torch.abs(W_projected)
        W_projected_norm = W_projected_abs / (
            torch.sum(W_projected_abs, dim=1, keepdim=True) + 1e-16
        )
        W_entropy = -torch.sum(
            W_projected_norm * torch.log(W_projected_norm), dim=1
        )  # [N_I]
        return torch.mean(W_entropy).item()


class GaussianDistributionGenerator(EigenbasisInputGenerator):

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "{self.N_E} {self.num_latents}"],
        input_eigenspectrum: Float[torch.Tensor, "{self.num_latents}"],
    ):
        super().__init__(
            parameters=parameters,
            input_eigenbasis=input_eigenbasis,
            input_eigenspectrum=input_eigenspectrum,
        )
        self.covariance_sqrt = self.input_eigenbasis @ torch.diag(
            torch.sqrt(self.input_eigenspectrum)
        )  # [N_E, num_latents]

    @jaxtyped(typechecker=typechecked)
    def stimuli_batch(self, num_stimuli: int) -> Tuple[
        Float[torch.Tensor, "{self.N_E} {num_stimuli}"],
        Float[torch.Tensor, "{self.num_latents} {num_stimuli}"],
    ]:
        white_noise = torch.randn(
            (self.num_latents, num_stimuli), device=self.device, dtype=self.dtype
        )
        # For each stimulus calculate the mode contributions
        mode_stimuli_contributions = (
            torch.diag(torch.sqrt(self.input_eigenspectrum)) @ white_noise
        )  # [N_E, num_stimuli]
        stimuli = self.input_eigenbasis @ mode_stimuli_contributions
        return stimuli, mode_stimuli_contributions


class LaplacianDistributionGenerator(EigenbasisInputGenerator):

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "{self.N_E} {self.num_latents}"],
        input_eigenspectrum: Float[torch.Tensor, "{self.num_latents}"],
    ):
        super().__init__(
            parameters=parameters,
            input_eigenbasis=input_eigenbasis,
            input_eigenspectrum=input_eigenspectrum,
        )
        self.scales = torch.sqrt(self.input_eigenspectrum / 2).unsqueeze(
            1
        )  # [num_latents, 1]

    @jaxtyped(typechecker=typechecked)
    def stimuli_batch(self, num_stimuli: int) -> Tuple[
        Float[torch.Tensor, "{self.N_E} num_stimuli"],
        Float[torch.Tensor, "{self.num_latents} num_stimuli"],
    ]:
        white_noise = torch.randn(
            (self.num_latents, num_stimuli), device=self.device, dtype=self.dtype
        )
        mode_stimuli_contributions = gaussian_to_laplace(white_noise)
        stimuli = self.input_eigenbasis @ mode_stimuli_contributions
        return stimuli, mode_stimuli_contributions

    def gaussian_to_laplace(
        z: Float[torch.Tensor, "{self.num_latents} num_stimuli"],
    ) -> Float[torch.Tensor, "{self.num_latents} num_stimuli"]:
        uniforms = 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))
        signs = torch.sign(2 * uniforms - 1.0)
        abs_diff = torch.abs(2 * uniforms - 1.0)
        x = signs * self.scales * torch.log(abs_diff + EPSILON)

        return x


class PiecewiseConstantGaussianGenerator(
    GaussianDistributionGenerator, PiecewiseConstantGenerator
):
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "{self.N_E} {self.num_latents}"],
        input_eigenspectrum: Float[torch.Tensor, "{self.num_latents}"],
    ):
        PiecewiseConstantGenerator.__init__(self, parameters=parameters)
        GaussianDistributionGenerator.__init__(
            self,
            parameters=parameters,
            input_eigenbasis=input_eigenbasis,
            input_eigenspectrum=input_eigenspectrum,
        )

    def __dict__(self) -> Dict[str, Any]:
        return {"tau_u": self.tau_u}


class PiecewiseConstantLaplacianGenerator(
    LaplacianDistributionGenerator, PiecewiseConstantGenerator
):
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "{self.N_E} {self.num_latents}"],
        input_eigenspectrum: Float[torch.Tensor, "{self.num_latents}"],
    ):
        PiecewiseConstantGenerator.__init__(self, parameters=parameters)
        LaplacianDistributionGenerator.__init__(
            self,
            parameters=parameters,
            input_eigenbasis=input_eigenbasis,
            input_eigenspectrum=input_eigenspectrum,
        )

    def __dict__(self) -> Dict[str, Any]:
        return {"tau_u": self.tau_u}


class OUGaussianGenerator(GaussianDistributionGenerator, OUProcessGenerator):
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "{self.N_E} {self.num_latents}"],
        input_eigenspectrum: Float[torch.Tensor, "{self.num_latents}"],
    ):
        OUProcessGenerator.__init__(self, parameters=parameters)
        GaussianDistributionGenerator.__init__(
            self,
            parameters=parameters,
            input_eigenbasis=input_eigenbasis,
            input_eigenspectrum=input_eigenspectrum,
        )

    def __dict__(self) -> Dict[str, Any]:
        return {"tau_u": self.tau_u}

    @jaxtyped(typechecker=typechecked)
    def latents_to_input(
        self, latents: Float[torch.Tensor, "{self.num_latents} 1"]
    ) -> Float[torch.Tensor, "{self.N_E} 1"]:
        return self.covariance_sqrt @ latents


class OULaplacianGenerator(LaplacianDistributionGenerator, OUProcessGenerator):
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "{self.N_E} {self.num_latents}"],
        input_eigenspectrum: Float[torch.Tensor, "{self.num_latents}"],
    ):
        OUProcessGenerator.__init__(self, parameters=parameters)
        LaplacianDistributionGenerator.__init__(
            self,
            parameters=parameters,
            input_eigenbasis=input_eigenbasis,
            input_eigenspectrum=input_eigenspectrum,
        )

    def __dict__(self) -> Dict[str, Any]:
        return {"tau_u": self.tau_u}

    @jaxtyped(typechecker=typechecked)
    def latents_to_input(
        self, latents: Float[torch.Tensor, "{self.num_latents} 1"]
    ) -> Float[torch.Tensor, "{self.N_E} 1"]:
        x = gaussian_to_laplace(latents)

        return self.input_eigenbasis @ x


class DiscreteGenerator(ABC):
    """
    Mixin for input generators with a discrete set of possible stimuli.
    This allows for a fixed number of distinct input patterns (N_stimuli) that
    may differ from the number of input neurons (N_E).
    """

    @property
    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def stimuli_patterns(self) -> Float[torch.Tensor, "{self.N_E} {self.num_latents}"]:
        """
        Returns all possible input stimuli patterns.

        Returns:
            torch.Tensor: Tensor of shape [N_E, n_stimuli]
        """
        pass

    @property
    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def stimuli_probabilities(self) -> Float[torch.Tensor, "{self.num_latents}"]:
        """
        Returns the probability of sampling each stimulus.

        Returns:
            torch.Tensor: Tensor of shape [n_stimuli] containing probabilities that sum to 1
        """
        pass

    @jaxtyped(typechecker=typechecked)
    def stimuli_batch(self, num_stimuli: int) -> Tuple[
        Float[torch.Tensor, "{self.N_E} num_stimuli"],
        Float[torch.Tensor, "{self.num_latents} num_stimuli"],
    ]:
        """
        Sample stimuli based on sampling probabilities.

        Args:
            num_stimuli: Number of stimuli to sample

        """
        # Sample stimulus indices according to their probabilities
        stimulus_indices = torch.multinomial(
            self.stimuli_probabilities, num_samples=num_stimuli, replacement=True
        )  # [num_stimuli]

        # Get the selected stimuli patterns
        selected_patterns = self.stimuli_patterns[
            :, stimulus_indices
        ]  # [N_E, num_stimuli]

        # Create mode contributions (one-hot encoding by default)
        mode_stimuli_contributions = torch.zeros(
            self.num_latents, num_stimuli, device=self.device, dtype=self.dtype
        )
        mode_stimuli_contributions[stimulus_indices, torch.arange(num_stimuli)] = (
            1  # [num_latents, num_stimuli]
        )

        return selected_patterns, mode_stimuli_contributions


class CircularGenerator(DiscreteGenerator, PiecewiseConstantGenerator):

    def __init__(
        self,
        parameters: SimulationParameters,
        mixing_parameter: float,
        vm_concentration: float,
        tuning_width: float,
    ):
        super().__init__(parameters=parameters)
        assert (
            0 <= mixing_parameter <= 1
        ), f"Mixing parameter must be between 0 and 1. Got {mixing_parameter}."
        assert (
            0 < vm_concentration
        ), f"Von Mises concentration must be positive. Got {vm_concentration}."
        assert 0 < tuning_width, f"Tuning width must be positive. Got {tuning_width}."
        self.mixing_parameter = mixing_parameter
        self.vm_concentration = vm_concentration
        self.tuning_width = tuning_width
        self.neuron_positions = torch.linspace(
            -torch.pi, torch.pi, self.N_E + 1, device=self.device
        )[: self.N_E].view(
            -1, 1
        )  # [N_E, 1]
        self.stimuli_positions = torch.linspace(
            -torch.pi, torch.pi, self.num_latents + 1, device=self.device
        )[: self.num_latents].view(
            -1, 1
        )  # [num_latents, 1]

        # Normalize probabilities to sum to 1
        self._stimuli_probabilities = self._compute_stimuli_probabilities()

        # Pre-compute all stimulus patterns
        self._stimuli_patterns = self._compute_stimuli_patterns()

    def __dict__(self) -> Dict[str, Any]:
        return {
            "tau_u": self.tau_u,
            "mixing_parameter": self.mixing_parameter,
            "vm_concentration": self.vm_concentration,
            "tuning_width": self.tuning_width,
        }

    def _compute_stimuli_probabilities(
        self,
    ) -> Float[torch.Tensor, "{self.num_latents}"]:
        """
        Compute the probabilities of sampling each stimulus.

        Returns:
            torch.Tensor: Tensor of shape [n_stimuli] containing probabilities that sum to 1
        """
        # Compute mode strengths as the density of the uniform von-mises mixture
        stimuli_probabilities = self.mixing_parameter * torch.exp(
            torch.distributions.VonMises(
                loc=torch.zeros(1, device=self.device),
                concentration=self.vm_concentration,
            ).log_prob(self.stimuli_positions)
        ) + (1 - self.mixing_parameter) / (2 * torch.pi)
        # Renormlise mode strenghts to sum to one
        stimuli_probabilities /= stimuli_probabilities.sum()

        return stimuli_probabilities.squeeze()  # [num_latents]

    @jaxtyped(typechecker=typechecked)
    def _compute_stimuli_patterns(
        self,
    ) -> Float[torch.Tensor, "{self.N_E} {self.num_latents}"]:
        """
        Compute all possible stimulus patterns.

        Returns:

        """
        # Calculate circular distances between all stimulus positions and neuron positions
        circ_distances = torch.abs(self.stimuli_positions.T - self.neuron_positions)
        min_distances = torch.minimum(circ_distances, 2 * torch.pi - circ_distances)  #

        # Compute tuning curve responses
        # stimuli_patterns = (
        #     compute_input_magnitude(self.parameters)
        #     * torch.exp(-(min_distances**2) / (2 * self.tuning_width**2))
        #     / (self.tuning_width)
        # ).to(device=self.device, dtype=self.dtype)

        stimuli_patterns = torch.exp(
            -(min_distances**2) / (2 * self.tuning_width**2)
        ).to(device=self.device, dtype=self.dtype)

        return stimuli_patterns

    @property
    @jaxtyped(typechecker=typechecked)
    def stimuli_patterns(self) -> Float[torch.Tensor, "{self.N_E} {self.num_latents}"]:
        """
        Returns all possible input stimuli patterns.

        Returns:
            torch.Tensor: Tensor of shape [n_stimuli, N_E] containing all possible stimuli
        """
        return self._stimuli_patterns

    @property
    @jaxtyped(typechecker=typechecked)
    def stimuli_probabilities(self) -> Float[torch.Tensor, "{self.num_latents}"]:
        """
        Returns the probability of sampling each stimulus.

        Returns:
            torch.Tensor: Tensor of shape [n_stimuli] containing probabilities that sum to 1
        """
        return self._stimuli_probabilities

    @jaxtyped(typechecker=typechecked)
    def attunement_entropy(self, W: Float[torch.Tensor, "N_I {self.N_E}"]) -> float:
        return 0.0


def generate_conditions(
    parameters: SimulationParameters,
) -> Tuple[
    Float[torch.Tensor, "{parameters.N_E} {parameters.num_latents}"],
    Float[torch.Tensor, "{parameters.num_latents}"],
]:
    N_E = parameters.N_E
    num_latents = parameters.num_latents

    assert N_E >= num_latents, "Number of latents must be less than or equal to N_E"

    device = parameters.device
    dtype = parameters.dtype
    random_seed = parameters.random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    spectrum_multiplier = compute_input_magnitude(parameters)

    # For orthogonal matrix generation, we'll still use numpy and then convert
    input_eigenbasis = torch.tensor(
        scipy.stats.ortho_group.rvs(N_E), device=device, dtype=dtype
    )
    input_eigenbasis = input_eigenbasis[:, :num_latents]

    input_eigenspectrum = (
        2
        * torch.sort(
            torch.rand(num_latents, device=device, dtype=dtype), descending=True
        )[0]
    )
    input_eigenspectrum = spectrum_multiplier * input_eigenspectrum

    return input_eigenbasis, input_eigenspectrum


def compute_input_magnitude(parameters: SimulationParameters):
    num_latents = parameters.num_latents
    N_I = parameters.N_I
    omega = parameters.omega

    if parameters.target_rate is not None:
        target = parameters.target_rate
    elif parameters.target_variance is not None:
        target = parameters.target_variance
    else:
        raise ValueError("Must specify either target rate or target variance")

    return (target * N_I) / (omega * num_latents)
