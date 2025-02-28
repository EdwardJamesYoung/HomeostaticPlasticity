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
        self.N_E = parameters.N_E
        self.tau_u = parameters.tau_u
        self.dt = parameters.dt
        self.device = parameters.device
        self.dtype = parameters.dtype

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def step(self) -> Float[torch.Tensor, "N_E 1"]:
        pass

    @abstractmethod
    def stimuli_batch(
        self, num_stimuli: int
    ) -> Tuple[
        Float[torch.Tensor, "N_E num_stimuli"], Float[torch.Tensor, "N_E num_stimuli"]
    ]:
        """
        This outputs first the stimuli, and then the mode contributions for each stimulus.
        """
        pass

    @abstractmethod
    def __dict__(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def mode_strengths(self) -> Float[torch.Tensor, "N_E"]:
        pass

    @abstractmethod
    def attunement_entropy(self, W: Float[torch.Tensor, "N_I N_E"]) -> float:
        pass


class PiecewiseConstantGenerator(InputGenerator, ABC):

    def __init__(self, parameters: SimulationParameters):
        super().__init__(parameters=parameters)
        self.steps_remaining = 0

    @jaxtyped(typechecker=typechecked)
    def step(self) -> Float[torch.Tensor, "N_E 1"]:
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
        self.latents = torch.randn((self.N_E, 1), device=self.device, dtype=self.dtype)

    @abstractmethod
    def latents_to_input(
        self, latents: Float[torch.Tensor, "N_E 1"]
    ) -> Float[torch.Tensor, "N_E 1"]:
        pass

    @jaxtyped(typechecker=typechecked)
    def step(self) -> Float[torch.Tensor, "N_E 1"]:
        dB = torch.randn((self.N_E, 1), device=self.device, dtype=self.dtype)
        self.latents = (
            1 - self.dt / self.tau_u
        ) * self.latents + self.noise_amplitude * dB
        return self.latents_to_input(self.latents)


class EigenbasisInputGenerator(ABC):

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
        input_eigenspectrum: Float[torch.Tensor, "N_E"],
    ):
        self.input_eigenbasis = input_eigenbasis.to(
            device=parameters.device, dtype=parameters.dtype
        )
        self.input_eigenspectrum = input_eigenspectrum.to(
            device=parameters.device, dtype=parameters.dtype
        )
        # Validate eigenbasis and eigenspectrum
        assert torch.allclose(
            input_eigenbasis @ input_eigenbasis.T,
            torch.eye(parameters.N_E, device=parameters.device, dtype=parameters.dtype),
        ), "Input eigenbasis is not orthogonal"
        assert torch.all(
            input_eigenspectrum >= 0
        ), "Input eigenspectrum has negative entries"

    @jaxtyped(typechecker=typechecked)
    def mode_strengths(self) -> Float[torch.Tensor, "N_E"]:
        return self.input_eigenspectrum

    @jaxtyped(typechecker=typechecked)
    def attunement_entropy(self, W: Float[torch.Tensor, "N_I N_E"]) -> float:
        return covariance_attunement_entropy(W, self.input_eigenbasis)


class GaussianDistributionGenerator(EigenbasisInputGenerator):

    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
        input_eigenspectrum: Float[torch.Tensor, "N_E"],
    ):
        super().__init__(
            parameters=parameters,
            input_eigenbasis=input_eigenbasis,
            input_eigenspectrum=input_eigenspectrum,
        )
        self.covariance_sqrt = self.input_eigenbasis @ torch.diag(
            torch.sqrt(self.input_eigenspectrum)
        )

    @jaxtyped(typechecker=typechecked)
    def stimuli_batch(self, num_stimuli: int) -> Tuple[
        Float[torch.Tensor, "N_E {num_stimuli}"],
        Float[torch.Tensor, "N_E {num_stimuli}"],
    ]:
        white_noise = torch.randn(
            (self.N_E, num_stimuli), device=self.device, dtype=self.dtype
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
        input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
        input_eigenspectrum: Float[torch.Tensor, "N_E"],
    ):
        super().__init__(
            parameters=parameters,
            input_eigenbasis=input_eigenbasis,
            input_eigenspectrum=input_eigenspectrum,
        )
        self.scales = torch.sqrt(self.input_eigenspectrum / 2).unsqueeze(1)

    @jaxtyped(typechecker=typechecked)
    def stimuli_batch(self, num_stimuli: int) -> Tuple[
        Float[torch.Tensor, "N_E {num_stimuli}"],
        Float[torch.Tensor, "N_E {num_stimuli}"],
    ]:
        white_noise = torch.randn(
            (self.N_E, num_stimuli), device=self.device, dtype=self.dtype
        )
        mode_stimuli_contributions = gaussian_to_laplace(white_noise, self.scales)
        stimuli = self.input_eigenbasis @ mode_stimuli_contributions
        return stimuli, mode_stimuli_contributions


class PiecewiseConstantGaussianGenerator(
    GaussianDistributionGenerator, PiecewiseConstantGenerator
):
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
        input_eigenspectrum: Float[torch.Tensor, "N_E"],
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
        input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
        input_eigenspectrum: Float[torch.Tensor, "N_E"],
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
        input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
        input_eigenspectrum: Float[torch.Tensor, "N_E"],
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
        self, latents: Float[torch.Tensor, "N_E 1"]
    ) -> Float[torch.Tensor, "N_E 1"]:
        return self.covariance_sqrt @ latents


class OULaplacianGenerator(LaplacianDistributionGenerator, OUProcessGenerator):
    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
        input_eigenspectrum: Float[torch.Tensor, "N_E"],
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
        self, latents: Float[torch.Tensor, "N_E 1"]
    ) -> Float[torch.Tensor, "N_E 1"]:
        x = gaussian_to_laplace(latents, self.scales)

        return self.input_eigenbasis @ x


class CircularGenerator(PiecewiseConstantGenerator):

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
        self.positions = torch.linspace(
            -torch.pi, torch.pi, self.N_E, device=self.device
        ).view(
            -1, 1
        )  # [N_E, 1]

        self.input_scale = compute_input_magnitude(parameters)

        # Compute mode strengths as the density of the uniform von-mises mixture
        self.sampling_probabilities = self.mixing_parameter * torch.exp(
            torch.distributions.VonMises(
                loc=torch.zeros(1, device=self.device),
                concentration=self.vm_concentration,
            ).log_prob(self.positions)
        ) + (1 - self.mixing_parameter) / (2 * torch.pi)
        # Renormlise mode strenghts to sum to one
        self.sampling_probabilities /= self.sampling_probabilities.sum()

        self.current_input, _ = self.stimuli_batch(1)
        self.steps_remaining = int(self.tau_u / self.dt)

    def __dict__(self) -> Dict[str, Any]:
        return {
            "tau_u": self.tau_u,
            "mixing_parameter": self.mixing_parameter,
            "vm_concentration": self.vm_concentration,
            "tuning_width": self.tuning_width,
        }

    @jaxtyped(typechecker=typechecked)
    def stimuli_batch(
        self, num_stimuli: int
    ) -> Tuple[
        Float[torch.Tensor, "N_E num_stimuli"], Float[torch.Tensor, "N_E num_stimuli"]
    ]:
        mode_indices = torch.multinomial(
            self.sampling_probabilities.squeeze(),
            num_samples=num_stimuli,
            replacement=True,
        )  # [num_stimuli]

        # Compute the mode stimuli contributions as one-hot encodings of the mode_indices
        mode_stimuli_contributions = torch.zeros(
            self.N_E, num_stimuli, device=self.device, dtype=self.dtype
        )  # [N_E, num_stimuli]
        mode_stimuli_contributions[mode_indices, torch.arange(num_stimuli)] = 1
        orientations = self.positions[mode_indices].view(1, -1)  # [1, num_stimuli]

        circ_distances = torch.abs(self.positions - orientations)  # [N_E, num_stimuli]
        min_distances = torch.minimum(circ_distances, 2 * torch.pi - circ_distances)

        input_activities = (
            self.input_scale
            * torch.exp(-(min_distances**2) / (2 * self.tuning_width**2)).to(
                device=self.device, dtype=self.dtype
            )
            / (self.N_E * self.tuning_width)
        )

        return input_activities, mode_stimuli_contributions

    @jaxtyped(typechecker=typechecked)
    def sample_orientations(
        self, num_stimuli: int = 1
    ) -> Float[torch.Tensor, "1 num_stimuli"]:
        # Sample from the mode_strengths distribution
        mode_indices = torch.multinomial(
            self.sampling_probabilities, num_samples=num_stimuli, replacement=True
        )
        orientations = self.positions[mode_indices]

        return orientations.view(1, -1)

    @jaxtyped(typechecker=typechecked)
    def mode_strengths(self):
        return self.sampling_probabilities

    @jaxtyped(typechecker=typechecked)
    def attunement_entropy(self, W: Float[torch.Tensor, "N_I N_E"]) -> float:
        return 0.0


def generate_conditions(
    parameters: SimulationParameters,
) -> Tuple[Float[torch.Tensor, "N_E N_E"], Float[torch.Tensor, "N_E"]]:
    N_E = parameters.N_E
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

    input_eigenspectrum = (
        2 * torch.sort(torch.rand(N_E, device=device, dtype=dtype), descending=True)[0]
    )
    input_eigenspectrum = spectrum_multiplier * input_eigenspectrum

    return input_eigenbasis, input_eigenspectrum


def compute_input_magnitude(parameters: SimulationParameters):
    N_E = parameters.N_E
    N_I = parameters.N_I
    omega = parameters.omega

    if parameters.target_rate is not None:
        target = parameters.target_rate
    elif parameters.target_variance is not None:
        target = parameters.target_variance
    else:
        raise ValueError("Must specify either target rate or target variance")

    return (target * N_I) / (omega * N_E)


def covariance_attunement_entropy(
    W: Float[torch.Tensor, "N_I N_E"], input_eigenbasis: Float[torch.Tensor, "N_E N_E"]
) -> float:
    W_projected = W @ input_eigenbasis
    W_projected_abs = torch.abs(W_projected)
    W_projected_norm = W_projected_abs / (
        torch.sum(W_projected_abs, dim=1, keepdim=True) + 1e-16
    )
    W_entropy = -torch.sum(W_projected_norm * torch.log(W_projected_norm), dim=1)
    return torch.mean(W_entropy).item()


def gaussian_to_laplace(
    z: Float[torch.Tensor, "N_E num_stimuli"],
    scales: Float[torch.Tensor, "N_E 1"],
) -> Float[torch.Tensor, "N_E num_stimuli"]:
    uniforms = 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))
    signs = torch.sign(2 * uniforms - 1.0)
    abs_diff = torch.abs(2 * uniforms - 1.0)
    x = signs * scales * torch.log(abs_diff + EPSILON)

    return x
