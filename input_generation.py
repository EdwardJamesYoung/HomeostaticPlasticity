import torch
import scipy.stats
import numpy as np
from abc import ABC, abstractmethod
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Dict, Any, Tuple
from params import SimulationParameters


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


class GaussianGenerator(InputGenerator):

    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
        input_eigenspectrum: Float[torch.Tensor, "N_E"],
    ):
        super().__init__(parameters=parameters)
        assert torch.allclose(
            input_eigenbasis @ input_eigenbasis.T,
            torch.eye(self.N_E, device=self.device, dtype=self.dtype),
        ), "Input eigenbasis is not orthogonal"
        # Verify that the input eigenspectrum is positive
        assert torch.all(
            input_eigenspectrum >= 0
        ), "Input eigenspectrum has negative entries"

        self.input_eigenbasis = input_eigenbasis.to(
            device=self.device, dtype=self.dtype
        )
        self.input_eigenspectrum = input_eigenspectrum.to(
            device=self.device, dtype=self.dtype
        )
        self.covariance_sqrt = self.input_eigenbasis @ torch.diag(
            torch.sqrt(self.input_eigenspectrum)
        )

        self.noise_amplitude = torch.sqrt(
            torch.tensor(
                2.0 * self.dt / self.tau_u, device=self.device, dtype=self.dtype
            )
        )
        self.covariance_sqrt = self.covariance_sqrt.to(
            device=self.device, dtype=self.dtype
        )
        self.u = self.covariance_sqrt @ torch.randn(
            (self.N_E, 1), device=self.device, dtype=self.dtype
        )

    def __dict__(self) -> Dict[str, Any]:
        return {"tau_u": self.tau_u}

    @jaxtyped(typechecker=typechecked)
    def step(self) -> Float[torch.Tensor, "N_E 1"]:
        dB = torch.randn((self.N_E, 1), device=self.device, dtype=self.dtype)
        colored_noise = self.noise_amplitude * self.covariance_sqrt @ dB
        self.u = (1 - self.dt / self.tau_u) * self.u + colored_noise
        return self.u

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
        stimuli = torch.matmul(self.input_eigenbasis, mode_stimuli_contributions)
        return stimuli, mode_stimuli_contributions

    @jaxtyped(typechecker=typechecked)
    def mode_strengths(self) -> Float[torch.Tensor, "N_E"]:
        return self.input_eigenspectrum


class LaplacianGenerator(InputGenerator):

    def __init__(
        self,
        parameters: SimulationParameters,
        input_eigenbasis: Float[torch.Tensor, "N_E N_E"],
        input_eigenspectrum: Float[torch.Tensor, "N_E"],
    ):
        super().__init__(parameters=parameters)
        self.input_eigenbasis = input_eigenbasis
        self.input_eigenspectrum = input_eigenspectrum
        assert torch.allclose(
            input_eigenbasis @ input_eigenbasis.T,
            torch.eye(self.N_E, device=self.device, dtype=self.dtype),
        ), "Input eigenbasis is not orthogonal"
        # Verify that the input eigenspectrum is positive
        assert torch.all(
            input_eigenspectrum >= 0
        ), "Input eigenspectrum has negative entries"

        self.epsilon = 1e-12

        self.noise_amplitude = torch.sqrt(
            torch.tensor(
                2.0 * self.dt / self.tau_u, device=self.device, dtype=self.dtype
            )
        )
        self.scales = torch.sqrt(self.input_eigenspectrum / 2).unsqueeze(1)
        self.z = torch.randn((self.N_E, 1), device=self.device, dtype=self.dtype)

    def __dict__(self) -> Dict[str, Any]:
        return {"tau_u": self.tau_u}

    @jaxtyped(typechecker=typechecked)
    def step(self) -> Float[torch.Tensor, "N_E 1"]:
        dB = self.noise_amplitude * torch.randn(
            (self.N_E, 1), device=self.device, dtype=self.dtype
        )
        self.z = (1 - self.dt / self.tau_u) * self.z + dB

        x = self.gaussian_to_laplace(self.z)

        return self.input_eigenbasis @ x

    def gaussian_to_laplace(
        self, z: Float[torch.Tensor, "N_E num_stimuli"]
    ) -> Float[torch.Tensor, "N_E num_stimuli"]:
        uniforms = 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))
        signs = torch.sign(2 * uniforms - 1.0)
        abs_diff = torch.abs(2 * uniforms - 1.0)
        x = signs * self.scales * torch.log(abs_diff + self.epsilon)

        return x

    @jaxtyped(typechecker=typechecked)
    def stimuli_batch(self, num_stimuli: int) -> Tuple[
        Float[torch.Tensor, "N_E {num_stimuli}"],
        Float[torch.Tensor, "N_E {num_stimuli}"],
    ]:
        white_noise = torch.randn(
            (self.N_E, num_stimuli), device=self.device, dtype=self.dtype
        )
        mode_stimuli_contributions = self.gaussian_to_laplace(white_noise)
        stimuli = self.input_eigenbasis @ mode_stimuli_contributions
        return (
            stimuli,
            mode_stimuli_contributions,
        )

    @jaxtyped(typechecker=typechecked)
    def mode_strengths(self) -> Float[torch.Tensor, "N_E"]:
        return self.input_eigenspectrum


class CircularGenerator(InputGenerator):

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
        self._mode_strengths = self.mixing_parameter * torch.exp(
            torch.distributions.VonMises(
                loc=torch.zeros(1, device=self.device),
                concentration=self.vm_concentration,
            ).log_prob(self.positions)
        ) + (1 - self.mixing_parameter) / (2 * torch.pi)
        # Renormlise mode strenghts to sum to one
        self._mode_strengths /= self._mode_strengths.sum()

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
    def step(self) -> Float[torch.Tensor, "N_E 1"]:
        if self.steps_remaining == 0:
            self.current_input, _ = self.stimuli_batch(1)
            self.steps_remaining = int(self.tau_u / self.dt)
        else:
            self.steps_remaining -= 1

        return self.current_input

    @jaxtyped(typechecker=typechecked)
    def stimuli_batch(
        self, num_stimuli: int
    ) -> Tuple[
        Float[torch.Tensor, "N_E num_stimuli"], Float[torch.Tensor, "N_E num_stimuli"]
    ]:
        mode_indices = torch.multinomial(
            self._mode_strengths.squeeze(), num_samples=num_stimuli, replacement=True
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
            / self.tuning_width
        )

        return input_activities, mode_stimuli_contributions

    @jaxtyped(typechecker=typechecked)
    def sample_orientations(
        self, num_stimuli: int = 1
    ) -> Float[torch.Tensor, "1 num_stimuli"]:
        # Sample from the mode_strengths distribution
        mode_indices = torch.multinomial(
            self._mode_strengths, num_samples=num_stimuli, replacement=True
        )
        orientations = self.positions[mode_indices]

        return orientations.view(1, -1)

    @jaxtyped(typechecker=typechecked)
    def mode_strengths(self):
        return self._mode_strengths


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
