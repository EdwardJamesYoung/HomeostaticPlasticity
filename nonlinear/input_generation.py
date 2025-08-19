import torch
import numpy as np
from abc import ABC, abstractmethod
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Tuple, Union, TypeAlias
from dataclasses import dataclass
import scipy.stats
import scipy.optimize
from params import SimulationParameters


@dataclass
class DistributionConfig1D:
    mixing_parameter: float = 0.0
    concentration: float = 1.0
    location: float = 0.0


@dataclass
class DistributionConfig2D:
    mixing_parameter: float = 0.0
    concentration: float = 1.0
    location: Tuple[float, float] = (0.0, 0.0)

    def marginal(self, dim: int) -> DistributionConfig1D:
        assert dim in (0, 1), "Dimension must be 0 or 1"

        return DistributionConfig1D(
            mixing_parameter=self.mixing_parameter,
            concentration=self.concentration,
            location=self.location[dim],
        )


DistributionConfig: TypeAlias = Union[DistributionConfig1D, DistributionConfig2D]


def evaluate_mixture_1d(
    locations: Float[torch.Tensor, "batch 1"], config: DistributionConfig1D
) -> Float[torch.Tensor, "batch"]:
    """Evaluate mixture distribution at given angles"""
    device = locations.device
    dtype = locations.dtype

    mixing_parameter = config.mixing_parameter
    location = config.location
    concentration = config.concentration

    density = mixing_parameter * torch.exp(
        torch.distributions.VonMises(
            loc=location * torch.ones(1, device=device, dtype=dtype),
            concentration=concentration,
        ).log_prob(locations)
    ) + (1 - mixing_parameter) / (2 * torch.pi)

    density /= density.sum()
    density = density.squeeze(-1)  # [batch, 1] -> [batch]
    return density.to(device=device, dtype=dtype)


def mixture_cdf_1d(theta: float, config: DistributionConfig1D) -> float:
    """CDF of mixture distribution"""
    uniform_cdf = (theta + torch.pi) / (2 * torch.pi)  # CDF of uniform on [-π, π)
    von_mises_cdf = scipy.stats.vonmises.cdf(
        theta, config.concentration, loc=config.location
    )
    return (
        config.mixing_parameter * von_mises_cdf
        + (1 - config.mixing_parameter) * uniform_cdf
    )


def inverse_cdf_1d(u: float, config: DistributionConfig1D) -> float:
    """Inverse CDF of mixture distribution using numerical inversion"""

    def objective(theta):
        return mixture_cdf_1d(theta, config) - u

    result = scipy.optimize.brentq(objective, -torch.pi, torch.pi - 1e-6)
    return result


def evaluate_mixture_2d(
    locations: Float[torch.Tensor, "batch 2"], config: DistributionConfig2D
) -> Float[torch.Tensor, "batch"]:
    """Evaluate separable 2D mixture at given locations"""
    # locations: [batch, 2]
    theta1, theta2 = locations[:, 0], locations[:, 1]

    device = locations.device
    dtype = locations.dtype

    density1 = evaluate_mixture_1d(theta1.unsqueeze(1), config.marginal(0))  # [batch]
    density2 = evaluate_mixture_1d(theta2.unsqueeze(1), config.marginal(1))  # [batch]

    density = density1 * density2  # Outer product for 2D mixture
    density /= density.sum()
    return density.to(device=device, dtype=dtype)


class InputGenerator(ABC):

    def __init__(
        self,
        parameters: SimulationParameters,
        probability_config: DistributionConfig,
        density_config: DistributionConfig,
        gain_config: DistributionConfig,
        width_config: DistributionConfig,
        excitatory_third_factor_config: DistributionConfig,
        inhibitory_third_factor_config: DistributionConfig,
    ):
        self.probability_config = probability_config
        self.density_config = density_config
        self.gain_config = gain_config
        self.width_config = width_config
        self.excitatory_third_factor_config = excitatory_third_factor_config
        self.inhibitory_third_factor_config = inhibitory_third_factor_config

        self.parameters = parameters
        self.N_E = parameters.N_E
        self.num_stimuli = parameters.num_stimuli
        self.tuning_width = parameters.tuning_width
        self.device = parameters.device
        self.dtype = parameters.dtype

        # Initialise locations and pre-compute distance matrix
        self.neuron_locations = self._initialise_neuron_locations()
        self.stimuli_locations = self._initialise_stimuli_locations()
        self.distances = self._precompute_distances()

        # Initialise the gains, widths, and density
        self.input_density = self._compute_input_density()
        self.input_gains = self._compute_input_gains()
        self.input_widths = self._compute_input_widths()

        # Initialise the third factors
        self.excitatory_third_factor = self._compute_excitatory_third_factors()
        self.inhibitory_third_factor = self._compute_inhibitory_third_factors()

        # Compute the stimuli probabilities and input patterns
        self.stimuli_probabilities = self._compute_stimuli_probabilities()
        self.stimuli_patterns = self._compute_stimuli_patterns()

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _initialise_neuron_locations(
        self,
    ) -> Float[torch.Tensor, "{self.N_E} num_dimensions"]:
        """Initialise neuron locations on the manifold"""
        pass

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _initialise_stimuli_locations(
        self,
    ) -> Float[torch.Tensor, "{self.num_stimuli} num_dimensions"]:
        """Initialise stimulus locations on the manifold"""
        pass

    @jaxtyped(typechecker=typechecked)
    def _precompute_distances(
        self,
    ) -> Float[torch.Tensor, "{self.N_E} {self.num_stimuli}"]:
        return self._compute_circular_distance(
            self.neuron_locations, self.stimuli_locations
        )

    @jaxtyped(typechecker=typechecked)
    def _compute_circular_distance(
        self,
        loc_1: Float[torch.Tensor, "batch_1 num_dimensions"],
        loc_2: Float[torch.Tensor, "batch_2 num_dimensions"],
    ) -> Float[torch.Tensor, "batch_1 batch_2"]:

        diff = torch.abs(
            loc_1.unsqueeze(1) - loc_2.unsqueeze(0)
        )  # [batch_1, batch_2, num_dimensions]
        circular_diffs = torch.minimum(
            diff, 2 * torch.pi - diff
        )  # [batch_1, batch_2, num_dimensions]

        return torch.sqrt(torch.sum(circular_diffs**2, dim=-1))  # [batch_1, batch_2]

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def evaluate_mixture(
        self,
        locations: Float[torch.Tensor, "batch num_dimensions"],
        config: Union[DistributionConfig1D, DistributionConfig2D],
    ) -> Float[torch.Tensor, "batch"]:
        pass

    @jaxtyped(typechecker=typechecked)
    def _compute_stimuli_probabilities(
        self,
    ) -> Float[torch.Tensor, "{self.num_stimuli}"]:
        return self.evaluate_mixture(self.stimuli_locations, self.probability_config)

    @jaxtyped(typechecker=typechecked)
    def _compute_input_density(self) -> Float[torch.Tensor, "{self.num_stimuli}"]:
        """Compute density at stimulus locations"""
        return self.evaluate_mixture(self.stimuli_locations, self.density_config)

    @jaxtyped(typechecker=typechecked)
    def _compute_input_gains(self) -> Float[torch.Tensor, "{self.num_stimuli}"]:
        """Compute gains at stimulus locations"""
        gains = self.evaluate_mixture(self.stimuli_locations, self.gain_config)
        return gains / gains.mean()

    @jaxtyped(typechecker=typechecked)
    def _compute_excitatory_third_factors(
        self,
    ) -> Float[torch.Tensor, "{self.num_stimuli}"]:
        """Compute excitatory third factors at stimulus locations"""
        q_e = self.evaluate_mixture(
            self.stimuli_locations, self.excitatory_third_factor_config
        )
        return q_e / q_e.mean()

    @jaxtyped(typechecker=typechecked)
    def _compute_inhibitory_third_factors(
        self,
    ) -> Float[torch.Tensor, "{self.num_stimuli}"]:
        """Compute inhibitory third factors at stimulus locations"""
        q_i = self.evaluate_mixture(
            self.stimuli_locations, self.inhibitory_third_factor_config
        )
        return q_i / q_i.mean()

    @jaxtyped(typechecker=typechecked)
    def _compute_input_widths(self) -> Float[torch.Tensor, "{self.num_stimuli}"]:
        """Compute width modulation at stimulus locations"""
        inverse_widths = self.evaluate_mixture(
            self.stimuli_locations, self.width_config
        )
        widths = 1.0 / (inverse_widths + 1e-8)
        return widths / widths.mean()

    @jaxtyped(typechecker=typechecked)
    def _compute_stimuli_patterns(
        self,
    ) -> Float[torch.Tensor, "{self.N_E} {self.num_stimuli}"]:
        """
        Compute tuning curves: g(s) * h([s - l]/w(s))
        """
        # Apply width scaling: distances / widths[stimulus]
        scaled_distances = self.distances / (
            self.tuning_width * self.input_widths.unsqueeze(0)
        )  # [N_E, num_stimuli]

        # Apply wrapped Gaussian kernel
        kernel_responses = torch.exp(-0.5 * scaled_distances**2)  # [N_E, num_stimuli]

        # Apply gains: multiply by gain[stimulus]
        responses = kernel_responses * self.input_gains.unsqueeze(
            0
        )  # [N_E, num_stimuli]

        return responses


class CircularInputGenerator(InputGenerator):
    """Input generator for circular stimulus space S^1"""

    def __init__(
        self,
        parameters: SimulationParameters,
        probability_config: DistributionConfig1D,
        density_config: DistributionConfig1D,
        gain_config: DistributionConfig1D,
        width_config: DistributionConfig1D,
        excitatory_third_factor_config: DistributionConfig1D,
        inhibitory_third_factor_config: DistributionConfig1D,
    ):
        super().__init__(
            parameters=parameters,
            probability_config=probability_config,
            density_config=density_config,
            gain_config=gain_config,
            width_config=width_config,
            excitatory_third_factor_config=excitatory_third_factor_config,
            inhibitory_third_factor_config=inhibitory_third_factor_config,
        )

    @jaxtyped(typechecker=typechecked)
    def evaluate_mixture(
        self,
        locations: Float[torch.Tensor, "{self.num_stimuli} 1"],
        config: DistributionConfig1D,
    ):
        return evaluate_mixture_1d(locations=locations, config=config)

    @jaxtyped(typechecker=typechecked)
    def _initialise_stimuli_locations(
        self,
    ) -> Float[torch.Tensor, "{self.num_stimuli} 1"]:
        """Initialize uniform stimulus grid"""
        angles = torch.linspace(
            -torch.pi,
            torch.pi,
            self.num_stimuli + 1,
            device=self.device,
            dtype=self.dtype,
        )[:-1]
        return angles.unsqueeze(-1)  # [num_stimuli, 1]

    @jaxtyped(typechecker=typechecked)
    def _initialise_neuron_locations(self) -> Float[torch.Tensor, "{self.N_E} 1"]:
        """Initialize neuron locations using inverse CDF of density"""
        # Create uniform grid in [0, 1)
        uniform_grid = torch.linspace(
            0, 1, self.N_E + 1, device=self.device, dtype=self.dtype
        )[:-1]

        # Apply inverse CDF to get angles in [-π, π)
        if self.density_config.mixing_parameter == 0:
            # Pure uniform case
            angles = 2 * torch.pi * uniform_grid - torch.pi
        else:
            # Need to compute inverse CDF numerically
            angles = torch.tensor(
                [inverse_cdf_1d(u.item(), self.density_config) for u in uniform_grid],
                device=self.device,
                dtype=self.dtype,
            )

        return angles.unsqueeze(-1)  # [N_E, 1]


class TorusInputGenerator(InputGenerator):

    def __init__(
        self,
        parameters: SimulationParameters,
        probability_config: DistributionConfig2D,
        density_config: DistributionConfig2D,
        gain_config: DistributionConfig2D,
        width_config: DistributionConfig2D,
        excitatory_third_factor_config: DistributionConfig2D,
        inhibitory_third_factor_config: DistributionConfig2D,
    ):
        # Validate that N_E and num_stimuli are perfect squares
        N_E_sqrt = int(np.sqrt(parameters.N_E))
        if N_E_sqrt * N_E_sqrt != parameters.N_E:
            raise ValueError(f"N_E must be a perfect square, got {parameters.N_E}")

        num_stimuli_sqrt = int(np.sqrt(parameters.num_stimuli))
        if num_stimuli_sqrt * num_stimuli_sqrt != parameters.num_stimuli:
            raise ValueError(
                f"num_stimuli must be a perfect square, got {parameters.num_stimuli}"
            )

        self.N_E_sqrt = N_E_sqrt
        self.num_stimuli_sqrt = num_stimuli_sqrt

        super().__init__(
            parameters=parameters,
            probability_config=probability_config,
            density_config=density_config,
            gain_config=gain_config,
            width_config=width_config,
            excitatory_third_factor_config=excitatory_third_factor_config,
            inhibitory_third_factor_config=inhibitory_third_factor_config,
        )

    @jaxtyped(typechecker=typechecked)
    def evaluate_mixture(
        self,
        locations: Float[torch.Tensor, "{self.num_stimuli} 2"],
        config: DistributionConfig2D,
    ):
        return evaluate_mixture_2d(locations=locations, config=config)

    @jaxtyped(typechecker=typechecked)
    def _initialise_neuron_locations(self) -> Float[torch.Tensor, "{self.N_E} 2"]:
        """Initialize neuron locations using separable inverse CDF"""
        # Create uniform grids for each dimension
        uniform_grid = torch.linspace(
            0, 1, self.N_E_sqrt + 1, device=self.device, dtype=self.dtype
        )[:-1]

        if self.density_config.mixing_parameter == 0:
            angles_0 = 2 * torch.pi * uniform_grid - torch.pi
            angles_1 = 2 * torch.pi * uniform_grid - torch.pi
        else:
            angles_0 = torch.tensor(
                [
                    inverse_cdf_1d(u.item(), self.density_config.marginal(0))
                    for u in uniform_grid
                ],
                device=self.device,
                dtype=self.dtype,
            )
            angles_1 = torch.tensor(
                [
                    inverse_cdf_1d(u.item(), self.density_config.marginal(1))
                    for u in uniform_grid
                ],
                device=self.device,
                dtype=self.dtype,
            )

        # Create 2D grid via outer product
        theta0_grid, theta1_grid = torch.meshgrid(angles_0, angles_1, indexing="ij")
        return torch.stack(
            [theta0_grid.flatten(), theta1_grid.flatten()], dim=-1
        )  # [N_E, 2]

    @jaxtyped(typechecker=typechecked)
    def _initialise_stimuli_locations(
        self,
    ) -> Float[torch.Tensor, "{self.num_stimuli} 2"]:
        """Initialise uniform stimulus grid"""
        angles = torch.linspace(
            -torch.pi,
            torch.pi,
            self.num_stimuli_sqrt + 1,
            device=self.device,
            dtype=self.dtype,
        )[:-1]
        theta0_grid, theta1_grid = torch.meshgrid(angles, angles, indexing="ij")
        return torch.stack(
            [theta0_grid.flatten(), theta1_grid.flatten()], dim=-1
        )  # [num_stimuli, 2]
