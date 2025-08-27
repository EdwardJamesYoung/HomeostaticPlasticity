import torch
import numpy as np
from abc import ABC, abstractmethod
from jaxtyping import Float, jaxtyped
from typeguard import typechecked
from typing import Tuple, Union, TypeAlias, List
from dataclasses import dataclass
import scipy.stats
import scipy.optimize
from params import SimulationParameters
from itertools import product
from utils import compute_circular_distance


@jaxtyped(typechecker=typechecked)
@dataclass
class DistributionConfig1D:
    mixing_parameter: Float[torch.Tensor, "#batch 1"] = torch.zeros(1, 1)
    concentration: Float[torch.Tensor, "#batch 1"] = torch.ones(1, 1)
    location: Float[torch.Tensor, "#batch 1"] = torch.zeros(1, 1)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        # Ensure all tensors are on the correct device
        self.mixing_parameter = self.mixing_parameter.to(self.device)
        self.concentration = self.concentration.to(self.device)
        self.location = self.location.to(self.device)

        # Broadcast so that all the batch sizes are the same shape
        self.batch_size = max(
            self.mixing_parameter.shape[0],
            self.concentration.shape[0],
            self.location.shape[0],
        )

        self.mixing_parameter, self.concentration, self.location = (
            torch.broadcast_tensors(
                self.mixing_parameter,
                self.concentration,
                self.location,
            )
        )


@dataclass
@jaxtyped(typechecker=typechecked)
class DistributionConfig2D:
    mixing_parameter: Float[torch.Tensor, "#batch 1"] = torch.zeros(1, 1)
    concentration: Float[torch.Tensor, "#batch 1"] = torch.ones(1, 1)
    location: Float[torch.Tensor, "#batch 2"] = torch.zeros(1, 2)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        # Ensure all tensors are on the correct device
        self.mixing_parameter = self.mixing_parameter.to(self.device)
        self.concentration = self.concentration.to(self.device)
        self.location = self.location.to(self.device)

        self.batch_size = max(
            self.mixing_parameter.shape[0],
            self.concentration.shape[0],
            self.location.shape[0],
        )

        # Broadcast only the first dimension (batch) to match batch_size
        self.mixing_parameter = self.mixing_parameter.expand(self.batch_size, -1)
        self.concentration = self.concentration.expand(self.batch_size, -1)
        self.location = self.location.expand(self.batch_size, -1)

    def marginal(self, dim: int) -> DistributionConfig1D:
        assert dim in (0, 1), "Dimension must be 0 or 1"

        return DistributionConfig1D(
            mixing_parameter=self.mixing_parameter,
            concentration=self.concentration,
            location=self.location[:, dim : dim + 1],
        )


DistributionConfig: TypeAlias = Union[DistributionConfig1D, DistributionConfig2D]


@jaxtyped(typechecker=typechecked)
def create_config_grid_1d(
    mixing_parameters: List[float],
    concentrations: List[float],
    locations: List[float],
) -> DistributionConfig1D:
    """
    Create a DistributionConfig1D with a grid of all parameter combinations.

    Args:
        mixing_parameters: List of mixing parameter values
        concentrations: List of concentration values
        locations: List of location values
        device: Device to place tensors on (if None, uses dataclass default)

    Returns:
        DistributionConfig1D with batch_size = len(mixing_parameters) * len(concentrations) * len(locations)

    Example:
        config = create_config_grid_1d(
            mixing_parameters=[0.0, 0.5],
            concentrations=[1.0, 2.0],
            locations=[0.0, 1.57]
        )
        # Results in batch_size = 8 with all combinations
    """
    # Create Cartesian product of all parameter combinations
    param_combinations = list(product(mixing_parameters, concentrations, locations))

    # Unpack into separate lists
    mixing_vals, concentration_vals, location_vals = zip(*param_combinations)

    # Convert to tensors with shape [batch, 1]
    mixing_tensor = torch.tensor(mixing_vals, dtype=torch.float).unsqueeze(1)
    concentration_tensor = torch.tensor(
        concentration_vals, dtype=torch.float
    ).unsqueeze(1)
    location_tensor = torch.tensor(location_vals, dtype=torch.float).unsqueeze(1)

    config = DistributionConfig1D(
        mixing_parameter=mixing_tensor,
        concentration=concentration_tensor,
        location=location_tensor,
    )

    return config


@jaxtyped(typechecker=typechecked)
def create_config_grid_2d(
    mixing_parameters: List[float],
    concentrations: List[float],
    locations: List[List[float]],  # Each element is [loc_0, loc_1]
) -> DistributionConfig2D:
    """
    Create a DistributionConfig2D with a grid of all parameter combinations.

    Args:
        mixing_parameters: List of mixing parameter values
        concentrations: List of concentration values
        locations: List of 2D location pairs [[x1, y1], [x2, y2], ...]

    Returns:
        DistributionConfig2D with batch_size = len(mixing_parameters) * len(concentrations) * len(locations)

    Example:
        config = create_config_grid_2d(
            mixing_parameters=[0.0, 0.5],
            concentrations=[1.0, 2.0],
            locations=[[0.0, 0.0], [1.57, 1.57]]
        )
        # Results in batch_size = 8 with all combinations
    """
    # Create Cartesian product of all parameter combinations
    param_combinations = list(product(mixing_parameters, concentrations, locations))

    # Unpack into separate lists
    mixing_vals, concentration_vals, location_vals = zip(*param_combinations)

    # Convert to tensors
    mixing_tensor = torch.tensor(mixing_vals, dtype=torch.float).unsqueeze(
        1
    )  # [batch, 1]
    concentration_tensor = torch.tensor(
        concentration_vals, dtype=torch.float
    ).unsqueeze(
        1
    )  # [batch, 1]
    location_tensor = torch.tensor(location_vals, dtype=torch.float)  # [batch, 2]

    config = DistributionConfig2D(
        mixing_parameter=mixing_tensor,
        concentration=concentration_tensor,
        location=location_tensor,
    )

    return config


@jaxtyped(typechecker=typechecked)
def evaluate_mixture_1d(
    eval_positions: Float[torch.Tensor, "num_points 1"], config: DistributionConfig1D
) -> Float[torch.Tensor, "config_batch num_points"]:
    device = eval_positions.device
    dtype = eval_positions.dtype

    # Broadcast config parameters to final batch size
    batch_size = config.batch_size
    mixing_parameter = config.mixing_parameter.expand(batch_size, 1)  # [batch, 1]
    concentration = config.concentration.expand(batch_size, 1)  # [batch, 1]
    location = config.location.expand(batch_size, 1)  # [batch, 1]

    # Compute von Mises component: [batch, num_points]
    vm_log_prob = torch.distributions.VonMises(
        loc=location,  # [batch, 1]
        concentration=concentration,  # [batch, 1]
    ).log_prob(
        eval_positions.squeeze(-1).unsqueeze(0)
    )  # [batch, num_points]

    density = mixing_parameter * torch.exp(vm_log_prob) + (1 - mixing_parameter) / (
        2 * torch.pi
    )  # [batch, num_points]

    # Normalize each batch element
    density = density / density.sum(dim=1, keepdim=True)  # [batch, num_points]

    return density.to(device=device, dtype=dtype)


def mixture_cdf_1d(
    theta: float, mixing_parameter: float, concentration: float, location: float
) -> float:
    """CDF of mixture distribution"""
    uniform_cdf = (theta + torch.pi) / (2 * torch.pi)
    von_mises_cdf = scipy.stats.vonmises.cdf(theta, concentration, loc=location)
    return mixing_parameter * von_mises_cdf + (1 - mixing_parameter) * uniform_cdf


def inverse_cdf_1d(
    u: float, mixing_parameter: float, concentration: float, location: float
) -> float:
    """Inverse CDF of mixture distribution using numerical inversion"""

    def objective(theta):
        return mixture_cdf_1d(theta, mixing_parameter, concentration, location) - u

    result = scipy.optimize.brentq(objective, -torch.pi, torch.pi - 1e-6)
    return result


@jaxtyped(typechecker=typechecked)
def compute_unique_inverse_cdf_1d(
    uniform_grid: Float[torch.Tensor, "num_points"], config: DistributionConfig1D
) -> Float[torch.Tensor, "batch num_points"]:
    """Efficiently compute inverse CDF for batched config using unique parameter combinations"""
    device = config.mixing_parameter.device
    dtype = config.mixing_parameter.dtype

    # Stack parameters for uniqueness detection
    params = torch.cat(
        [
            config.mixing_parameter,  # [batch, 1]
            config.concentration,  # [batch, 1]
            config.location,  # [batch, 1]
        ],
        dim=1,
    )  # [batch, 3]

    # Find unique parameter combinations
    unique_params, inverse_indices = torch.unique(params, dim=0, return_inverse=True)
    num_unique = unique_params.shape[0]

    # Compute inverse CDF only for unique parameter combinations
    results = torch.zeros(num_unique, len(uniform_grid), device=device, dtype=dtype)

    for i in range(num_unique):
        mixing_param = unique_params[i, 0].item()
        concentration = unique_params[i, 1].item()
        location = unique_params[i, 2].item()

        # Handle pure uniform case analytically
        if mixing_param == 0:
            angles = 2 * torch.pi * uniform_grid - torch.pi
        else:
            # Compute numerically for mixed distributions
            angles = torch.tensor(
                [
                    inverse_cdf_1d(u.item(), mixing_param, concentration, location)
                    for u in uniform_grid
                ],
                device=device,
                dtype=dtype,
            )

        results[i] = angles

    # Broadcast results back to full batch size using inverse indices
    batch_results = results[inverse_indices]  # [batch, num_points]

    return batch_results


@jaxtyped(typechecker=typechecked)
def evaluate_mixture_2d(
    locations: Float[torch.Tensor, "num_points 2"], config: DistributionConfig2D
) -> Float[torch.Tensor, "batch num_points"]:
    """Evaluate batched separable 2D mixture at given locations"""
    # Extract angles for each dimension
    theta1_locations = locations[:, 0:1]  # [num_points, 1]
    theta2_locations = locations[:, 1:2]  # [num_points, 1]

    # Evaluate marginal distributions
    density1 = evaluate_mixture_1d(
        theta1_locations, config.marginal(0)
    )  # [batch, num_points]
    density2 = evaluate_mixture_1d(
        theta2_locations, config.marginal(1)
    )  # [batch, num_points]

    # Compute product (separable distribution)
    density = density1 * density2  # [batch, num_points]

    # Normalize each batch element
    density = density / density.sum(dim=1, keepdim=True)

    return density


class InputGenerator(ABC):
    """Base class for batched input generators"""

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
        # Validate that all configs have compatible batch sizes
        configs = [
            probability_config,
            density_config,
            gain_config,
            width_config,
            excitatory_third_factor_config,
            inhibitory_third_factor_config,
        ]
        batch_sizes = [config.batch_size for config in configs]
        self.batch_size = max(batch_sizes)

        # Store configs
        self.probability_config = probability_config
        self.density_config = density_config
        self.gain_config = gain_config
        self.width_config = width_config
        self.excitatory_third_factor_config = excitatory_third_factor_config
        self.inhibitory_third_factor_config = inhibitory_third_factor_config

        # Store parameters
        self.parameters = parameters
        self.N_E = parameters.N_E
        self.num_stimuli = parameters.num_stimuli
        self.tuning_width = parameters.tuning_width
        self.device = parameters.device
        self.dtype = parameters.dtype

        # Initialize locations (computed once, then broadcasted as needed)
        self.neuron_locations = (
            self._initialise_neuron_locations()
        )  # [batch, N_E, num_dims]
        self.stimuli_locations = (
            self._initialise_stimuli_locations()
        )  # [num_stimuli, num_dims]

        # Precompute distance matrix
        self.distances = self._precompute_distances()  # [batch, N_E, num_stimuli]

        # Initialize distributions (lazy broadcasting)
        self.input_density = self._compute_input_density()  # [batch, num_stimuli]
        self.input_gains = self._compute_input_gains()  # [batch, num_stimuli]
        self.input_widths = self._compute_input_widths()  # [batch, num_stimuli]
        self.excitatory_third_factor = (
            self._compute_excitatory_third_factors()
        )  # [batch, num_stimuli]
        self.inhibitory_third_factor = (
            self._compute_inhibitory_third_factors()
        )  # [batch, num_stimuli]

        # Compute final outputs
        self.stimuli_probabilities = (
            self._compute_stimuli_probabilities()
        )  # [batch, num_stimuli]
        self.stimuli_patterns = (
            self._compute_stimuli_patterns()
        )  # [batch, N_E, num_stimuli]

        self.convolved_probabilities = (
            self._compute_convolved_probabilities()
        )  # [batch, num_stimuli]

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _initialise_neuron_locations(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.N_E} num_dimensions"]:
        """Initialize neuron locations on the manifold"""
        pass

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _initialise_stimuli_locations(
        self,
    ) -> Float[torch.Tensor, "{self.num_stimuli} num_dimensions"]:
        """Initialize stimulus locations on the manifold"""
        pass

    @jaxtyped(typechecker=typechecked)
    def _precompute_distances(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.N_E} {self.num_stimuli}"]:
        """Precompute distance matrix between neurons and stimuli"""
        return compute_circular_distance(
            self.neuron_locations, self.stimuli_locations.unsqueeze(0)
        )

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def evaluate_mixture(
        self,
        locations: Float[torch.Tensor, "num_points num_dimensions"],
        config: DistributionConfig,
    ) -> Float[torch.Tensor, "batch num_points"]:
        """Evaluate mixture distribution at given locations"""
        pass

    @jaxtyped(typechecker=typechecked)
    def _compute_stimuli_probabilities(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.num_stimuli}"]:
        """Compute probability of each stimulus"""
        # Use first batch element's stimuli locations for evaluation
        return self.evaluate_mixture(self.stimuli_locations, self.probability_config)

    @jaxtyped(typechecker=typechecked)
    def _compute_input_density(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.num_stimuli}"]:
        """Compute density at stimulus locations"""
        return self.evaluate_mixture(self.stimuli_locations, self.density_config)

    @jaxtyped(typechecker=typechecked)
    def _compute_input_gains(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.num_stimuli}"]:
        """Compute gains at stimulus locations"""
        gains = self.evaluate_mixture(self.stimuli_locations, self.gain_config)
        return gains / gains.mean(dim=-1, keepdim=True)

    @jaxtyped(typechecker=typechecked)
    def _compute_excitatory_third_factors(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.num_stimuli}"]:
        """Compute excitatory third factors at stimulus locations"""
        q_e = self.evaluate_mixture(
            self.stimuli_locations, self.excitatory_third_factor_config
        )
        return q_e / q_e.mean(dim=-1, keepdim=True)

    @jaxtyped(typechecker=typechecked)
    def _compute_inhibitory_third_factors(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.num_stimuli}"]:
        """Compute inhibitory third factors at stimulus locations"""
        q_i = self.evaluate_mixture(
            self.stimuli_locations, self.inhibitory_third_factor_config
        )
        return q_i / q_i.mean(dim=-1, keepdim=True)

    @jaxtyped(typechecker=typechecked)
    def _compute_input_widths(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.num_stimuli}"]:
        """Compute width modulation at stimulus locations"""
        inverse_widths = self.evaluate_mixture(
            self.stimuli_locations, self.width_config
        )
        widths = 1.0 / (inverse_widths + 1e-8)
        return widths / widths.mean(dim=-1, keepdim=True)

    @jaxtyped(typechecker=typechecked)
    def _compute_stimuli_patterns(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.N_E} {self.num_stimuli}"]:
        """Compute tuning curves: g(s) * h([s - l]/w(s))"""
        # Apply width scaling: distances / widths[stimulus]
        # distances: [batch, N_E, num_stimuli], widths: [batch, num_stimuli]
        scaled_distances = self.distances / (
            self.tuning_width * self.input_widths.unsqueeze(1)
        )  # [batch, N_E, num_stimuli]

        # Apply wrapped Gaussian kernel
        kernel_responses = torch.exp(
            -0.5 * scaled_distances**2
        )  # [batch, N_E, num_stimuli]
        # Normalise the responses so that the sum for each neuron is equal to 1
        kernel_responses = kernel_responses / kernel_responses.mean(
            dim=-1, keepdim=True
        )

        # Apply gains: multiply by gain[stimulus]
        # gains: [batch, num_stimuli] -> [batch, 1, num_stimuli]
        responses = kernel_responses * self.input_gains.unsqueeze(
            1
        )  # [batch, N_E, num_stimuli]

        return responses

    @jaxtyped(typechecker=typechecked)
    def _compute_convolved_probabilities(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.num_stimuli}"]:
        self.stimuli_locations  # [num_stimuli, num_dims]
        self.stimuli_probabilities  # [batch, num_stimuli]
        scaled_stimuli_distances = (
            compute_circular_distance(self.stimuli_locations, self.stimuli_locations)
            / self.tuning_width
        )
        convolution_kernel = torch.exp(
            -0.5 * scaled_stimuli_distances**2
        )  # [num_stimuli, num_stimuli]

        convolved_probabilities = torch.einsum(
            "bs,sz->bz", self.stimuli_probabilities, convolution_kernel
        )  # [batch, num_stimuli]

        # Normalise the convolved probabilities
        convolved_probabilities = (
            convolved_probabilities / convolved_probabilities.mean(dim=-1, keepdim=True)
        )

        return convolved_probabilities


class CircularInputGenerator(InputGenerator):
    """Batched input generator for circular stimulus space S^1"""

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
    ) -> Float[torch.Tensor, "{self.batch_size} {self.num_stimuli}"]:
        mixture = evaluate_mixture_1d(
            eval_positions=locations, config=config
        )  # [#batch, num_stimuli]

        # Broadcast to batch size
        return mixture.expand(self.batch_size, -1)  # [batch, num_stimuli]

    @jaxtyped(typechecker=typechecked)
    def _initialise_stimuli_locations(
        self,
    ) -> Float[torch.Tensor, "{self.num_stimuli} 1"]:
        """Initialize uniform stimulus grid (same for all batch elements)"""
        stimuli_locations = torch.linspace(
            -torch.pi,
            torch.pi,
            self.num_stimuli + 1,
            device=self.device,
            dtype=self.dtype,
        )[:-1]

        return stimuli_locations.unsqueeze(-1)

    @jaxtyped(typechecker=typechecked)
    def _initialise_neuron_locations(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.N_E} 1"]:
        """Initialize neuron locations using inverse CDF of density"""
        # Create uniform grid in [0, 1)
        uniform_grid = torch.linspace(
            0, 1, self.N_E + 1, device=self.device, dtype=self.dtype
        )[:-1]

        # Compute inverse CDF efficiently for batched density config
        angles_batch = compute_unique_inverse_cdf_1d(
            uniform_grid, self.density_config
        )  # [#batch, N_E]

        # Ensure that the leading dimension is batch rather than 1
        angles_batch = angles_batch.expand(self.batch_size, -1)  # [batch, N_E]

        return angles_batch.unsqueeze(-1)  # [batch, N_E, 1]


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
    ) -> Float[torch.Tensor, "{self.batch_size} {self.num_stimuli}"]:
        mixture = evaluate_mixture_2d(locations=locations, config=config)
        return mixture.expand(self.batch_size, -1)

    @jaxtyped(typechecker=typechecked)
    def _initialise_neuron_locations(
        self,
    ) -> Float[torch.Tensor, "{self.batch_size} {self.N_E} 2"]:
        """Initialize neuron locations using separable inverse CDF"""
        # Create uniform grids for each dimension
        uniform_grid = torch.linspace(
            0, 1, self.N_E_sqrt + 1, device=self.device, dtype=self.dtype
        )[:-1]

        # Compute inverse CDF for each marginal
        angles_0_batch = compute_unique_inverse_cdf_1d(
            uniform_grid, self.density_config.marginal(0)
        )  # [#batch, N_E_sqrt]
        angles_1_batch = compute_unique_inverse_cdf_1d(
            uniform_grid, self.density_config.marginal(1)
        )  # [#batch, N_E_sqrt]

        # Create 2D grids for each batch element
        batch_neuron_locations = []
        for b in range(angles_0_batch.shape[0]):
            # Create 2D grid via outer product for this batch element
            theta0_grid, theta1_grid = torch.meshgrid(
                angles_0_batch[b], angles_1_batch[b], indexing="ij"
            )
            locations_2d = torch.stack(
                [theta0_grid.flatten(), theta1_grid.flatten()], dim=-1
            )  # [N_E, 2]
            batch_neuron_locations.append(locations_2d)

        locations = torch.stack(batch_neuron_locations, dim=0)  # [#batch, N_E, 2]
        # if locations.shape[0] = 1, expand along the first dimension
        if locations.shape[0] == 1:
            locations = locations.expand(self.batch_size, -1, -1)
        return locations

    @jaxtyped(typechecker=typechecked)
    def _initialise_stimuli_locations(
        self,
    ) -> Float[torch.Tensor, "{self.num_stimuli} 2"]:
        """Initialize uniform stimulus grid (same for all batch elements)"""
        angles = torch.linspace(
            -torch.pi,
            torch.pi,
            self.num_stimuli_sqrt + 1,
            device=self.device,
            dtype=self.dtype,
        )[:-1]

        theta0_grid, theta1_grid = torch.meshgrid(angles, angles, indexing="ij")
        stimuli_locations = torch.stack(
            [theta0_grid.flatten(), theta1_grid.flatten()], dim=-1
        )  # [num_stimuli, 2]

        return stimuli_locations
