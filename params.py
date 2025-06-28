import torch
from dataclasses import dataclass, fields, asdict
from typing import Optional
from activation_functions import *

ACTIVATION_FUNCTION_MAP = {
    "rectified_quadratic": RectifiedQuadratic,
    "rectified_linear": RectifiedLinear,
    "rectified_powerlaw_1p5": RectifiedPowerlaw1p5,
    "rectified_cubic": RectifiedCubic,
    "cubic": Cubic,
    "linear": Linear,
}


@dataclass
class SimulationParameters:
    batch_size: int = 1
    N_E: int = 10
    num_latents: Optional[int] = None
    N_I: int = 100
    k_I: float = 10.0
    T: float = 120000.0
    dt: float = 0.05
    tau_v: float = 1.0
    tau_u: float = 10.0
    tau_M: float = 50.0
    tau_W: float = 250.0
    tau_k: float = 500.0
    zeta: float = 1.0
    alpha: float = 1.0
    homeostasis: bool = True
    homeostasis_power: float = 1.0
    homeostasis_target: float = 1.0
    omega: float = 1.0
    initial_feedforward_weight_scaling: float = 1.0
    activation_function_name: str = "rectified_quadratic"
    activation_function: ActivationFunction = RectifiedQuadratic()
    feedforward_covariance_learning: bool = True
    recurrent_covariance_learning: bool = True
    feedforward_voltage_learning: bool = False
    recurrent_voltage_learning: bool = False
    dynamics_log_time: float = 20.0
    mode_log_time: float = 100.0
    num_samples: int = 10000
    wandb_logging: bool = False
    random_seed: int = 0
    rate_computation_threshold: float = 1e-4
    rate_computation_iterations: int = 10000
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float64

    def __init__(self, **kwargs):
        field_defaults = {f.name: f.default for f in fields(self)}

        # Get the set of valid field names
        valid_fields = set(field_defaults.keys())

        # Filter kwargs to only include valid fields
        valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        # Initialize with filtered kwargs
        for k, v in valid_kwargs.items():
            setattr(self, k, v)

        # Run post-init checks
        self.__post_init__()

    def __post_init__(self):

        # Check that the homeostasis power is positive
        assert (
            self.homeostasis_power > 0
        ), f"Homeostasis power must be positive. Got {self.homeostasis_power}."

        # Check whether the activation function is the same as the activation function name
        if self.activation_function_name in ACTIVATION_FUNCTION_MAP:
            if not isinstance(
                self.activation_function,
                ACTIVATION_FUNCTION_MAP[self.activation_function_name],
            ):
                self.activation_function = ACTIVATION_FUNCTION_MAP[
                    self.activation_function_name
                ]()

        # Set the number of latents if not specified
        if self.num_latents is None:
            self.num_latents = self.N_E

    def to_dict(self):
        return asdict(self)
