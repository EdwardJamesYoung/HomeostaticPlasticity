import torch
from dataclasses import dataclass, fields, asdict
from typing import Optional
from activation_functions import *

ACTIVATION_FUNCTION_MAP = {
    "rectified_quadratic": RectifiedQuadratic,
    "rectified_linear": RectifiedLinear,
    "rectified_powerlaw_1p5": RectifiedPowerlaw1p5,
    "rectified_cubic": RectifiedCubic,
}


@dataclass
class SimulationParameters:
    repeats: int = 1
    N_E: int = 100
    N_I: int = 100
    num_stimuli: int = 100
    tuning_width: float = 0.25
    k_I: float = 16.0
    T: float = 2500.0
    dt_v: float = 0.05
    tau_v: float = 1.0
    dt: float = 0.05
    tau_M: float = 1.0
    tau_W: float = 4.0
    tau_k: float = 20.0
    zeta: float = 1.0
    gamma: float = 1.0
    homeostasis: bool = True
    homeostasis_power: float = 1.0
    homeostasis_target: float = 1.0
    activation_function_name: str = "rectified_linear"
    activation_function: ActivationFunction = RectifiedLinear()
    covariance_learning: bool = False
    voltage_learning: bool = False
    log_time: float = 20
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
        if self.num_stimuli is None:
            self.num_stimuli = self.N_E

    def to_dict(self):
        return asdict(self)
