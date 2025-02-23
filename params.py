import torch
from dataclasses import dataclass, fields, asdict
from typing import Optional
from activation_functions import *

ACTIVATION_FUNCTION_MAP = {
    "rectified_quadratic": RectifiedQuadratic,
    "rectified_linear": RectifiedLinear,
    "rectified_cubic": RectifiedCubic,
    "cubic": Cubic,
    "linear": Linear,
}


@dataclass
class SimulationParameters:
    N_E: int = 10
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
    variable_input_mass: bool = True
    variance_homeostasis: bool = False
    rate_homeostasis: bool = False
    target_rate: Optional[float] = None
    target_variance: Optional[float] = None
    omega: float = 1.0
    activation_function_name: str = "rectified_quadratic"
    activation_function: ActivationFunction = RectifiedQuadratic()
    covariance_learning: bool = True
    dynamics_log_time: float = 20.0
    mode_log_time: float = 100.0
    num_samples: int = 10000
    wandb_logging: bool = False
    random_seed: int = 0
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
        assert not (
            self.variance_homeostasis and self.rate_homeostasis
        ), "Cannot have both rate and variance homeostasis"
        if self.rate_homeostasis:
            if self.target_rate is None:
                self.target_rate = 1.0
            self.target_variance = None

        if self.variance_homeostasis:
            if self.target_variance is None:
                self.target_variance = 0.002
            self.target_rate = None

        assert (
            self.target_rate is not None or self.target_variance is not None
        ), "Must specify either target rate or target variance"

        # Check whether the activation function is the same as the activation function name
        if self.activation_function_name in ACTIVATION_FUNCTION_MAP:
            if not isinstance(
                self.activation_function,
                ACTIVATION_FUNCTION_MAP[self.activation_function_name],
            ):
                self.activation_function = ACTIVATION_FUNCTION_MAP[
                    self.activation_function_name
                ]()

    def to_dict(self):
        return asdict(self)
