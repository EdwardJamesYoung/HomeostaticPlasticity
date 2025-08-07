from typing import Optional
import torch
from dataclasses import dataclass, fields, asdict


@dataclass
class LinearParameters:
    batch_size: int = 1
    N_E: int = 10
    N_I: int = 100
    homeostasis: bool = True
    target_variance: float = 1.0
    k_I: float = 10.0
    T: float = 120000.0
    dt: float = 0.05
    tau_M: float = 50.0
    tau_W: float = 250.0
    tau_k: float = 500.0
    zeta: float = 1.0
    alpha: float = 1.0
    log_time: float = 20.0
    wandb_logging: bool = False
    random_seed: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float64
    tau_product: Optional[float] = None

    def __init__(self, **kwargs):
        field_defaults = {f.name: f.default for f in fields(self)}

        # Get the set of valid field names
        valid_fields = set(field_defaults.keys())

        # Filter kwargs to only include valid fields
        valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        # Initialize with filtered kwargs
        for k, v in valid_kwargs.items():
            setattr(self, k, v)

        self.__post_init__()

    def __post_init__(self):
        if self.tau_product is None:
            self.tau_product = self.tau_M * self.tau_W
        else:
            self.tau_M = self.tau_product / self.tau_W

    def to_dict(self):
        return asdict(self)
