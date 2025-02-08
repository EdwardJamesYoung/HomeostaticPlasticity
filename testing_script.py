import wandb
import torch
import importlib
from nonlinear_simulator import (
    SimulationParameters,
    generate_conditions,
    run_simulation,
)
from itertools import product
from typing import Dict, List, Optional
import gc

print("Hello world!")
