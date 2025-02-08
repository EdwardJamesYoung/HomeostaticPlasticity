import torch
from abc import ABC, abstractmethod

SATURATION_VALUE = 100.0


class NonLinearity(ABC):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.unsaturated_call(x).clamp(max=SATURATION_VALUE)

    @abstractmethod
    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, r: float) -> float:
        pass


class RectifiedQuadratic(NonLinearity):
    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x) ** 2

    def inverse(self, r: float) -> float:
        return r**0.5


class RectifiedLinear(NonLinearity):
    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x)

    def inverse(self, r: float) -> float:
        return r


class RectifiedCubic(NonLinearity):
    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x) ** 3

    def inverse(self, r: float) -> float:
        return r ** (1 / 3)
