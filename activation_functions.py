import torch
from abc import ABC, abstractmethod

SATURATION_VALUE = 100.0


class ActivationFunction(ABC):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.unsaturated_call(x).clip(
            max=SATURATION_VALUE, min=-SATURATION_VALUE
        )

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def rectified(self) -> float:
        pass

    @abstractmethod
    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self, r: float) -> float:
        pass


class RectifiedQuadratic(ActivationFunction):
    @property
    def rectified(self) -> float:
        return 1.0

    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x) ** 2

    def inverse(self, r: float) -> float:
        return r**0.5


class RectifiedLinear(ActivationFunction):
    @property
    def rectified(self) -> float:
        return 1.0

    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x)

    def inverse(self, r: float) -> float:
        return r


class RectifiedCubic(ActivationFunction):
    @property
    def rectified(self) -> float:
        return 1.0

    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x) ** 3

    def inverse(self, r: float) -> float:
        return r ** (1 / 3)


class Cubic(ActivationFunction):
    @property
    def rectified(self) -> float:
        return 0.0

    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return x**3

    def inverse(self, r: float) -> float:
        return r ** (1 / 3)


class Linear(ActivationFunction):
    @property
    def rectified(self) -> float:
        return 0.0

    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, r: float) -> float:
        return r
