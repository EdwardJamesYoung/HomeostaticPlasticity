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

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RectifiedQuadratic(ActivationFunction):
    @property
    def rectified(self) -> float:
        return 1.0

    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x) ** 2

    def inverse(self, r: float) -> float:
        return r**0.5

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.max(torch.zeros_like(x), x)


class RectifiedLinear(ActivationFunction):
    @property
    def rectified(self) -> float:
        return 1.0

    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x)

    def inverse(self, r: float) -> float:
        return r

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


class RectifiedPowerlaw1p5(ActivationFunction):
    @property
    def rectified(self) -> float:
        return 1.0

    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x) ** 1.5

    def inverse(self, r: float) -> float:
        return r ** (2 / 3)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return (
            1.5
            * torch.max(torch.zeros_like(x), x) ** 0.5
            * torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        )


class RectifiedCubic(ActivationFunction):
    @property
    def rectified(self) -> float:
        return 1.0

    def unsaturated_call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.zeros_like(x), x) ** 3

    def inverse(self, r: float) -> float:
        return r ** (1 / 3)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return (
            3
            * torch.max(torch.zeros_like(x), x) ** 2
            * torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        )
