import torch
from typing import Optional


def expected_perplexity(
    dim: int, num_samples: int = 100000, seed: Optional[int] = None
) -> float:
    if seed is not None:
        torch.manual_seed(seed)

    z = torch.randn(num_samples, dim)  # iid N(0,1)
    g = z / z.norm(dim=1, keepdim=True)  # project to unit sphere
    p = g**2  # probability vector
    assert torch.allclose(
        torch.sum(p, dim=1), torch.ones(num_samples)
    ), "Probabilities must sum to 1"

    entropy = -torch.sum(
        p * torch.log(p + 1e-12), dim=1
    )  # natural‑log entropy per sample
    return torch.exp(entropy).mean().item()  # average perplexity
