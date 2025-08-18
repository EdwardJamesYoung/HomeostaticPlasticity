import torch
import wandb
from jaxtyping import Float, jaxtyped
from typeguard import typechecked

from params import LinearParameters


@jaxtyped(typechecker=typechecked)
def log_variables(
    W: Float[torch.Tensor, "batch N_I N_E"],
    dW: Float[torch.Tensor, "batch N_I N_E"],
    M: Float[torch.Tensor, "batch N_I N_I"],
    dM: Float[torch.Tensor, "batch N_I N_I"],
    k_E: Float[torch.Tensor, "batch N_I"],
    dk_E: Float[torch.Tensor, "batch N_I"],
    parameters: LinearParameters,
    iteration_step: int,
) -> dict[str, float]:
    dt = parameters.dt
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Calculate metrics per batch item
    recurrent_update_magnitude = torch.mean(torch.abs(dM), dim=(-2, -1)) / dt
    recurrent_percentage_change = recurrent_update_magnitude / (
        torch.mean(M, dim=(-2, -1)) + 1e-12
    )

    feedforward_update_magnitude = torch.mean(torch.abs(dW), dim=(-2, -1)) / dt
    feedforward_percentage_change = feedforward_update_magnitude / (
        torch.mean(torch.abs(W), dim=(-2, -1)) + 1e-12
    )

    excitatory_mass_update_magnitude = torch.mean(torch.abs(dk_E), dim=-1) / dt
    average_excitatory_mass = torch.mean(k_E, dim=-1)
    excitatory_mass_percentage_change = excitatory_mass_update_magnitude / (
        average_excitatory_mass + 1e-12
    )

    metrics_to_log = {
        "recurrent_update_magnitude": recurrent_update_magnitude,
        "recurrent_percentage_change": recurrent_percentage_change,
        "feedforward_update_magnitude": feedforward_update_magnitude,
        "feedforward_percentage_change": feedforward_percentage_change,
        "excitatory_mass_update_magnitude": excitatory_mass_update_magnitude,
        "average_excitatory_mass": average_excitatory_mass,
        "excitatory_mass_percentage_change": excitatory_mass_percentage_change,
    }

    log_dict = {}
    for name, metric_tensor in metrics_to_log.items():
        for q in quantiles:
            q_value = torch.quantile(metric_tensor, q).item()
            key = f"dynamics/{name}_q{int(q*100)}"
            log_dict[key] = q_value

    log_dict["time"] = dt * iteration_step

    return log_dict


def spectrum_differences(
    X: Float[torch.Tensor, "batch N_I N_E"],
    spectrum: Float[torch.Tensor, "batch N_E"],
    basis: Float[torch.Tensor, "batch N_E N_E"],
) -> dict[str, float]:

    N_I = X.shape[-2]

    # Configuration
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    rotated_response = torch.einsum("bij,bjk->bik", X, basis)  # [batch, N_I, N_E]

    # Compute the allocation of neurons to modes
    rot_squared = rotated_response**2  # [batch, N_I, N_E]
    # Normalise over the input modes
    mode_allocations = rot_squared / torch.sum(
        rot_squared, dim=-1, keepdim=True
    )  # [batch, N_I, N_E]

    # Compute the neuron allocations to modes
    neuron_allocations = torch.mean(mode_allocations, dim=1)  # [batch, N_E]
    normalised_allocations = neuron_allocations / torch.sum(
        neuron_allocations, dim=-1, keepdim=True
    )  # [batch, N_E]

    # Entropies of the mean allocations (keeping this separate as it doesn't fit the TV distance pattern)
    neuron_entropies = -torch.sum(
        mode_allocations * torch.log(mode_allocations + 1e-12), dim=-1
    )  # [batch, N_I]
    neuron_perplexities = torch.exp(neuron_entropies)  # [batch, N_I]
    average_neuron_perplexities = torch.mean(neuron_perplexities, dim=-1)  # [batch]

    mode_variances = torch.einsum("bj,bij->bj", spectrum, rot_squared)  # [batch, N_E]
    total_variance = torch.sum(mode_variances, dim=-1)  # [batch]
    average_variance = total_variance / N_I  # [batch]

    # Check that the mode variances are all non-negative
    assert torch.all(
        mode_variances >= 0
    ), f"Mode variances contain negative values: {mode_variances}"

    # Normalize all distributions
    normalised_variances = mode_variances / torch.sum(
        mode_variances, dim=-1, keepdim=True
    )  # [batch, N_E]

    normalised_spectrum = spectrum / torch.sum(
        spectrum, dim=-1, keepdim=True
    )  # [batch, N_E]

    uniform_distribution = (
        torch.ones_like(normalised_variances) / normalised_variances.shape[-1]
    )  # [batch, N_E]

    # Create one-hot vector (1 for max value, 0 elsewhere)
    one_hot = torch.zeros_like(normalised_variances)
    one_hot.scatter_(-1, torch.argmax(normalised_variances, dim=-1, keepdim=True), 1)

    # Define all distributions for pairwise comparison
    distributions = {
        "variances": normalised_variances,
        "spectrum": normalised_spectrum,
        "allocations": normalised_allocations,
        "uniform": uniform_distribution,
        "one_hot": one_hot,
    }

    # Compute TV distances for all pairs and extract quantiles
    results = {}

    from itertools import combinations

    for name1, name2 in combinations(distributions.keys(), 2):
        dist1, dist2 = distributions[name1], distributions[name2]

        # Compute TV distance: 0.5 * sum of absolute differences
        tv_distance = 0.5 * torch.sum(torch.abs(dist1 - dist2), dim=-1)  # [batch]

        # Extract quantiles
        for q in quantiles:
            q_value = torch.quantile(tv_distance, q).item()
            key = f"mode_diff/{name1}_vs_{name2}_q{int(q*100)}"
            results[key] = q_value

    # Add neuron perplexities (separate pattern)
    for q in quantiles:
        q_value = torch.quantile(average_neuron_perplexities, q).item()
        key = f"statistics/neuron_perplexities_q{int(q*100)}"
        results[key] = q_value

    # Add average variance
    for q in quantiles:
        q_value = torch.quantile(average_variance, q).item()
        key = f"statistics/average_variance_q{int(q*100)}"
        results[key] = q_value

    # Add total variance
    for q in quantiles:
        q_value = torch.quantile(total_variance, q).item()
        key = f"statistics/total_variance_q{int(q*100)}"
        results[key] = q_value

    return results
