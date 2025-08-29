#!/usr/bin/env python3

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
from scipy.optimize import minimize_scalar
import sys

# Import the necessary classes
from params import SimulationParameters
from input_generation import (
    CircularInputGenerator,
    TorusInputGenerator,
    DistributionConfig1D,
    DistributionConfig2D,
)


class PowerLawAnalyzer:
    """Analyze power law relationships between different neural response curves."""

    # Define curve aliases and their sources
    METRICS_CURVES = {
        "g_I": "curves/gains",
        "d_I": "curves/density",
        "w_I": "curves/widths",
    }

    GENERATOR_CURVES = {
        "p": "stimuli_probabilities",
        "hp": "convolved_probabilities",
        "d_E": "input_density",
        "g_E": "input_gains",
        "w_E": "input_widths",
        "q_E": "excitatory_third_factor",
        "q_I": "inhibitory_third_factor",
    }

    def __init__(self, results_dir: str):
        """Initialize analyzer for a specific results directory."""
        self.results_dir = Path("results") / results_dir
        self.output_dir = Path("power_law_analysis") / results_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

    def load_experiment_data(self, pt_file: Path) -> Dict[str, Any]:
        """Load experiment data from .pt file."""
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load {pt_file}: {e}")

    def reconstruct_input_generator(self, data: Dict[str, Any]):
        """Reconstruct InputGenerator from saved data."""
        # Extract parameters and configs
        parameters = SimulationParameters(**data["parameters"])
        parameters.device = torch.device("cpu")  # Force CPU for analysis

        # Reconstruct distribution configs
        distribution_configs = {}
        for config_name, config_data in data["distribution_configs"].items():
            if config_data["type"] == "1D":
                config = DistributionConfig1D(
                    mixing_parameter=config_data["mixing_parameter"],
                    concentration=config_data["concentration"],
                    location=config_data["location"],
                    device=torch.device("cpu"),  # Force CPU for analysis
                )
            else:  # 2D
                config = DistributionConfig2D(
                    mixing_parameter=config_data["mixing_parameter"],
                    concentration=config_data["concentration"],
                    location=config_data["location"],
                    device=torch.device("cpu"),
                )
            distribution_configs[config_name] = config

        # Determine generator type and create
        generator_type = data["input_generator_info"]["type"]
        if generator_type == "CircularInputGenerator":
            return CircularInputGenerator(parameters=parameters, **distribution_configs)
        elif generator_type == "TorusInputGenerator":
            return TorusInputGenerator(parameters=parameters, **distribution_configs)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

    def extract_curve(self, data: Dict[str, Any], curve_alias: str) -> torch.Tensor:
        """Extract specified curve from experiment data."""
        if curve_alias in self.METRICS_CURVES:
            # Extract from metrics_over_time (final time step)
            curve_key = self.METRICS_CURVES[curve_alias]
            if curve_key not in data["metrics_over_time"]:
                raise KeyError(f"Curve {curve_key} not found in metrics_over_time")

            curve_data = data["metrics_over_time"][
                curve_key
            ]  # [log_steps, repeats, batch, num_stimuli]
            return curve_data[-1]  # Take final time step: [repeats, batch, num_stimuli]

        elif curve_alias in self.GENERATOR_CURVES:
            # Extract from reconstructed InputGenerator
            generator = self.reconstruct_input_generator(data)
            curve_attr = self.GENERATOR_CURVES[curve_alias]

            if not hasattr(generator, curve_attr):
                raise AttributeError(f"Generator does not have attribute {curve_attr}")

            curve_data = getattr(generator, curve_attr)  # [batch, num_stimuli]

            # Expand to match metrics shape: [repeats, batch, num_stimuli]
            repeats = data["parameters"]["repeats"]
            curve_data = curve_data.unsqueeze(0).expand(repeats, -1, -1)
            return curve_data

        else:
            valid_curves = list(self.METRICS_CURVES.keys()) + list(
                self.GENERATOR_CURVES.keys()
            )
            raise ValueError(
                f"Unknown curve alias: {curve_alias}. Valid options: {valid_curves}"
            )

    def normalize_curve(self, curve: torch.Tensor) -> torch.Tensor:
        """Normalize curve to have mean 1 along the stimulus dimension."""
        # curve: [repeats, batch, num_stimuli]
        mean_vals = curve.mean(dim=-1, keepdim=True)  # [repeats, batch, 1]
        return curve / (mean_vals + 1e-12)  # Avoid division by zero

    def compute_l1_distance(
        self, curve1: torch.Tensor, curve2_powered: torch.Tensor
    ) -> torch.Tensor:
        """Compute L1 distance between curves."""
        # Both curves: [repeats, batch, num_stimuli]
        diff = torch.abs(curve1 - curve2_powered)
        return diff.mean(dim=-1)  # [repeats, batch]

    def optimize_gamma_single(
        self, curve1_norm: torch.Tensor, curve2_norm: torch.Tensor
    ) -> Tuple[float, float]:
        """Optimize gamma for a single [repeat, batch] combination."""
        curve1_np = curve1_norm.detach().cpu().numpy()
        curve2_np = curve2_norm.detach().cpu().numpy()

        def objective(gamma):
            curve2_powered = np.power(curve2_np, gamma)
            curve2_powered_norm = curve2_powered / (curve2_powered.mean() + 1e-12)
            l1_dist = np.mean(np.abs(curve1_np - curve2_powered_norm)) / 2
            return l1_dist

        # Multi-start optimization with different initial points
        gamma_starts = np.linspace(-3, 5, 20)  # 20 starting points
        best_gamma = None
        best_l1 = float("inf")

        for gamma_start in gamma_starts:
            try:
                result = minimize_scalar(objective, bounds=(-3, 100), method="bounded")

                if result.fun < best_l1:
                    best_l1 = result.fun
                    best_gamma = result.x

            except Exception:
                continue

        return best_gamma if best_gamma is not None else 0.0, best_l1

    def fit_power_law(
        self, curve1: torch.Tensor, curve2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit power law relationship between two curves."""
        # Normalize curves
        curve1_norm = self.normalize_curve(curve1)  # [repeats, batch, num_stimuli]
        curve2_norm = self.normalize_curve(curve2)  # [repeats, batch, num_stimuli]

        repeats, batch_size, num_stimuli = curve1_norm.shape

        # Initialize output tensors
        gammas = torch.zeros(repeats, batch_size, dtype=torch.float32)
        l1_distances = torch.zeros(repeats, batch_size, dtype=torch.float32)

        # Optimize for each [repeat, batch] combination
        for r in range(repeats):
            for b in range(batch_size):
                gamma, l1_dist = self.optimize_gamma_single(
                    curve1_norm[r, b], curve2_norm[r, b]
                )
                gammas[r, b] = gamma
                l1_distances[r, b] = l1_dist

        return gammas, l1_distances

    def parse_curve_pairs(self, pairs_args: list) -> list:
        """Parse curve pairs from command line arguments."""
        pairs = []
        for pair_str in pairs_args:
            # Handle both "curve1,curve2" and quoted pairs
            if "," in pair_str:
                parts = pair_str.split(",")
                if len(parts) == 2:
                    curve1, curve2 = parts[0].strip(), parts[1].strip()
                    # Validate both curves exist
                    all_curves = list(self.METRICS_CURVES.keys()) + list(
                        self.GENERATOR_CURVES.keys()
                    )
                    if curve1 in all_curves and curve2 in all_curves:
                        pairs.append((curve1, curve2))
                    else:
                        invalid_curves = [
                            c for c in [curve1, curve2] if c not in all_curves
                        ]
                        print(
                            f"  Warning: Skipping invalid curve pair '{pair_str}' - unknown curves: {invalid_curves}"
                        )
                else:
                    print(
                        f"  Warning: Skipping malformed pair '{pair_str}' - expected format 'curve1,curve2'"
                    )
            else:
                print(
                    f"  Warning: Skipping malformed pair '{pair_str}' - missing comma"
                )
        return pairs

    def analyze_file(self, pt_file: Path, curve_pairs: list) -> Dict[str, Any]:
        """Analyze a single experiment file for multiple curve pairs."""
        print(f"Processing {pt_file.name}...")

        # Load existing results if they exist
        output_file = self.output_dir / f"{pt_file.stem}_analysis.pt"
        if output_file.exists():
            try:
                existing_results = torch.load(
                    output_file, map_location="cpu", weights_only=False
                )
                print(f"  Found existing analysis file, will merge results")
            except Exception as e:
                print(f"  Warning: Could not load existing analysis file: {e}")
                existing_results = {}
        else:
            existing_results = {}

        # Load experiment data once
        data = self.load_experiment_data(pt_file)

        # Process each curve pair
        for curve1_alias, curve2_alias in curve_pairs:
            pair_key = f"{curve1_alias}_vs_{curve2_alias}"
            try:
                # Extract curves
                curve1 = self.extract_curve(data, curve1_alias).to(device="cpu")
                curve2 = self.extract_curve(data, curve2_alias).to(device="cpu")

                print(f"  Processing {pair_key}: shapes {curve1.shape}, {curve2.shape}")

                # Fit power law
                gammas, l1_distances = self.fit_power_law(curve1, curve2)

                # Store results for this pair
                existing_results[pair_key] = {
                    "gammas": gammas,
                    "l1_distances": l1_distances,
                    "curve1_alias": curve1_alias,
                    "curve2_alias": curve2_alias,
                }

                print(f"    Gamma range: [{gammas.min():.3f}, {gammas.max():.3f}]")
                print(
                    f"    L1 distance range: [{l1_distances.min():.3f}, {l1_distances.max():.3f}]"
                )

            except Exception as e:
                print(f"  Skipping {pair_key}: {e}")
                continue

        # Add/update metadata
        existing_results["original_file"] = pt_file.name

        return existing_results

    def run_analysis(self, curve_pairs: list):
        """Run power law analysis on all files in results directory for multiple curve pairs."""
        if not curve_pairs:
            print("No valid curve pairs to analyze!")
            return

        pair_names = [f"{c1}_vs_{c2}" for c1, c2 in curve_pairs]
        print(f"Running power law analysis for pairs: {pair_names}")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")

        # Find all .pt files
        pt_files = list(self.results_dir.glob("*.pt"))
        if not pt_files:
            print("No .pt files found in results directory!")
            return

        print(f"Found {len(pt_files)} experiment files to process")

        # Process each file
        for pt_file in pt_files:
            try:
                results = self.analyze_file(pt_file, curve_pairs)

                # Save results
                output_file = self.output_dir / f"{pt_file.stem}_analysis.pt"
                torch.save(results, output_file)

                # Show summary
                pair_count = len([k for k in results.keys() if "_vs_" in k])
                print(f"  Saved {pair_count} curve pair results to {output_file}")

            except Exception as e:
                print(f"  Failed to process {pt_file.name}: {e}")
                continue

        print("Analysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze power law relationships between neural response curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python power_law_analysis.py --results_dir density_gain_power_law --pairs d_I,p g_I,p
  python power_law_analysis.py --results_dir my_experiment --pairs "w_I,d_E" "g_I,q_E"

Available curve aliases:
  From simulation metrics: g_I, d_I, w_I
  From input generator: p, d_E, g_E, w_E, q_E, q_I
        """,
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Results directory name (e.g., 'density_gain_power_law')",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Curve pairs in format 'curve1,curve2' (e.g., 'd_I,p g_I,p')",
    )

    args = parser.parse_args()

    # Run analysis
    try:
        analyzer = PowerLawAnalyzer(args.results_dir)
        curve_pairs = analyzer.parse_curve_pairs(args.pairs)

        if not curve_pairs:
            print(
                "No valid curve pairs found. Check your input format and curve names."
            )
            print("\nAvailable curves:")
            print("  From simulation: g_I, d_I, w_I")
            print("  From generator: p, d_E, g_E, w_E, q_E, q_I")
            sys.exit(1)

        analyzer.run_analysis(curve_pairs)

    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
