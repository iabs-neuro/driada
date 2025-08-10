#!/usr/bin/env python
"""Debug SNR calculation."""

import numpy as np
from driada.experiment.synthetic.mixed_selectivity import generate_synthetic_exp_with_mixed_selectivity

# Generate weak params experiment
exp_weak, _ = generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=2,
    n_continuous_feats=0,
    n_neurons=10,
    duration=100,
    fps=10,
    selectivity_prob=0.3,   # Low selectivity
    rate_0=0.5,            # High baseline
    rate_1=2.0,            # Active rate (reasonable for calcium)
    noise_std=0.2,         # Moderate noise (was 0.5)
    seed=42,
    verbose=False
)

# Generate strong params experiment
exp_strong, _ = generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=2,
    n_continuous_feats=0,
    n_neurons=10,
    duration=100,
    fps=10,
    selectivity_prob=1.0,   # High selectivity
    rate_0=0.05,           # Low baseline
    rate_1=5.0,            # High active rate
    noise_std=0.05,        # Low noise
    seed=42,
    verbose=False
)

def analyze_exp(exp, name):
    """Analyze signal properties."""
    print(f"\n{name} parameters:")
    for i in range(min(3, exp.n_cells)):
        signal = exp.neurons[i].ca.data
        print(f"  Neuron {i}:")
        print(f"    Min: {np.min(signal):.3f}")
        print(f"    25th percentile: {np.percentile(signal, 25):.3f}")
        print(f"    Median: {np.median(signal):.3f}")
        print(f"    75th percentile: {np.percentile(signal, 75):.3f}")
        print(f"    90th percentile: {np.percentile(signal, 90):.3f}")
        print(f"    Max: {np.max(signal):.3f}")
        print(f"    Std: {np.std(signal):.3f}")
        # Check if mostly noise
        below_median = signal[signal < np.median(signal)]
        print(f"    Std below median: {np.std(below_median):.3f}")

analyze_exp(exp_weak, "Weak")
analyze_exp(exp_strong, "Strong")

# Test SNR calculation
def compute_snr(exp):
    """Compute average signal-to-noise ratio."""
    snrs = []
    for i in range(exp.n_cells):
        signal = exp.neurons[i].ca.data
        # Use robust statistics
        baseline = np.percentile(signal, 25)
        peaks = np.percentile(signal, 90)
        # Use standard deviation as noise measure to avoid division by near-zero
        noise_std = np.std(signal[signal < np.median(signal)])
        if noise_std > 0:
            snrs.append((peaks - baseline) / noise_std)
        print(f"Neuron {i}: baseline={baseline:.3f}, peaks={peaks:.3f}, noise_std={noise_std:.6f}, SNR={(peaks - baseline) / noise_std:.2f}")
    return np.mean(snrs) if snrs else 0

print("\nWeak SNR calculation:")
snr_weak = compute_snr(exp_weak)
print(f"Average SNR: {snr_weak:.2f}")

print("\nStrong SNR calculation:")
snr_strong = compute_snr(exp_strong)
print(f"Average SNR: {snr_strong:.2f}")