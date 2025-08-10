#!/usr/bin/env python
"""Debug mixed selective signal generation."""

import numpy as np
from driada.experiment.synthetic.mixed_selectivity import generate_mixed_selective_signal

# Test parameters from failing test
duration = 10
fps = 20
n_points = int(duration * fps)

# Create test features - binary
feat1 = np.zeros(n_points)
feat1[50:70] = 1  # Active period 1
feat1[120:140] = 1  # Active period 2

feat2 = np.zeros(n_points)
feat2[60:80] = 1  # Overlaps with feat1  
feat2[150:170] = 1  # Separate period

features = [feat1, feat2]
weights = [0.6, 0.4]

signal = generate_mixed_selective_signal(
    features, weights, duration, fps,
    rate_0=0.1, rate_1=3.0,
    skip_prob=0.0,
    noise_std=0.05,
    seed=42
)

# Check signal values
both_active = signal[60:70]  # Both features active
neither_active = signal[90:110]  # Neither active

print(f"Signal when both active: mean={np.mean(both_active):.3f}")
print(f"Signal when neither active: mean={np.mean(neither_active):.3f}")
print(f"Expected: both_active > neither_active")
print(f"Actual: {np.mean(both_active)} > {np.mean(neither_active)} = {np.mean(both_active) > np.mean(neither_active)}")

# Check what's happening with the features
print("\nFeature 1 active periods:", np.where(feat1 > 0)[0])
print("Feature 2 active periods:", np.where(feat2 > 0)[0])

# Check calcium signal during different periods
print(f"\nSignal during feat1 only (50:60): mean={np.mean(signal[50:60]):.3f}")
print(f"Signal during feat2 only (150:160): mean={np.mean(signal[150:160]):.3f}")
print(f"Signal during overlap (60:70): mean={np.mean(signal[60:70]):.3f}")