"""Trace overhead in calculate_optimal_delays."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import time
import os

# Load cached experiment
from driada.experiment import load_exp_from_pickle
exp = load_exp_from_pickle("benchmark_exp_cache.pickle", verbose=False)

from driada.information import TimeSeries, MultiTimeSeries
from driada.intense.intense_base import (
    get_fft_type, _extract_fft_data, _FFT_COMPUTE,
    FFT_CONTINUOUS, FFT_DISCRETE, FFT_MULTIVARIATE
)

# Get neurons and features
neurons = [exp.neurons[i].ca for i in range(exp.n_cells)]
feat_bunch = [f for f in exp.dynamic_features.keys() if f not in ['x', 'y']]
skip_delays = ['position_2d']
features = [exp.dynamic_features[f] for f in feat_bunch if f not in skip_delays]
feature_names = [f for f in feat_bunch if f not in skip_delays]

print("=" * 70)
print("DELAY OVERHEAD TRACING")
print("=" * 70)
print(f"Neurons: {len(neurons)}")
print(f"Features for delays: {len(features)}")
print(f"Feature names: {feature_names}")

ds = 5
shift_window = 40
shifts = np.arange(-shift_window, shift_window + ds, ds) // ds
print(f"Shifts: {len(shifts)}")

# Measure different components
times = {
    'get_fft_type': 0,
    'extract_fft_data': 0,
    'compute_fn': 0,
    'argmax': 0,
    'loop_overhead': 0,
}

# First, warm up
print("\nWarming up...")
for _ in range(3):
    for ts2 in features[:2]:
        ts1 = neurons[0]
        fft_type = get_fft_type(ts1, ts2, "mi", "gcmi", len(shifts), "auto", for_delays=True)
        if fft_type:
            data1, data2 = _extract_fft_data(ts1, ts2, fft_type, ds)
            compute_fn = _FFT_COMPUTE[fft_type]
            n = len(data1) if data1.ndim == 1 else data1.shape[-1]
            fft_shifts = np.where(shifts >= 0, shifts, n + shifts).astype(int)
            mi_values = compute_fn(data1, data2, fft_shifts)

# Measure with a subset (first 50 neurons, all features)
n_test_neurons = 50
test_neurons = neurons[:n_test_neurons]

print(f"\nTiming {n_test_neurons} neurons x {len(features)} features = {n_test_neurons * len(features)} pairs")

# Pre-compute FFT types
fft_types = [
    get_fft_type(test_neurons[0], ts2, "mi", "gcmi", len(shifts), "auto", for_delays=True)
    for ts2 in features
]
print(f"FFT types: {fft_types}")

total_start = time.perf_counter()
pair_count = 0

for i, ts1 in enumerate(test_neurons):
    for j, ts2 in enumerate(features):
        fft_type = fft_types[j]

        if fft_type is not None:
            t0 = time.perf_counter()
            data1, data2 = _extract_fft_data(ts1, ts2, fft_type, ds)
            times['extract_fft_data'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            compute_fn = _FFT_COMPUTE[fft_type]
            n = len(data1) if data1.ndim == 1 else data1.shape[-1]
            fft_shifts = np.where(shifts >= 0, shifts, n + shifts).astype(int)
            times['loop_overhead'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            mi_values = compute_fn(data1, data2, fft_shifts)
            times['compute_fn'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            best_idx = np.argmax(mi_values)
            times['argmax'] += time.perf_counter() - t0

        pair_count += 1

total_elapsed = time.perf_counter() - total_start
times['other'] = total_elapsed - sum(times.values())

print(f"\nTotal time: {total_elapsed*1000:.1f}ms for {pair_count} pairs")
print(f"Time per pair: {total_elapsed/pair_count*1000:.3f}ms")

print("\nBreakdown:")
for name, t in sorted(times.items(), key=lambda x: -x[1]):
    pct = t / total_elapsed * 100 if total_elapsed > 0 else 0
    print(f"  {name:20s}: {t*1000:8.1f}ms ({pct:5.1f}%)")

# Project to full dataset
full_pairs = 500 * 8  # 500 neurons, 8 features with delays
scale = full_pairs / pair_count
projected_time = total_elapsed * scale
print(f"\nProjected time for {full_pairs} pairs: {projected_time:.1f}s")
print(f"Observed in benchmark: 25.6s")
print(f"Ratio: {25.6 / projected_time:.2f}x (parallel overhead)")
