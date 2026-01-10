"""Time individual FFT paths WITH PROPER WARMUP."""
import numpy as np
import time
import sys

sys.path.insert(0, 'src')

from driada.information import TimeSeries, MultiTimeSeries
from driada.intense.intense_base import (
    get_fft_type, _extract_fft_data, _FFT_COMPUTE,
    FFT_CONTINUOUS, FFT_DISCRETE, FFT_MULTIVARIATE
)
from driada.information.info_base import compute_mi_batch_fft, compute_mi_gd_fft, compute_mi_mts_fft

np.random.seed(42)
n = 3600
nsh = 100

# Create test data
print("Creating test data...")
neuron = TimeSeries(np.random.randn(n), discrete=False)
hd = TimeSeries(np.random.randn(n), discrete=False)
event = TimeSeries(np.random.randint(0, 5, n), discrete=True)
position = MultiTimeSeries(np.vstack([np.random.randn(n), np.random.randn(n)]), discrete=False)

# Force copula normalization
_ = neuron.copula_normal_data
_ = hd.copula_normal_data
_ = position.copula_normal_data
_ = event.int_data

shifts = np.random.randint(0, n, size=nsh)

print("Warming up JIT/imports...")
# Warmup ALL paths
for _ in range(5):
    # CC warmup
    fft_type = get_fft_type(neuron, hd, "mi", "gcmi", nsh, "auto")
    data1, data2 = _extract_fft_data(neuron, hd, fft_type, ds=1)
    compute_fn = _FFT_COMPUTE[fft_type]
    mi = compute_fn(data1, data2, shifts)

    ny1 = neuron.copula_normal_data
    ny2 = hd.copula_normal_data
    mi = compute_mi_batch_fft(ny1, ny2, shifts)

    # GD warmup
    fft_type = get_fft_type(neuron, event, "mi", "gcmi", nsh, "auto")
    data1, data2 = _extract_fft_data(neuron, event, fft_type, ds=1)
    compute_fn = _FFT_COMPUTE[fft_type]
    mi = compute_fn(data1, data2, shifts)

    mi = compute_mi_gd_fft(neuron.copula_normal_data, event.int_data, shifts)

    # MTS warmup
    fft_type = get_fft_type(neuron, position, "mi", "gcmi", nsh, "auto")
    data1, data2 = _extract_fft_data(neuron, position, fft_type, ds=1)
    compute_fn = _FFT_COMPUTE[fft_type]
    mi = compute_fn(data1, data2, shifts)

    mi = compute_mi_mts_fft(neuron.copula_normal_data, position.copula_normal_data, shifts)

print(f"\nData length: {n}, shuffles: {nsh}")
print("=" * 60)

# Test 1: Continuous-Continuous
print("\n[1] Continuous-Continuous (neuron vs hd)")

start = time.time()
for _ in range(100):
    fft_type = get_fft_type(neuron, hd, "mi", "gcmi", nsh, "auto")
    data1, data2 = _extract_fft_data(neuron, hd, fft_type, ds=1)
    compute_fn = _FFT_COMPUTE[fft_type]
    mi = compute_fn(data1, data2, shifts)
new_time = time.time() - start
print(f"  New path (100 iters): {new_time*1000:.1f} ms")

start = time.time()
for _ in range(100):
    ny1 = neuron.copula_normal_data
    ny2 = hd.copula_normal_data
    mi = compute_mi_batch_fft(ny1, ny2, shifts)
old_time = time.time() - start
print(f"  Direct call (100 iters): {old_time*1000:.1f} ms")
print(f"  Overhead: {(new_time - old_time)*1000:.1f} ms ({(new_time/old_time - 1)*100:.1f}%)")

# Test 2: Discrete-Continuous
print("\n[2] Discrete-Continuous (neuron vs event)")
start = time.time()
for _ in range(100):
    fft_type = get_fft_type(neuron, event, "mi", "gcmi", nsh, "auto")
    data1, data2 = _extract_fft_data(neuron, event, fft_type, ds=1)
    compute_fn = _FFT_COMPUTE[fft_type]
    mi = compute_fn(data1, data2, shifts)
new_time = time.time() - start
print(f"  New path (100 iters): {new_time*1000:.1f} ms")

start = time.time()
for _ in range(100):
    continuous_data = neuron.copula_normal_data
    discrete_data = event.int_data
    mi = compute_mi_gd_fft(continuous_data, discrete_data, shifts)
old_time = time.time() - start
print(f"  Direct call (100 iters): {old_time*1000:.1f} ms")
print(f"  Overhead: {(new_time - old_time)*1000:.1f} ms ({(new_time/old_time - 1)*100:.1f}%)")

# Test 3: MultiTimeSeries
print("\n[3] MultiTimeSeries (neuron vs position)")
start = time.time()
for _ in range(100):
    fft_type = get_fft_type(neuron, position, "mi", "gcmi", nsh, "auto")
    data1, data2 = _extract_fft_data(neuron, position, fft_type, ds=1)
    compute_fn = _FFT_COMPUTE[fft_type]
    mi = compute_fn(data1, data2, shifts)
new_time = time.time() - start
print(f"  New path (100 iters): {new_time*1000:.1f} ms")

start = time.time()
for _ in range(100):
    ts_data = neuron.copula_normal_data
    mts_data = position.copula_normal_data
    mi = compute_mi_mts_fft(ts_data, mts_data, shifts)
old_time = time.time() - start
print(f"  Direct call (100 iters): {old_time*1000:.1f} ms")
print(f"  Overhead: {(new_time - old_time)*1000:.1f} ms ({(new_time/old_time - 1)*100:.1f}%)")

print("\n" + "=" * 60)
print("Summary: After warmup, overhead should be minimal.")
