"""Diagnose FFT overhead by timing each component individually."""
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
iters = 1000

# Create test data
print("Creating test data...")
neuron = TimeSeries(np.random.randn(n), discrete=False)
hd = TimeSeries(np.random.randn(n), discrete=False)
position = MultiTimeSeries(np.vstack([np.random.randn(n), np.random.randn(n)]), discrete=False)

# Pre-cache copula normal data
_ = neuron.copula_normal_data
_ = hd.copula_normal_data
_ = position.copula_normal_data

shifts = np.random.randint(0, n, size=nsh)

print(f"Testing with {iters} iterations each\n")
print("=" * 60)

# Test 1: get_fft_type() overhead
print("\n[1] get_fft_type() overhead:")
start = time.time()
for _ in range(iters):
    fft_type = get_fft_type(neuron, hd, "mi", "gcmi", nsh, "auto")
t1 = time.time() - start
print(f"  CC pair: {t1*1000:.1f} ms")

start = time.time()
for _ in range(iters):
    fft_type = get_fft_type(neuron, position, "mi", "gcmi", nsh, "auto")
t2 = time.time() - start
print(f"  MTS pair: {t2*1000:.1f} ms")

# Test 2: _extract_fft_data() overhead
print("\n[2] _extract_fft_data() overhead:")
start = time.time()
for _ in range(iters):
    data1, data2 = _extract_fft_data(neuron, hd, FFT_CONTINUOUS, ds=1)
t3 = time.time() - start
print(f"  CC pair: {t3*1000:.1f} ms")

start = time.time()
for _ in range(iters):
    data1, data2 = _extract_fft_data(neuron, position, FFT_MULTIVARIATE, ds=1)
t4 = time.time() - start
print(f"  MTS pair: {t4*1000:.1f} ms")

# Test 3: Direct property access
print("\n[3] Direct property access:")
start = time.time()
for _ in range(iters):
    d1 = neuron.copula_normal_data
    d2 = hd.copula_normal_data
t5 = time.time() - start
print(f"  CC pair: {t5*1000:.1f} ms")

start = time.time()
for _ in range(iters):
    d1 = neuron.copula_normal_data
    d2 = position.copula_normal_data
t6 = time.time() - start
print(f"  MTS pair: {t6*1000:.1f} ms")

# Test 4: isinstance() overhead
print("\n[4] isinstance() checks:")
start = time.time()
for _ in range(iters):
    x = isinstance(neuron, MultiTimeSeries)
    y = isinstance(hd, TimeSeries) and not isinstance(hd, MultiTimeSeries)
t7 = time.time() - start
print(f"  Basic isinstance: {t7*1000:.1f} ms")

start = time.time()
for _ in range(iters):
    x = isinstance(position, MultiTimeSeries)
    y = isinstance(neuron, TimeSeries) and not isinstance(neuron, MultiTimeSeries)
t8 = time.time() - start
print(f"  MTS isinstance: {t8*1000:.1f} ms")

# Test 5: dispatch table vs direct call
print("\n[5] Dispatch table lookup:")
start = time.time()
for _ in range(iters):
    compute_fn = _FFT_COMPUTE[FFT_CONTINUOUS]
t9 = time.time() - start
print(f"  Dict lookup: {t9*1000:.1f} ms")

# Test 6: Full path comparison for CC
print("\n[6] Full path comparison (CC):")
d1 = neuron.copula_normal_data
d2 = hd.copula_normal_data

start = time.time()
for _ in range(100):
    fft_type = get_fft_type(neuron, hd, "mi", "gcmi", nsh, "auto")
    data1, data2 = _extract_fft_data(neuron, hd, fft_type, ds=1)
    compute_fn = _FFT_COMPUTE[fft_type]
    mi = compute_fn(data1, data2, shifts)
t10 = time.time() - start
print(f"  New path (100x): {t10*1000:.1f} ms")

start = time.time()
for _ in range(100):
    mi = compute_mi_batch_fft(d1, d2, shifts)
t11 = time.time() - start
print(f"  Pre-extracted (100x): {t11*1000:.1f} ms")

start = time.time()
for _ in range(100):
    ny1 = neuron.copula_normal_data
    ny2 = hd.copula_normal_data
    mi = compute_mi_batch_fft(ny1, ny2, shifts)
t12 = time.time() - start
print(f"  Direct call (100x): {t12*1000:.1f} ms")

# Test 7: Full path comparison for MTS
print("\n[7] Full path comparison (MTS):")
d1 = neuron.copula_normal_data
d2 = position.copula_normal_data

start = time.time()
for _ in range(100):
    fft_type = get_fft_type(neuron, position, "mi", "gcmi", nsh, "auto")
    data1, data2 = _extract_fft_data(neuron, position, fft_type, ds=1)
    compute_fn = _FFT_COMPUTE[fft_type]
    mi = compute_fn(data1, data2, shifts)
t13 = time.time() - start
print(f"  New path (100x): {t13*1000:.1f} ms")

start = time.time()
for _ in range(100):
    mi = compute_mi_mts_fft(d1, d2, shifts)
t14 = time.time() - start
print(f"  Pre-extracted (100x): {t14*1000:.1f} ms")

start = time.time()
for _ in range(100):
    ny1 = neuron.copula_normal_data
    ny2 = position.copula_normal_data
    mi = compute_mi_mts_fft(ny1, ny2, shifts)
t15 = time.time() - start
print(f"  Direct call (100x): {t15*1000:.1f} ms")

print("\n" + "=" * 60)
print("Summary:")
print(f"  get_fft_type overhead: CC={t1:.1f}ms MTS={t2:.1f}ms for {iters} calls")
print(f"  _extract_fft_data overhead: CC={t3:.1f}ms MTS={t4:.1f}ms for {iters} calls")
