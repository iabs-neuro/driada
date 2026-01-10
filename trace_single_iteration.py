"""Trace a single iteration to find exactly where time is spent."""
import numpy as np
import time
import sys

sys.path.insert(0, 'src')

from driada.information import TimeSeries, MultiTimeSeries
from driada.intense.intense_base import (
    get_fft_type, _extract_fft_data, _FFT_COMPUTE,
    FFT_CONTINUOUS, FFT_DISCRETE, FFT_MULTIVARIATE
)
from driada.information.info_base import compute_mi_batch_fft, compute_mi_mts_fft

np.random.seed(42)
n = 3600
nsh = 100
iters = 100

# Create test data
neuron = TimeSeries(np.random.randn(n), discrete=False)
position = MultiTimeSeries(np.vstack([np.random.randn(n), np.random.randn(n)]), discrete=False)

# Pre-cache
_ = neuron.copula_normal_data
_ = position.copula_normal_data

shifts = np.random.randint(0, n, size=nsh)

print("=" * 60)
print("Tracing MTS path (100 iterations):")
print("=" * 60)

# Time each step separately within the loop
t_fft_type = 0
t_extract = 0
t_dispatch = 0
t_compute = 0

for _ in range(iters):
    start = time.perf_counter()
    fft_type = get_fft_type(neuron, position, "mi", "gcmi", nsh, "auto")
    t_fft_type += time.perf_counter() - start

    start = time.perf_counter()
    data1, data2 = _extract_fft_data(neuron, position, fft_type, ds=1)
    t_extract += time.perf_counter() - start

    start = time.perf_counter()
    compute_fn = _FFT_COMPUTE[fft_type]
    t_dispatch += time.perf_counter() - start

    start = time.perf_counter()
    mi = compute_fn(data1, data2, shifts)
    t_compute += time.perf_counter() - start

print(f"  get_fft_type:     {t_fft_type*1000:.2f} ms")
print(f"  _extract_fft_data: {t_extract*1000:.2f} ms")
print(f"  dispatch lookup:   {t_dispatch*1000:.2f} ms")
print(f"  compute_fn:        {t_compute*1000:.2f} ms")
print(f"  TOTAL:             {(t_fft_type+t_extract+t_dispatch+t_compute)*1000:.2f} ms")

print("\n" + "=" * 60)
print("Direct call comparison (100 iterations):")
print("=" * 60)

# Pre-extract data once
d1 = neuron.copula_normal_data
d2 = position.copula_normal_data

t_compute2 = 0
for _ in range(iters):
    start = time.perf_counter()
    mi = compute_mi_mts_fft(d1, d2, shifts)
    t_compute2 += time.perf_counter() - start

print(f"  compute_mi_mts_fft: {t_compute2*1000:.2f} ms")

print("\n" + "=" * 60)
print("Comparison of extracted data:")
print("=" * 60)

fft_type = get_fft_type(neuron, position, "mi", "gcmi", nsh, "auto")
data1, data2 = _extract_fft_data(neuron, position, fft_type, ds=1)
d1 = neuron.copula_normal_data
d2 = position.copula_normal_data

print(f"  _extract_fft_data returns: data1.shape={data1.shape}, data2.shape={data2.shape}")
print(f"  Direct access returns:     d1.shape={d1.shape}, d2.shape={d2.shape}")
print(f"  data1 is d1: {data1 is d1}")
print(f"  data2 is d2: {data2 is d2}")
print(f"  data1 dtype: {data1.dtype}, d1 dtype: {d1.dtype}")
print(f"  data2 dtype: {data2.dtype}, d2 dtype: {d2.dtype}")
print(f"  data1 contiguous: {data1.flags['C_CONTIGUOUS']}, d1 contiguous: {d1.flags['C_CONTIGUOUS']}")
print(f"  data2 contiguous: {data2.flags['C_CONTIGUOUS']}, d2 contiguous: {d2.flags['C_CONTIGUOUS']}")

# Check if slicing creates a view or copy
print("\n  Testing array views:")
arr = np.random.randn(3600)
sliced = arr[::1]
print(f"  arr[::1] is view: {np.shares_memory(arr, sliced)}")
print(f"  arr[::1].flags['C_CONTIGUOUS']: {sliced.flags['C_CONTIGUOUS']}")

arr2d = np.random.randn(2, 3600)
sliced2d = arr2d[:, ::1]
print(f"  arr2d[:, ::1] is view: {np.shares_memory(arr2d, sliced2d)}")
print(f"  arr2d[:, ::1].flags['C_CONTIGUOUS']: {sliced2d.flags['C_CONTIGUOUS']}")
