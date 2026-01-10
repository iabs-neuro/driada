"""Quick diagnostic for FFT path selection."""
import sys
sys.path.insert(0, 'src')

from driada.information import TimeSeries, MultiTimeSeries
from driada.intense.intense_base import get_fft_type, FFT_CONTINUOUS, FFT_DISCRETE, FFT_MULTIVARIATE
import numpy as np

np.random.seed(42)
n = 1000

# Create test time series
neuron = TimeSeries(np.random.randn(n), discrete=False)
hd = TimeSeries(np.random.randn(n), discrete=False)
event = TimeSeries(np.random.randint(0, 3, n), discrete=True)
position = MultiTimeSeries(np.vstack([np.random.randn(n), np.random.randn(n)]), discrete=False)

print("FFT Type Detection:")
print("=" * 50)

# Test various combinations
pairs = [
    ("neuron vs hd (CC)", neuron, hd),
    ("neuron vs event (GD)", neuron, event),
    ("neuron vs position (MTS)", neuron, position),
]

for name, ts1, ts2 in pairs:
    for engine in ["auto", "fft", "loop"]:
        for nsh in [10, 100, 1000]:
            fft_type = get_fft_type(ts1, ts2, "mi", "gcmi", nsh, engine)
            print(f"{name:30} engine={engine:5} nsh={nsh:4} -> {fft_type}")
    print()

print("\nConstants check:")
print(f"FFT_CONTINUOUS = {FFT_CONTINUOUS!r}")
print(f"FFT_DISCRETE = {FFT_DISCRETE!r}")
print(f"FFT_MULTIVARIATE = {FFT_MULTIVARIATE!r}")
