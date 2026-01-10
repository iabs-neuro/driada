"""Compare delay optimization time vs shuffle time."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import time
from driada.information import TimeSeries, MultiTimeSeries
from driada.information.info_base import compute_mi_batch_fft, compute_mi_gd_fft, compute_mi_mts_fft

np.random.seed(42)

# Parameters matching benchmark
n_frames = 18000
ds = 5
n_downsampled = n_frames // ds  # 3600

n_neurons = 500
n_continuous_feats = 4  # hd, speed, fbm_0, and one more
n_discrete_feats = 5    # event_0 through event_4
n_mts_feats = 1        # position_2d

shift_window = 40  # 2 sec * 20 fps
n_shifts = 2 * shift_window // ds + 1  # 17 shifts

n_shuffles_stage1 = 100
n_shuffles_stage2 = 10000

print("=" * 70)
print("DELAY VS SHUFFLE TIMING COMPARISON")
print("=" * 70)
print(f"\nData: {n_downsampled} samples (after ds={ds})")
print(f"Neurons: {n_neurons}")
print(f"Features: {n_continuous_feats} continuous, {n_discrete_feats} discrete, {n_mts_feats} MTS")
print(f"Shifts for delays: {n_shifts}")
print(f"Shuffles stage1: {n_shuffles_stage1}")
print(f"Shuffles stage2: {n_shuffles_stage2}")

# Create test data
neuron_data = np.random.randn(n_downsampled)
cont_feat_data = np.random.randn(n_downsampled)
disc_feat_data = np.random.randint(0, 5, n_downsampled)
mts_feat_data = np.random.randn(2, n_downsampled)

# Apply copula normalization
from scipy.stats import rankdata, norm
def copnorm(x):
    if x.ndim == 1:
        ranks = rankdata(x)
        return norm.ppf(ranks / (len(ranks) + 1))
    else:
        return np.array([copnorm(row) for row in x])

neuron_cn = copnorm(neuron_data)
cont_feat_cn = copnorm(cont_feat_data)
mts_feat_cn = copnorm(mts_feat_data)

# Warmup
shifts = np.arange(n_shifts)
for _ in range(3):
    compute_mi_batch_fft(neuron_cn, cont_feat_cn, shifts)
    compute_mi_gd_fft(neuron_cn, disc_feat_data, shifts)
    compute_mi_mts_fft(neuron_cn, mts_feat_cn, shifts)

print("\n" + "-" * 70)
print("TIMING SINGLE CALLS")
print("-" * 70)

# Time single calls for delays (17 shifts)
delay_shifts = np.arange(n_shifts)
iters = 100

start = time.perf_counter()
for _ in range(iters):
    compute_mi_batch_fft(neuron_cn, cont_feat_cn, delay_shifts)
t_cc_delay = (time.perf_counter() - start) / iters * 1000
print(f"CC delay ({n_shifts} shifts): {t_cc_delay:.3f}ms")

start = time.perf_counter()
for _ in range(iters):
    compute_mi_gd_fft(neuron_cn, disc_feat_data, delay_shifts)
t_gd_delay = (time.perf_counter() - start) / iters * 1000
print(f"GD delay ({n_shifts} shifts): {t_gd_delay:.3f}ms")

start = time.perf_counter()
for _ in range(iters):
    compute_mi_mts_fft(neuron_cn, mts_feat_cn, delay_shifts)
t_mts_delay = (time.perf_counter() - start) / iters * 1000
print(f"MTS delay ({n_shifts} shifts): {t_mts_delay:.3f}ms")

# Time single calls for stage1 (100 shuffles)
shuffle_shifts_s1 = np.random.randint(0, n_downsampled, n_shuffles_stage1)

start = time.perf_counter()
for _ in range(iters):
    compute_mi_batch_fft(neuron_cn, cont_feat_cn, shuffle_shifts_s1)
t_cc_s1 = (time.perf_counter() - start) / iters * 1000
print(f"CC stage1 ({n_shuffles_stage1} shuffles): {t_cc_s1:.3f}ms")

start = time.perf_counter()
for _ in range(iters):
    compute_mi_gd_fft(neuron_cn, disc_feat_data, shuffle_shifts_s1)
t_gd_s1 = (time.perf_counter() - start) / iters * 1000
print(f"GD stage1 ({n_shuffles_stage1} shuffles): {t_gd_s1:.3f}ms")

start = time.perf_counter()
for _ in range(iters):
    compute_mi_mts_fft(neuron_cn, mts_feat_cn, shuffle_shifts_s1)
t_mts_s1 = (time.perf_counter() - start) / iters * 1000
print(f"MTS stage1 ({n_shuffles_stage1} shuffles): {t_mts_s1:.3f}ms")

print("\n" + "-" * 70)
print("PROJECTED TOTAL TIMES")
print("-" * 70)

# Calculate projected total times
# Delays: n_neurons * (n_continuous + n_discrete) pairs (MTS skipped)
n_delay_pairs = n_neurons * (n_continuous_feats + n_discrete_feats)
t_delays_cc = t_cc_delay * n_neurons * n_continuous_feats / 1000
t_delays_gd = t_gd_delay * n_neurons * n_discrete_feats / 1000
t_delays_total = t_delays_cc + t_delays_gd

print(f"\nDelay optimization ({n_delay_pairs} pairs):")
print(f"  CC ({n_neurons * n_continuous_feats} pairs): {t_delays_cc:.1f}s")
print(f"  GD ({n_neurons * n_discrete_feats} pairs): {t_delays_gd:.1f}s")
print(f"  Total projected: {t_delays_total:.1f}s")

# Stage 1: all pairs
n_all_pairs = n_neurons * (n_continuous_feats + n_discrete_feats + n_mts_feats)
t_s1_cc = t_cc_s1 * n_neurons * n_continuous_feats / 1000
t_s1_gd = t_gd_s1 * n_neurons * n_discrete_feats / 1000
t_s1_mts = t_mts_s1 * n_neurons * n_mts_feats / 1000
t_s1_total = t_s1_cc + t_s1_gd + t_s1_mts

print(f"\nStage 1 ({n_all_pairs} pairs, {n_shuffles_stage1} shuffles):")
print(f"  CC ({n_neurons * n_continuous_feats} pairs): {t_s1_cc:.1f}s")
print(f"  GD ({n_neurons * n_discrete_feats} pairs): {t_s1_gd:.1f}s")
print(f"  MTS ({n_neurons * n_mts_feats} pairs): {t_s1_mts:.1f}s")
print(f"  Total projected: {t_s1_total:.1f}s")

print("\n" + "=" * 70)
