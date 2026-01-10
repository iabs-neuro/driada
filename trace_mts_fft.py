"""Trace compute_mi_mts_fft internals."""
import numpy as np
import time
import sys

sys.path.insert(0, 'src')

# Manual implementation tracing
np.random.seed(42)
n = 3600
nsh = 100
iters = 100

# Create test data
z_orig = np.random.randn(n)
x_orig = np.random.randn(2, n)
shifts = np.random.randint(0, n, size=nsh)

# Apply copula normalization
from scipy.stats import rankdata, norm
def copnorm(x):
    if x.ndim == 1:
        ranks = rankdata(x)
        return norm.ppf(ranks / (len(ranks) + 1))
    else:
        return np.array([copnorm(row) for row in x])

copnorm_z = copnorm(z_orig)
copnorm_x = copnorm(x_orig)

print("=" * 60)
print("Tracing compute_mi_mts_fft step by step")
print("=" * 60)

# Time each step of compute_mi_mts_fft
def trace_mts_fft(copnorm_z, copnorm_x, shifts):
    """Traced version of compute_mi_mts_fft."""
    times = {}

    start = time.perf_counter()
    n = len(copnorm_z)
    copnorm_x = np.atleast_2d(copnorm_x)
    d = copnorm_x.shape[0]
    nsh = len(shifts)
    ln2 = np.log(2)
    times['setup'] = time.perf_counter() - start

    start = time.perf_counter()
    z = copnorm_z - copnorm_z.mean()
    x = copnorm_x - copnorm_x.mean(axis=1, keepdims=True)
    times['demean'] = time.perf_counter() - start

    start = time.perf_counter()
    var_z = np.var(z, ddof=1)
    cov_xx = np.cov(x)
    times['var_cov'] = time.perf_counter() - start

    start = time.perf_counter()
    H_Z = 0.5 * np.log(var_z)
    from driada.information.gcmi import regularized_cholesky
    chol_xx = regularized_cholesky(cov_xx)
    H_X = np.sum(np.log(np.diag(chol_xx)))
    times['entropy'] = time.perf_counter() - start

    start = time.perf_counter()
    fft_z = np.fft.rfft(z)
    cov_zx_all = np.zeros((d, n))
    for i in range(d):
        fft_xi = np.fft.rfft(x[i])
        cross_corr = np.fft.irfft(fft_z * np.conj(fft_xi), n=n)
        cov_zx_all[i] = cross_corr / (n - 1)
    times['fft_cov'] = time.perf_counter() - start

    start = time.perf_counter()
    shifts_int = shifts.astype(int) % n
    cov_zx = cov_zx_all[:, shifts_int]
    times['shift_extract'] = time.perf_counter() - start

    start = time.perf_counter()
    # d=2 case (3x3 determinant)
    from driada.information.info_base import _compute_joint_entropy_3x3_mts
    H_ZX = _compute_joint_entropy_3x3_mts(var_z, cov_xx, cov_zx)
    times['determinant'] = time.perf_counter() - start

    start = time.perf_counter()
    # Bias correction
    from driada.information.info_base import py_fast_digamma_arr
    dterm = (ln2 - np.log(n - 1.0)) / 2.0
    Nvarx, Nvary, Nvarxy = d, 1, d + 1
    psiterms = py_fast_digamma_arr((n - np.arange(1, Nvarxy + 1)) / 2.0) / 2.0
    H_Z = H_Z - Nvary * dterm - psiterms[:Nvary].sum()
    H_X = H_X - Nvarx * dterm - psiterms[:Nvarx].sum()
    H_ZX = H_ZX - Nvarxy * dterm - psiterms[:Nvarxy].sum()
    times['bias_correct'] = time.perf_counter() - start

    start = time.perf_counter()
    MI = (H_Z + H_X - H_ZX) / ln2
    MI = np.maximum(0, MI)
    times['final_mi'] = time.perf_counter() - start

    return MI, times

# Test 1: Single call timing breakdown
print("\n[1] Single call timing breakdown:")
mi, times = trace_mts_fft(copnorm_z, copnorm_x, shifts)
total = sum(times.values())
for name, t in times.items():
    pct = t/total*100 if total > 0 else 0
    print(f"  {name:15}: {t*1000:7.3f} ms ({pct:5.1f}%)")
print(f"  {'TOTAL':15}: {total*1000:7.3f} ms")

# Test 2: Same vs different arrays (100 iterations)
print("\n" + "=" * 60)
print("[2] Same arrays every iteration (100 calls):")
total_times = {k: 0 for k in times.keys()}
for _ in range(iters):
    _, t = trace_mts_fft(copnorm_z, copnorm_x, shifts)
    for k, v in t.items():
        total_times[k] += v
print(f"  Total time: {sum(total_times.values())*1000:.1f} ms")
print("  Breakdown:")
for name, t in total_times.items():
    pct = t/sum(total_times.values())*100
    print(f"    {name:15}: {t*1000:7.1f} ms ({pct:5.1f}%)")

print("\n" + "=" * 60)
print("[3] Different views every iteration (100 calls):")
total_times = {k: 0 for k in times.keys()}
for _ in range(iters):
    z_view = copnorm_z[::1]
    x_view = copnorm_x[:, ::1]
    _, t = trace_mts_fft(z_view, x_view, shifts)
    for k, v in t.items():
        total_times[k] += v
print(f"  Total time: {sum(total_times.values())*1000:.1f} ms")
print("  Breakdown:")
for name, t in total_times.items():
    pct = t/sum(total_times.values())*100
    print(f"    {name:15}: {t*1000:7.1f} ms ({pct:5.1f}%)")
