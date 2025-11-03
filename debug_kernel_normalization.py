"""Diagnostic script to investigate kernel normalization bug."""

import numpy as np
import matplotlib.pyplot as plt

def compute_kernel(t_array, t_rise, t_off):
    """Compute unnormalized kernel."""
    return (1 - np.exp(-t_array / t_rise)) * np.exp(-t_array / t_off)

def normalize_kernel(kernel):
    """Normalize kernel to max=1."""
    kernel_max = np.max(kernel)
    if kernel_max > 0:
        return kernel / kernel_max, kernel_max
    return kernel, 0

# Parameters from test
t_rise = 4.68
t_off = 40.58

# Full kernel (500 frames, as used in reconstruction)
t_full = np.arange(500)
kernel_full = compute_kernel(t_full, t_rise, t_off)
kernel_full_norm, max_full = normalize_kernel(kernel_full)

print(f"Full kernel (500 frames):")
print(f"  True peak value: {max_full:.6f}")
print(f"  Peak location: {np.argmax(kernel_full)} frames")

# Truncated kernels (as would happen near end of recording)
for remaining in [500, 100, 50, 20, 10]:
    t_trunc = np.arange(remaining)
    kernel_trunc = compute_kernel(t_trunc, t_rise, t_off)
    kernel_trunc_norm, max_trunc = normalize_kernel(kernel_trunc)

    print(f"\nTruncated kernel ({remaining} frames):")
    print(f"  Max value reached: {max_trunc:.6f}")
    print(f"  Ratio to true peak: {max_trunc / max_full:.6f}")
    print(f"  After normalization, amplitude will be scaled by: {max_full / max_trunc:.6f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Top: unnormalized kernels
ax = axes[0]
for remaining in [500, 100, 50, 20, 10]:
    t_trunc = np.arange(remaining)
    kernel_trunc = compute_kernel(t_trunc, t_rise, t_off)
    ax.plot(t_trunc, kernel_trunc, label=f'{remaining} frames', linewidth=2)
ax.axhline(max_full, color='k', linestyle='--', label='True peak')
ax.set_xlabel('Time (frames)')
ax.set_ylabel('Kernel value (unnormalized)')
ax.set_title('Unnormalized kernels with different truncation lengths')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom: normalized kernels
ax = axes[1]
for remaining in [500, 100, 50, 20, 10]:
    t_trunc = np.arange(remaining)
    kernel_trunc = compute_kernel(t_trunc, t_rise, t_off)
    kernel_trunc_norm, _ = normalize_kernel(kernel_trunc)
    ax.plot(t_trunc, kernel_trunc_norm, label=f'{remaining} frames', linewidth=2)
ax.set_xlabel('Time (frames)')
ax.set_ylabel('Kernel value (normalized)')
ax.set_title('Normalized kernels - all reach max=1.0 (THIS IS THE BUG!)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kernel_normalization_bug.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to kernel_normalization_bug.png")

# Calculate expected amplitude error for common scenario
print(f"\n" + "="*60)
print(f"EXPECTED AMPLITUDE ERROR ANALYSIS:")
print(f"="*60)
print(f"If an event is detected at frame where only 15 frames remain:")
t_15 = np.arange(15)
k_15 = compute_kernel(t_15, t_rise, t_off)
max_15 = np.max(k_15)
error_factor = max_full / max_15
print(f"  Kernel reaches only {max_15:.4f} of true peak ({max_15/max_full*100:.1f}%)")
print(f"  NNLS will extract amplitude {error_factor:.2f}x too high!")
print(f"  Reconstruction will then be {error_factor:.2f}x too high!")
print(f"\nFor our 50% overestimation (factor of 1.5):")
print(f"  Kernel would need to reach {max_full/1.5:.4f} of true peak")
t_search = np.arange(1, 100)
for frames_remaining in t_search:
    t = np.arange(frames_remaining)
    k = compute_kernel(t, t_rise, t_off)
    if np.max(k) / max_full <= 1/1.5:
        print(f"  This occurs at ~{frames_remaining} frames remaining")
        print(f"  Actual ratio: {max_full / np.max(k):.3f}")
        break
