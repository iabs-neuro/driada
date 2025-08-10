"""
Script to analyze how INTENSE shuffle test works and why it might produce p=1.0
even when there's a real relationship between signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, gamma, norm
from driada.information.info_base import TimeSeries

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic data with known relationship
n_samples = 1000
time = np.arange(n_samples)

# Create behavioral signal (e.g., running speed)
behavior = np.sin(2 * np.pi * time / 100) + 0.5 * np.random.randn(n_samples)

# Create neural signal correlated with behavior but with a delay
delay = 20  # frames
neural = np.roll(0.7 * behavior + 0.3 * np.random.randn(n_samples), delay)

# Add some noise
neural += 0.2 * np.random.randn(n_samples)

# Create TimeSeries objects (simulating INTENSE input)
ts_neural = TimeSeries(neural)
ts_behavior = TimeSeries(behavior)

print("=== Analyzing INTENSE Shuffle Test Mechanism ===\n")

# 1. Understanding the shuffle mechanism
print("1. SHUFFLE MECHANISM:")
print("-" * 50)

# The shuffle test works by rolling (circular shifting) one signal relative to the other
# Let's simulate what happens during shuffling

n_shuffles = 100
shuffle_shifts = np.random.randint(0, n_samples, n_shuffles)

print(f"Number of shuffles: {n_shuffles}")
print(f"Example shuffle shifts: {shuffle_shifts[:5]}")
print(f"Signal length: {n_samples}")

# 2. Calculate correlation for each shuffle
print("\n2. CORRELATION VALUES:")
print("-" * 50)

# True correlation at optimal delay
true_corr = np.corrcoef(neural, behavior)[0, 1]
print(f"True correlation (no shift): {true_corr:.4f}")

# Correlation at optimal shift
optimal_corr = np.corrcoef(neural, np.roll(behavior, -delay))[0, 1]
print(f"Optimal correlation (shift={-delay}): {optimal_corr:.4f}")

# Calculate correlations for all shuffles
shuffle_corrs = []
for shift in shuffle_shifts:
    shuffled_behavior = np.roll(behavior, shift)
    corr = np.corrcoef(neural, shuffled_behavior)[0, 1]
    shuffle_corrs.append(corr)

shuffle_corrs = np.array(shuffle_corrs)

print(f"\nShuffle correlations:")
print(f"  Mean: {np.mean(shuffle_corrs):.4f}")
print(f"  Std: {np.std(shuffle_corrs):.4f}")
print(f"  Min: {np.min(shuffle_corrs):.4f}")
print(f"  Max: {np.max(shuffle_corrs):.4f}")

# 3. The problem: some shuffle shifts might hit the true delay
print("\n3. THE CRITICAL ISSUE:")
print("-" * 50)

# Check if any shuffle shift is close to the true delay
close_to_true_delay = np.abs(shuffle_shifts - delay) < 5
n_close = np.sum(close_to_true_delay)

print(f"Number of shuffles close to true delay (±5 frames): {n_close}")
if n_close > 0:
    print(f"Shuffle shifts close to true delay: {shuffle_shifts[close_to_true_delay]}")
    print(f"Correlations for these shuffles: {shuffle_corrs[close_to_true_delay]}")

# 4. P-value calculation (as done in INTENSE)
print("\n4. P-VALUE CALCULATION:")
print("-" * 50)

# Add the true correlation to the shuffle distribution
all_corrs = np.concatenate([[optimal_corr], shuffle_corrs])

# Rank the correlations (as done in get_table_of_stats)
ranks = rankdata(all_corrs)
true_rank = ranks[0]
rank_ratio = true_rank / (n_shuffles + 1)

print(f"True correlation rank: {true_rank} out of {n_shuffles + 1}")
print(f"Rank ratio: {rank_ratio:.4f}")

# Fit distribution and calculate p-value (as done in INTENSE)
# Try both gamma and normal distributions
print("\nFitting distributions to shuffle correlations:")

# Gamma distribution (default in INTENSE)
# Note: gamma distribution requires positive values, so we shift the data
min_corr = min(np.min(shuffle_corrs), optimal_corr)
shifted_shuffle_corrs = shuffle_corrs - min_corr + 0.01  # Shift to make positive
shifted_optimal_corr = optimal_corr - min_corr + 0.01
gamma_params = gamma.fit(shifted_shuffle_corrs, floc=0)
gamma_pval = gamma(*gamma_params).sf(shifted_optimal_corr)
print(f"Gamma p-value: {gamma_pval:.6f}")

# Normal distribution
norm_params = norm.fit(shuffle_corrs)
norm_pval = norm(*norm_params).sf(optimal_corr)
print(f"Normal p-value: {norm_pval:.6f}")

# 5. Visualization
print("\n5. CREATING VISUALIZATION...")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Original signals
ax = axes[0, 0]
ax.plot(time[:200], behavior[:200], label='Behavior', alpha=0.7)
ax.plot(time[:200], neural[:200], label='Neural', alpha=0.7)
ax.set_xlabel('Time (frames)')
ax.set_ylabel('Signal')
ax.set_title('Original Signals (first 200 frames)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Correlation vs shift
ax = axes[0, 1]
all_shifts = np.arange(-100, 100)
all_shift_corrs = []
for shift in all_shifts:
    corr = np.corrcoef(neural, np.roll(behavior, shift))[0, 1]
    all_shift_corrs.append(corr)

ax.plot(all_shifts, all_shift_corrs, 'b-', alpha=0.7)
ax.axvline(-delay, color='r', linestyle='--', label=f'True delay ({-delay})')
ax.axhline(optimal_corr, color='r', linestyle=':', alpha=0.5)
ax.scatter(shuffle_shifts - n_samples//2, shuffle_corrs, color='orange', 
           alpha=0.5, s=20, label='Shuffle positions')
ax.set_xlabel('Shift (frames)')
ax.set_ylabel('Correlation')
ax.set_title('Correlation vs Shift')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Shuffle distribution
ax = axes[1, 0]
n_bins = 30
counts, bins, _ = ax.hist(shuffle_corrs, bins=n_bins, alpha=0.7, 
                          density=True, color='skyblue', edgecolor='black')
ax.axvline(optimal_corr, color='red', linestyle='--', linewidth=2, 
           label=f'True correlation ({optimal_corr:.3f})')

# Plot fitted distributions
x_range = np.linspace(shuffle_corrs.min(), shuffle_corrs.max(), 100)
# For gamma, we need to shift x_range too
x_range_shifted = x_range - min_corr + 0.01
gamma_pdf = gamma(*gamma_params).pdf(x_range_shifted)
norm_pdf = norm(*norm_params).pdf(x_range)
ax.plot(x_range, gamma_pdf, 'g-', linewidth=2, label=f'Gamma fit (p={gamma_pval:.3f})')
ax.plot(x_range, norm_pdf, 'm-', linewidth=2, label=f'Normal fit (p={norm_pval:.3f})')

ax.set_xlabel('Correlation')
ax.set_ylabel('Density')
ax.set_title('Shuffle Distribution with Fitted Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Key insights
ax = axes[1, 1]
ax.axis('off')
insights_text = f"""
KEY INSIGHTS:

1. SHUFFLE MECHANISM:
   • INTENSE uses np.roll for shuffling
   • This preserves temporal structure
   • Some shuffles may hit true delay

2. RANK-BASED TEST:
   • True value rank: {true_rank}/{n_shuffles + 1}
   • Rank ratio: {rank_ratio:.3f}
   • Top-k criterion: k=1 for stage 1

3. P-VALUE ISSUES:
   • Gamma p-value: {gamma_pval:.3f}
   • Normal p-value: {norm_pval:.3f}
   • P=1.0 if true correlation ≤ shuffle mean

4. WHY P=1.0 OCCURS:
   • Random shifts can hit optimal delay
   • Multi-feature relationships preserved
   • Distribution fitting assumptions
   • Conservative when true effect modest
"""
ax.text(0.05, 0.95, insights_text, transform=ax.transAxes,
        verticalalignment='top', fontfamily='monospace', fontsize=10)

plt.tight_layout()
plt.savefig('intense_shuffle_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. Additional analysis: Multi-feature case
print("\n6. MULTI-FEATURE ANALYSIS:")
print("-" * 50)

# In joint_distr mode, all features are shifted together
print("In joint distribution mode:")
print("- ALL features are shifted by the SAME random amount")
print("- This preserves relationships between features")
print("- Makes it harder to break genuine multi-feature patterns")
print("- Can lead to high p-values even with real effects")

# 7. Summary
print("\n7. SUMMARY:")
print("-" * 50)
print("INTENSE shuffle test can produce p=1.0 because:")
print("1. Random circular shifts may accidentally hit the true optimal delay")
print("2. The distribution fitting assumes shuffle distribution represents null")
print("3. Multi-feature relationships are preserved during shuffling")
print("4. Small sample sizes or weak effects make true value indistinguishable")
print("5. Conservative nature of the test (especially with normal distribution)")