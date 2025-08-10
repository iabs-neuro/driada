"""
Detailed analysis of INTENSE shuffle test focusing on why p=1.0 occurs
"""

import numpy as np
from scipy.stats import rankdata, gamma, norm
import matplotlib.pyplot as plt

print("=== DETAILED INTENSE SHUFFLE TEST ANALYSIS ===\n")

# Key finding from the code analysis:
print("1. HOW INTENSE SHUFFLE TEST WORKS:")
print("-" * 60)
print("From scan_pairs function (lines 560-580):")
print("- Random shifts are generated ONCE for all neuron-feature pairs")
print("- Same seed is used: np.random.seed(seed)")
print("- Shifts are chosen from valid indices based on shuffle_mask")
print("- The SAME shift is applied to ALL features in joint_distr mode")
print("- Shift is applied using np.roll (circular shift)")
print()

print("2. KEY ISSUE - CIRCULAR SHIFT PRESERVES STRUCTURE:")
print("-" * 60)
print("np.roll does CIRCULAR shifting, not random permutation!")
print("This means:")
print("- Temporal structure is preserved")
print("- Only the alignment between signals changes")
print("- Multi-feature relationships stay intact")
print("- Some shifts may accidentally align signals better than original!")
print()

# Demonstrate the difference
np.random.seed(42)
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Original signal:", signal)
print("np.roll(signal, 3):", np.roll(signal, 3))
print("Random permutation:", np.random.permutation(signal))
print()

print("3. P-VALUE CALCULATION ISSUES:")
print("-" * 60)
print("From get_table_of_stats (lines 320-330):")
print("- Ranks are calculated: rankdata(metable, axis=2)")
print("- P-value from fitted distribution: get_mi_distr_pvalue()")
print("- Issue: If true MI <= mean of shuffles, p-value can be > 0.5")
print("- With gamma/normal fit, this can easily become p=1.0")
print()

print("4. MULTI-FEATURE PROBLEM (joint_distr=True):")
print("-" * 60)
print("From scan_pairs (lines 561-570):")
print("- ALL features get the SAME random shift")
print("- Combined shuffle mask: ts1.shuffle_mask & ts2.shuffle_mask")
print("- This preserves feature relationships!")
print("Example: If analyzing (x, y) position jointly:")
print("  - Both x and y are shifted by SAME amount")
print("  - Spatial relationships remain intact")
print("  - Shuffle may not break the true pattern")
print()

print("5. WHY P=1.0 OCCURS - MATHEMATICAL EXPLANATION:")
print("-" * 60)

# Simulate a case that produces p=1.0
np.random.seed(123)
n_samples = 1000
n_shuffles = 100

# Create two signals with weak correlation
signal1 = np.random.randn(n_samples)
signal2 = 0.3 * signal1 + 0.7 * np.random.randn(n_samples)

# Add a small delay
delay = 10
signal2 = np.roll(signal2, delay)

# Calculate true correlation
true_corr = np.corrcoef(signal1, signal2)[0, 1]
print(f"True correlation: {true_corr:.4f}")

# Generate random shifts (as INTENSE does)
shifts = np.random.randint(0, n_samples, n_shuffles)

# Calculate shuffle correlations
shuffle_corrs = []
for shift in shifts:
    shuffled_signal2 = np.roll(signal2, shift)
    corr = np.corrcoef(signal1, shuffled_signal2)[0, 1]
    shuffle_corrs.append(corr)

shuffle_corrs = np.array(shuffle_corrs)

# Check how many shuffles beat the true correlation
better_than_true = np.sum(shuffle_corrs > true_corr)
print(f"\nShuffles better than true: {better_than_true}/{n_shuffles}")
print(f"Mean shuffle correlation: {np.mean(shuffle_corrs):.4f}")
print(f"Std shuffle correlation: {np.std(shuffle_corrs):.4f}")

# Fit distributions and calculate p-values
# Shift data for gamma (as INTENSE might do internally)
min_val = min(np.min(shuffle_corrs), true_corr)
shifted_shuffles = shuffle_corrs - min_val + 0.001
shifted_true = true_corr - min_val + 0.001

try:
    gamma_params = gamma.fit(shifted_shuffles, floc=0)
    gamma_pval = gamma(*gamma_params).sf(shifted_true)
except:
    gamma_pval = 1.0

norm_params = norm.fit(shuffle_corrs)
norm_pval = norm(*norm_params).sf(true_corr)

print(f"\nP-values:")
print(f"  Gamma: {gamma_pval:.6f}")
print(f"  Normal: {norm_pval:.6f}")

if norm_pval > 0.99:
    print("\n!!! P-VALUE IS ESSENTIALLY 1.0 !!!")
    print("This happens when true value is below shuffle distribution mean")

print("\n6. SOLUTIONS AND RECOMMENDATIONS:")
print("-" * 60)
print("1. Use PERMUTATION instead of CIRCULAR SHIFT:")
print("   - Breaks temporal structure completely")
print("   - More appropriate null hypothesis")
print()
print("2. For multi-feature analysis:")
print("   - Shuffle features INDEPENDENTLY")
print("   - Or use block permutation to preserve local structure")
print()
print("3. Consider alternative null hypotheses:")
print("   - Phase randomization (preserves power spectrum)")
print("   - ARIMA model residuals")
print("   - Surrogate data methods")
print()
print("4. Use more shuffles (10,000+) for better p-value resolution")
print()
print("5. Check shuffle distribution before fitting:")
print("   - Ensure true value is in tail, not center")
print("   - Consider rank-based tests only")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Circular shift vs permutation
ax = axes[0, 0]
test_signal = np.sin(np.linspace(0, 4*np.pi, 100))
ax.plot(test_signal, 'b-', label='Original', linewidth=2)
ax.plot(np.roll(test_signal, 25), 'r--', label='Circular shift (25)', linewidth=2)
ax.plot(np.random.permutation(test_signal), 'g:', label='Random permutation', linewidth=2)
ax.set_title('Circular Shift vs Permutation')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Shuffle distribution
ax = axes[0, 1]
ax.hist(shuffle_corrs, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
ax.axvline(true_corr, color='red', linestyle='--', linewidth=2, label=f'True ({true_corr:.3f})')
ax.axvline(np.mean(shuffle_corrs), color='green', linestyle=':', linewidth=2, label=f'Shuffle mean ({np.mean(shuffle_corrs):.3f})')
ax.set_xlabel('Correlation')
ax.set_ylabel('Density')
ax.set_title(f'Shuffle Distribution (p={norm_pval:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: P-value as function of effect size
ax = axes[1, 0]
effect_sizes = np.linspace(-0.5, 0.5, 50)
pvals = []
for effect in effect_sizes:
    test_val = np.mean(shuffle_corrs) + effect * np.std(shuffle_corrs)
    pval = norm(*norm_params).sf(test_val)
    pvals.append(pval)

ax.plot(effect_sizes, pvals, 'b-', linewidth=2)
ax.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='p=0.05')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Effect size (in shuffle std units)')
ax.set_ylabel('P-value')
ax.set_title('P-value vs Effect Size')
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Summary
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
INTENSE SHUFFLE TEST ISSUES:

1. CIRCULAR SHIFT (np.roll):
   • Preserves temporal structure
   • Not true randomization
   • Can align signals by chance

2. MULTI-FEATURE PROBLEM:
   • Same shift for all features
   • Preserves relationships
   • Reduces test power

3. P=1.0 OCCURS WHEN:
   • True value ≤ shuffle mean
   • Weak effects
   • Accidental alignments

4. CURRENT EXAMPLE:
   • True corr: {true_corr:.3f}
   • Shuffle mean: {np.mean(shuffle_corrs):.3f}
   • P-value: {norm_pval:.3f}
   • Better shuffles: {better_than_true}/{n_shuffles}
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        verticalalignment='top', fontfamily='monospace', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig('intense_shuffle_detailed_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n7. CODE REFERENCES:")
print("-" * 60)
print("Key functions in intense_base.py:")
print("- scan_pairs (lines 464-643): Generates shifts and calculates metrics")
print("- Lines 574-579: Random shift generation")
print("- Lines 626-635: Shuffle metric calculation")
print()
print("In stats.py:")
print("- get_table_of_stats (lines 283-346): Converts to p-values")
print("- get_mi_distr_pvalue (lines 95-130): Fits distribution")
print("- Line 320: rankdata(metable, axis=2) for ranking")