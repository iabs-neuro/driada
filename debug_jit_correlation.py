import numpy as np

# Test patterns
patterns = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],  # Identical to first
], dtype=float)

n_items, n_features = patterns.shape

# Manual standardization like in JIT function
patterns_std = np.zeros_like(patterns)
for i in range(n_items):
    # Compute mean
    mean = 0.0
    for j in range(n_features):
        mean += patterns[i, j]
    mean /= n_features
    
    # Compute std with ddof=1
    var = 0.0
    for j in range(n_features):
        diff = patterns[i, j] - mean
        var += diff * diff
    # Use n-1 for sample standard deviation
    if n_features > 1:
        std = np.sqrt(var / (n_features - 1))
    else:
        std = 0.0
    
    # Standardize
    if std > 0:
        for j in range(n_features):
            patterns_std[i, j] = (patterns[i, j] - mean) / std
    else:
        for j in range(n_features):
            patterns_std[i, j] = 0.0
    
    print(f"Pattern {i}:")
    print(f"  Mean: {mean}")
    print(f"  Var: {var}")
    print(f"  Std: {std}")
    print(f"  Standardized: {patterns_std[i]}")
    
    # Check sum of squares
    sum_sq = 0.0
    for j in range(n_features):
        sum_sq += patterns_std[i, j] * patterns_std[i, j]
    print(f"  Sum of squares: {sum_sq}")

# Compute correlation
i, j = 0, 1
corr = 0.0
for k in range(n_features):
    corr += patterns_std[i, k] * patterns_std[j, k]
    print(f"    k={k}: {patterns_std[i, k]} * {patterns_std[j, k]} = {patterns_std[i, k] * patterns_std[j, k]}")

print(f"\nSum of products: {corr}")
print(f"Divided by (n-1={n_features-1}): {corr / (n_features - 1)}")
print(f"Distance (1 - corr): {1 - corr / (n_features - 1)}")

# Compare with numpy
corr_numpy = np.corrcoef(patterns)[0, 1]
print(f"\nNumpy correlation: {corr_numpy}")
print(f"Numpy distance: {1 - corr_numpy}")