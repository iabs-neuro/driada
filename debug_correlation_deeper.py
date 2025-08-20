import numpy as np

# Test patterns
patterns = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],  # Identical to first
    [2, 4, 6, 8],  # Perfectly correlated but different scale
    [4, 3, 2, 1],  # Anti-correlated
])

print("Original patterns:")
print(patterns)

# Standardize manually with ddof=1
n_features = patterns.shape[1]
patterns_std = np.zeros_like(patterns, dtype=float)

for i in range(patterns.shape[0]):
    mean = patterns[i].mean()
    std = patterns[i].std(ddof=1)  # Sample std
    patterns_std[i] = (patterns[i] - mean) / std
    
    print(f"\nPattern {i}:")
    print(f"  Mean: {mean}")
    print(f"  Std (ddof=1): {std}")
    print(f"  Standardized: {patterns_std[i]}")
    print(f"  Sum of squares: {np.sum(patterns_std[i]**2)}")

# Check correlation using numpy
print("\n\nNumpy corrcoef:")
corr_matrix = np.corrcoef(patterns)
print(corr_matrix)

# Manual correlation calculation
print("\n\nManual correlations:")
for i in range(4):
    for j in range(i+1, 4):
        # Method 1: Direct dot product / (n-1)
        corr1 = np.dot(patterns_std[i], patterns_std[j]) / (n_features - 1)
        
        # Method 2: Mean of products
        corr2 = np.mean(patterns_std[i] * patterns_std[j])
        
        # Method 3: What numpy actually does
        cov = np.mean((patterns[i] - patterns[i].mean()) * (patterns[j] - patterns[j].mean()))
        std_i = patterns[i].std(ddof=1)
        std_j = patterns[j].std(ddof=1)
        corr3 = cov / (std_i * std_j) if std_i > 0 and std_j > 0 else 0
        
        print(f"  Corr[{i},{j}]:")
        print(f"    Method 1 (dot/(n-1)): {corr1:.6f}")
        print(f"    Method 2 (mean):      {corr2:.6f}")
        print(f"    Method 3 (numpy way): {corr3:.6f}")
        print(f"    Numpy:                {corr_matrix[i,j]:.6f}")