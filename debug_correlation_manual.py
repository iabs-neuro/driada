import numpy as np

# Test what numpy corrcoef does
patterns = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],  # Identical to first
    [2, 4, 6, 8],  # Perfectly correlated but different scale
    [4, 3, 2, 1],  # Anti-correlated
])

# Numpy's correlation
corr_matrix = np.corrcoef(patterns)
print("Numpy corrcoef:")
print(corr_matrix)
print("\nNumpy correlation distances (1 - corr):")
print(1 - corr_matrix)

# Manual implementation matching numpy
def manual_corrcoef(patterns):
    n_items, n_features = patterns.shape
    corr_matrix = np.zeros((n_items, n_items))
    
    # Standardize each pattern
    patterns_centered = patterns - patterns.mean(axis=1, keepdims=True)
    patterns_std = patterns_centered / patterns.std(axis=1, ddof=1, keepdims=True)
    
    # Compute correlations
    for i in range(n_items):
        for j in range(n_items):
            if patterns.std(axis=1, ddof=1)[i] == 0 or patterns.std(axis=1, ddof=1)[j] == 0:
                # Handle zero variance case
                if np.array_equal(patterns[i], patterns[j]):
                    corr_matrix[i, j] = 1.0
                else:
                    corr_matrix[i, j] = 0.0
            else:
                # Correlation is the mean of the products
                corr_matrix[i, j] = np.mean(patterns_std[i] * patterns_std[j])
    
    return corr_matrix

manual_corr = manual_corrcoef(patterns)
print("\nManual corrcoef:")
print(manual_corr)
print("\nManual correlation distances (1 - corr):")
print(1 - manual_corr)