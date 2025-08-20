import numpy as np
from scipy.spatial.distance import pdist, squareform

# Orthogonal patterns from the test
patterns = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

# Check with numpy corrcoef
corr_matrix = np.corrcoef(patterns)
print("Numpy correlation matrix:")
print(corr_matrix)
print("\nNumpy correlation distances:")
print(1 - corr_matrix)

# Check with scipy
rdm_scipy = squareform(pdist(patterns, metric='correlation'))
print("\nScipy correlation distances:")
print(rdm_scipy)

# The issue: these patterns don't have zero mean!
print("\nPattern means:", patterns.mean(axis=1))
print("Pattern stds:", patterns.std(axis=1, ddof=1))