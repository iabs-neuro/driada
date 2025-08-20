import numpy as np
from driada.rsa.core_jit import fast_correlation_distance
from driada.utils.jit import is_jit_enabled

print(f"JIT enabled: {is_jit_enabled()}")

patterns = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],  # Identical to first
    [2, 4, 6, 8],  # Perfectly correlated but different scale
    [4, 3, 2, 1],  # Anti-correlated
])

rdm = fast_correlation_distance(patterns)

print("RDM from fast_correlation_distance:")
print(rdm)
print(f"\nDistance [0,1] (identical): {rdm[0, 1]}")
print(f"Distance [0,2] (correlated): {rdm[0, 2]}")
print(f"Distance [0,3] (anti-corr): {rdm[0, 3]}")

# Compare with numpy
from scipy.spatial.distance import pdist, squareform
rdm_scipy = squareform(pdist(patterns, metric='correlation'))
print("\nRDM from scipy correlation distance:")
print(rdm_scipy)
print(f"\nDistance [0,1] (identical): {rdm_scipy[0, 1]}")
print(f"Distance [0,2] (correlated): {rdm_scipy[0, 2]}")
print(f"Distance [0,3] (anti-corr): {rdm_scipy[0, 3]}")