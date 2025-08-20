import numpy as np
from driada.rsa.core_jit import fast_correlation_distance

# Test patterns
patterns = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],  # Identical to first
    [2, 4, 6, 8],  # Perfectly correlated but different scale
    [4, 3, 2, 1],  # Anti-correlated
])

rdm = fast_correlation_distance(patterns)
print("RDM:")
print(rdm)
print(f"\nDistance between identical patterns [0,1]: {rdm[0, 1]}")
print(f"Distance between correlated patterns [0,2]: {rdm[0, 2]}")
print(f"Distance between anti-correlated [0,3]: {rdm[0, 3]}")

# Check correlation manually
def manual_correlation_distance(x, y):
    # Standardize
    x_std = (x - x.mean()) / x.std(ddof=1)
    y_std = (y - y.mean()) / y.std(ddof=1)
    # Correlation
    corr = np.mean(x_std * y_std) * len(x) / (len(x) - 1)
    # Distance
    return 1 - corr

print("\nManual calculation:")
print(f"Distance [0,1]: {manual_correlation_distance(patterns[0], patterns[1])}")
print(f"Distance [0,2]: {manual_correlation_distance(patterns[0], patterns[2])}")
print(f"Distance [0,3]: {manual_correlation_distance(patterns[0], patterns[3])}")