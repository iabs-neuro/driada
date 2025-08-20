import numpy as np

# Copy the function inline to debug
def fast_correlation_distance_debug(patterns):
    n_items, n_features = patterns.shape
    rdm = np.zeros((n_items, n_items))
    
    # Standardize patterns (mean=0, std=1) - manual computation for numba
    patterns_std = np.zeros_like(patterns)
    for i in range(n_items):
        # Compute mean
        mean = 0.0
        for j in range(n_features):
            mean += patterns[i, j]
        mean /= n_features
        
        # Compute std with ddof=1 (sample std) to match numpy.corrcoef
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
            # If std is 0, all values are the same, set to 0
            for j in range(n_features):
                patterns_std[i, j] = 0.0
        
        if i == 0:  # Debug first pattern
            print(f"Pattern {i}: mean={mean}, var={var}, std={std}")
            print(f"  Standardized: {patterns_std[i]}")
            sum_sq = sum(patterns_std[i]**2)
            print(f"  Sum of squares: {sum_sq}")
    
    # Compute correlation distances
    for i in range(n_items):
        for j in range(i + 1, n_items):
            # Handle edge case where both patterns have zero variance
            # Check if either pattern is all zeros (standardized)
            pattern_i_is_zero = True
            pattern_j_is_zero = True
            
            for k in range(n_features):
                if patterns_std[i, k] != 0.0:
                    pattern_i_is_zero = False
                if patterns_std[j, k] != 0.0:
                    pattern_j_is_zero = False
            
            if pattern_i_is_zero or pattern_j_is_zero:
                # If either pattern has zero variance, correlation is undefined
                # Set distance to 0 if patterns are identical, else 1
                patterns_equal = True
                for k in range(n_features):
                    if patterns[i, k] != patterns[j, k]:
                        patterns_equal = False
                        break
                dist = 0.0 if patterns_equal else 1.0
            else:
                # Normal correlation computation
                # When standardized with ddof=1, sum of squares = n-1
                # So correlation = dot product / (n-1)
                corr = 0.0
                for k in range(n_features):
                    corr += patterns_std[i, k] * patterns_std[j, k]
                
                if i == 0 and j == 1:  # Debug first pair
                    print(f"\nCorrelation between patterns {i} and {j}:")
                    print(f"  Sum of products: {corr}")
                    print(f"  n_features - 1 = {n_features - 1}")
                
                # Divide by (n-1) to get correlation coefficient
                if n_features > 1:
                    corr /= (n_features - 1)
                else:
                    corr = 1.0  # Single feature case
                
                if i == 0 and j == 1:  # Debug first pair
                    print(f"  Correlation: {corr}")
                
                # Clip to [-1, 1] to handle numerical errors
                if corr > 1.0:
                    corr = 1.0
                elif corr < -1.0:
                    corr = -1.0
                
                # Distance = 1 - correlation
                dist = 1.0 - corr
                
                if i == 0 and j == 1:  # Debug first pair
                    print(f"  Distance: {dist}")
            
            rdm[i, j] = dist
            rdm[j, i] = dist
    
    return rdm

# Test
patterns = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],  # Identical to first
    [2, 4, 6, 8],  # Perfectly correlated but different scale
    [4, 3, 2, 1],  # Anti-correlated
], dtype=float)

rdm = fast_correlation_distance_debug(patterns)
print("\nFinal RDM:")
print(rdm)