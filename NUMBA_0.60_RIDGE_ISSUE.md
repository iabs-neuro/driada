# Numba 0.60+ Ridge Detection Compilation Issue

## Critical Performance Consideration

**List comprehensions in Numba are significantly faster than explicit loops when they compile successfully.** The performance difference can be 2-10x depending on the operation. Therefore, we need a solution that preserves list comprehensions while fixing the type inference issue.

## Problem Summary

Starting from Numba 0.60, the `get_cwt_ridges_fast` function in `wavelet_event_detection.py` fails to compile with the following error:

```
AssertionError: assert recvr.is_precise()
File: numba/core/typeinfer.py:720
Context: resolving type of attribute "extend" of "all_ridges.1"
```

## Root Cause Analysis

### 1. Type Inference Changes in Numba 0.60+

Numba 0.60 introduced stricter type checking and inference rules. The specific issue is:

- The variable `all_ridges` is initialized using a list comprehension: `[Ridge(...) for mi in max_inds]`
- Later, the code attempts to use `all_ridges.extend(new_ridges)`
- Numba's type inference cannot determine the precise type of `all_ridges` because:
  - It starts as a list comprehension result
  - The type system sees it as `list(undefined)` rather than `list(Ridge)`
  - The `.extend()` method requires a precise type to be known

### 2. Problematic Code Pattern

```python
@njit()
def get_cwt_ridges_fast(wvtdata, peaks, wvt_times, wvt_scales):
    # This list comprehension creates type inference issues
    all_ridges = [Ridge(mi, peaks[si, mi], wvt_scales[si], wvt_time) for mi in max_inds]
    
    # Later in the code...
    new_ridges = [Ridge(...) for mi in max_inds if condition]
    all_ridges.extend(new_ridges)  # <- Type inference fails here
```

### 3. Why It Worked Before Numba 0.60

Earlier versions of Numba were more permissive with type inference and allowed some ambiguity in container types. The newer version requires explicit type information for all operations.

## Solution Approaches

### Option 1: Type-Stable List Comprehensions (Performance-Optimal)

The key is to ensure type stability while keeping list comprehensions. The issue is specifically with `.extend()` on a list created by comprehension. Solutions:

```python
@njit()
def get_cwt_ridges_fast(wvtdata, peaks, wvt_times, wvt_scales):
    # Solution 1a: Use += instead of extend()
    all_ridges = [Ridge(mi, peaks[si, mi], wvt_scales[si], wvt_time) for mi in max_inds]
    # Later...
    new_ridges = [Ridge(...) for mi in max_inds if condition]
    all_ridges += new_ridges  # += works, .extend() doesn't
    
    # Solution 1b: Create new list and reassign
    new_ridges = [Ridge(...) for mi in max_inds if condition]
    all_ridges = all_ridges + new_ridges  # Creates new list
    
    # Solution 1c: List comprehension with existing ridges
    all_ridges = all_ridges + [Ridge(...) for mi in max_inds if condition]
```

### Option 2: Typed List with Comprehension (Hybrid Approach)

Initialize with typed list but populate with comprehension:

```python
from numba.typed import List

# Create typed list first
all_ridges = List([Ridge(mi, peaks[si, mi], wvt_scales[si], wvt_time) 
                   for mi in max_inds])
# Now extend() should work because type is explicit
```

### Option 3: Explicit Typed List (Previous Recommended)

Replace list comprehensions with explicitly typed lists:

```python
@njit()
def get_cwt_ridges_fast(wvtdata, peaks, wvt_times, wvt_scales):
    # Create explicitly typed list
    all_ridges = typed.List.empty_list(Ridge.class_type.instance_type)
    
    # Populate using loops instead of comprehension
    for mi in max_inds:
        ridge = Ridge(mi, peaks[si, mi], wvt_scales[si], wvt_time)
        all_ridges.append(ridge)
```

### Option 2: Pre-allocation Pattern

Pre-allocate the list with known type:

```python
# Initialize with typed list
all_ridges = typed.List()
for i in range(len(max_inds)):
    all_ridges.append(Ridge(...))
```

### Option 3: Avoid extend() Method

Instead of using `.extend()`, use individual `.append()` calls:

```python
# Instead of: all_ridges.extend(new_ridges)
for ridge in new_ridges:
    all_ridges.append(ridge)
```

## Implementation Fix

The complete fix involves:

1. Replacing all list comprehensions with explicit loops
2. Using typed.List for Ridge collections
3. Avoiding the `.extend()` method
4. Ensuring all list operations have clear type information

## Testing Strategy

1. Test with small synthetic calcium signals
2. Verify Ridge detection accuracy remains unchanged
3. Benchmark performance (explicit loops may be slightly slower)
4. Test with both Numba 0.59 and 0.60+ for compatibility

## References

- [Numba 0.60 Release Notes](https://numba.readthedocs.io/en/stable/release-notes.html)
- [Numba Typed List Documentation](https://numba.readthedocs.io/en/stable/reference/pysupported.html#typed-list)
- [Numba Type Inference Guide](https://numba.readthedocs.io/en/stable/developer/type_system.html)

## Temporary Workaround

If immediate fix is needed:
1. Pin numba to version <0.60 in requirements
2. Or temporarily remove @njit decorator (performance impact)

## Long-term Recommendations

1. Refactor all numba-compiled functions to use explicit typing
2. Avoid list comprehensions in numba code
3. Consider splitting complex functions into smaller, type-stable components
4. Add numba version testing to CI/CD pipeline