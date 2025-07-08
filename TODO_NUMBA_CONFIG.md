# TODO: Make Numba Usage Configurable in DRIADA

## Problem Statement
Numba JIT compilation can sometimes cause unexpected errors with new versions, making the entire codebase unusable. We need a way to disable Numba compilation at runtime without modifying code.

## Proposed Solution

### 1. Global Configuration
Create a global configuration system that allows disabling Numba:

```python
# driada/config.py
import os

DRIADA_USE_NUMBA = os.environ.get('DRIADA_DISABLE_NUMBA', 'false').lower() != 'true'
```

### 2. Conditional Decorator
Create a wrapper decorator that conditionally applies Numba:

```python
# driada/utils/numba_utils.py
from numba import njit
from functools import wraps
from ..config import DRIADA_USE_NUMBA

def conditional_njit(*args, **kwargs):
    """Conditionally apply numba njit decorator based on config."""
    def decorator(func):
        if DRIADA_USE_NUMBA:
            return njit(*args, **kwargs)(func)
        else:
            return func
    return decorator
```

### 3. Function-Level Control
Allow per-function control:

```python
def mi_gg(x, y, biascorrect=True, demeaned=False, max_dim=3, use_numba=None):
    """
    Parameters
    ----------
    use_numba : bool, optional
        Override global numba setting. If None, uses global config.
    """
    if use_numba is None:
        use_numba = DRIADA_USE_NUMBA
    
    if use_numba:
        return _mi_gg_numba(x, y, biascorrect, demeaned, max_dim)
    else:
        return _mi_gg_pure(x, y, biascorrect, demeaned, max_dim)
```

## Files to Update

### High Priority (Core computation functions)
- [ ] `src/driada/information/gcmi.py` - mi_gg, demean, ent_g, mi_model_gd
- [ ] `src/driada/information/info_utils.py` - py_fast_digamma_arr, py_fast_digamma
- [ ] `src/driada/information/ksg.py` - All KSG estimator functions

### Medium Priority
- [ ] Other modules using Numba decorators
- [ ] Performance-critical functions

## Implementation Steps

1. Create config module with environment variable support
2. Create conditional decorator utility
3. Update existing @njit decorators to use conditional version
4. Add tests for both Numba-enabled and disabled modes
5. Update documentation with usage instructions

## Usage Examples

### Disable Numba globally:
```bash
export DRIADA_DISABLE_NUMBA=true
python my_script.py
```

### Disable for specific function:
```python
result = mi_gg(x, y, use_numba=False)
```

## Benefits
- Graceful fallback when Numba causes issues
- Easier debugging without JIT compilation
- Compatibility with different Numba versions
- No code changes needed for end users

## Testing
- Run test suite with DRIADA_DISABLE_NUMBA=true
- Ensure all tests pass in both modes
- Performance benchmarks for impact assessment
- Test newly added functions:
  - [ ] Test entropy functions with/without Numba
  - [ ] Test GCMI functions with/without Numba
  - [ ] Verify identical results in both modes
  - [ ] Compare performance differences
  - [ ] Test edge cases that might behave differently