# Test Suite Optimization Guide

## Performance Improvements

### 1. Fast Test Mode
Set environment variable `INTENSE_FAST_TESTS=1` to enable fast mode:
```bash
INTENSE_FAST_TESTS=1 pytest
```

### 2. Key Optimizations Implemented

1. **Configurable Duration in Synthetic Data**
   - Made `duration` parameter configurable in `generate_synthetic_exp`
   - Default: 1200 seconds → Fast mode: 400 seconds
   - Reduces data points by 67%

2. **Test Parameter Reductions**
   - Shuffles: 100/1000 → 10/50 (90-95% reduction)
   - Neurons: 20-40 → 5-10 (75% reduction)
   - Time series length: 10,000 → 2,000 (80% reduction)
   - Downsampling: ds=1 → ds=2 (50% data reduction)

3. **Test Markers**
   - Use `@pytest.mark.slow` for time-consuming tests
   - Run fast tests only: `pytest -m "not slow"`

4. **Parallel Execution**
   - Enabled by default in test configurations
   - Use `pytest -n auto` for multi-core execution

### 3. Performance Results

| Test Suite | Before | After | Improvement |
|------------|--------|-------|-------------|
| Full suite | 388.89s | ~150s* | 61% faster |
| INTENSE only | 216.89s | ~90s* | 58% faster |
| Fast tests | N/A | 45.63s | New option |

*Estimated with fast mode enabled

### 4. Running Tests

```bash
# Fast mode - for CI/quick validation
INTENSE_FAST_TESTS=1 pytest tests/test_intense.py -m "not slow"

# Normal mode - for thorough testing
pytest tests/test_intense.py

# With coverage (use alternative engine to avoid pandas compatibility issues)
export COVERAGE_CORE=sysmon && pytest tests/test_intense.py --cov=src/driada/intense --cov-report=term-missing

# Run specific test patterns (e.g., correlation detection at different scales)
pytest tests/test_intense.py::test_correlation_detection_scaled -v
```

### 5. Future Optimizations

1. **JIT Compilation**: Add Numba JIT to more functions:
   - `ctransform` and `copnorm` in gcmi.py (needs rewrite)
   - Entropy functions in entropy.py
   - Statistical computations in stats.py

2. **Test Data Caching**: Cache synthetic data between test runs

3. **Adaptive Testing**: Skip redundant tests based on coverage

4. **GPU Acceleration**: For large-scale MI computations