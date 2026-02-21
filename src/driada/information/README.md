# Information Theory Module

## Overview

This module provides the information-theoretic foundation for DRIADA, implementing various measures of statistical dependency including mutual information (MI), entropy, and conditional information measures.

## Key components

### Time series classes
- `TimeSeries` - Single variable time series
- `MultiTimeSeries` - Multi-dimensional time series (inherits from MVData)
  - Supports mixed continuous/discrete data
  - Automatic type detection
  - Direct access to dimensionality reduction methods

### Mutual information estimators

**Main interface:**
- `get_mi()` - Automatic method selection based on data types
- `get_1d_mi()` - Single variable pairs
- `get_multi_mi()` - Multi-dimensional features

**Specialized estimators:**
- **GCMI (Gaussian Copula MI)** - Fast, robust for continuous data
  - `mi_gg()` - Gaussian to Gaussian
  - `gcmi_cc()` - Continuous to continuous
  - `mi_model_gd()` - Model-based for discrete outcomes

- **KSG (k-Nearest Neighbors)** - Non-parametric, no assumptions
  - `nonparam_mi_cc()` - Continuous-continuous
  - `nonparam_mi_cd()` - Continuous-discrete
  - `nonparam_mi_dc()` - Discrete-continuous

### Advanced measures
- `conditional_mi()` - MI conditioned on third variable
- `interaction_information()` - Three-way interactions
- `get_tdmi()` - Time-delayed mutual information

## Example usage

```python
from driada.information import get_mi, MultiTimeSeries

# Basic MI calculation
mi_value = get_mi(neural_activity, behavior_variable)

# Multi-dimensional feature
position_2d = MultiTimeSeries(
    np.stack([x_pos, y_pos]), 
    names=['x', 'y'],
    discrete=False
)

# MI with 2D position
mi_2d = get_mi(neural_activity, position_2d)

# Time-delayed MI
from driada.information import get_tdmi
mi_profile = get_tdmi(neural_activity, stimulus, delays=range(-10, 11))
optimal_delay = delays[np.argmax(mi_profile)]
```

## Method selection guide

- **Continuous-Continuous**: Use GCMI (fast) or KSG (robust)
- **Continuous-Discrete**: Use KSG or binning methods
- **High-dimensional**: Use GCMI with copula normalization
- **Small samples**: Use bias-corrected estimators

## Technical notes

- GCMI transforms variables to Gaussian via copula normalization (rank → empirical CDF → inverse normal), then estimates MI from the Pearson correlation of the transformed data (`-0.5 log(1 - r^2)`). Because copula normalization preserves rank order, this is equivalent to MI based on Spearman rank correlation of the original data. GCMI captures monotonic dependencies well but underestimates non-monotonic relationships (e.g., bell-shaped tuning, ROI-based selectivity), providing only a lower bound on true MI. Use KSG when non-monotonic selectivity is expected.
- KSG makes no distributional assumptions and detects arbitrary dependencies, but is computationally intensive. Use `mi_estimator_kwargs={"alpha": 0}` to disable the LNC correction for a large speedup with negligible accuracy loss.
- All estimators return values in bits (log₂)