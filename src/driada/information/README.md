# Information Theory Module

## Overview

This module provides the information-theoretic foundation for DRIADA, implementing various measures of statistical dependency including mutual information (MI), entropy, and conditional information measures.

## Key Components

### Time Series Classes
- `TimeSeries` - Single variable time series
- `MultiTimeSeries` - Multi-dimensional time series (inherits from MVData)
  - Supports mixed continuous/discrete data
  - Automatic type detection
  - Direct access to dimensionality reduction methods

### Mutual Information Estimators

**Main Interface:**
- `get_mi()` - Automatic method selection based on data types
- `get_1d_mi()` - Single variable pairs
- `get_multi_mi()` - Multi-dimensional features

**Specialized Estimators:**
- **GCMI (Gaussian Copula MI)** - Fast, robust for continuous data
  - `mi_gg()` - Gaussian to Gaussian
  - `gcmi_cc()` - Continuous to continuous
  - `mi_model_gd()` - Model-based for discrete outcomes

- **KSG (k-Nearest Neighbors)** - Non-parametric, no assumptions
  - `nonparam_mi_cc()` - Continuous-continuous
  - `nonparam_mi_cd()` - Continuous-discrete
  - `nonparam_mi_dc()` - Discrete-continuous

### Advanced Measures
- `conditional_mi()` - MI conditioned on third variable
- `interaction_information()` - Three-way interactions
- `get_tdmi()` - Time-delayed mutual information

## Example Usage

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

## Method Selection Guide

- **Continuous-Continuous**: Use GCMI (fast) or KSG (robust)
- **Continuous-Discrete**: Use KSG or binning methods
- **High-dimensional**: Use GCMI with copula normalization
- **Small samples**: Use bias-corrected estimators

## Technical Notes

- GCMI assumes monotonic relationships (captures all dependencies after copula transform)
- KSG makes no assumptions but is computationally intensive
- All estimators return values in bits (log₂)