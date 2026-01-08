# Experiment Module

## Overview

The experiment module provides the core `Experiment` class that serves as the central data structure in DRIADA. It manages neural recordings, behavioral features, and analysis results in a unified framework.

## Key Components

### Experiment Class
The main container for all experimental data:
- **Neural data**: Calcium imaging or spike trains
- **Dynamic features**: Time-varying behavioral variables
- **Static features**: Experimental metadata
- **Analysis results**: Stored selectivity and embedding results

### Data Loading
- `load_exp_from_aligned_data()` - Create experiment from numpy arrays
- `load_experiment()` - Load from various file formats
- `save_exp_to_pickle()` / `load_exp_from_pickle()` - Persistence

### Synthetic Data Generation

**Canonical Generator:**
- `generate_tuned_selectivity_exp()` - **Recommended** for all synthetic data with ground truth

**Convenience Wrappers:**
- `generate_mixed_population_exp()` - Mixed manifold + feature-selective populations
- `generate_synthetic_exp()` - Simple discrete/continuous feature selectivity

**Standalone Manifold Generators:**
- `generate_circular_manifold_exp()` - Head direction cells only
- `generate_2d_manifold_exp()` - Place cells only

**Time Series Generators:**
- `generate_pseudo_calcium_signal()` - Realistic calcium dynamics
- `generate_binary_time_series()` - Discrete events
- `generate_fbm_time_series()` - Fractional Brownian motion

### Supported Feature Names

| Feature Type | Name | Description |
|--------------|------|-------------|
| Head direction | `head_direction` | Circular [0, 2pi), von Mises tuning |
| 2D Position | `position_2d` | MultiTimeSeries (x, y), Gaussian place field |
| X marginal | `x` | 1D position [0, 1], Gaussian tuning |
| Y marginal | `y` | 1D position [0, 1], Gaussian tuning |
| Speed | `speed` | Derived from trajectory, sigmoid tuning |
| Discrete event | `event_0`, `event_1`, ... | Binary 0/1, threshold response |
| Continuous FBM | `fbm_0`, `fbm_1`, ... | Fractional Brownian motion, sigmoid tuning |

### Ground Truth Structure

All synthetic generators attach ground truth to `exp.ground_truth`:

```python
exp.ground_truth = {
    "expected_pairs": [(neuron_idx, feature_name), ...],  # Which neurons respond to which features
    "neuron_types": {neuron_idx: "group_name", ...},      # Population membership
    "tuning_parameters": {neuron_idx: {...}, ...},        # Per-neuron tuning params
    "population_config": [...]                             # Original config
}
```

### Spike Reconstruction
- `reconstruct_spikes()` - Main interface
- `wavelet_reconstruction()` - Wavelet-based deconvolution
- `threshold_reconstruction()` - Simple thresholding

## Example Usage

```python
from driada.experiment import load_exp_from_aligned_data

# Load real data
data = {
    'calcium': neural_data,  # Shape: (n_neurons, n_timepoints)
    'position': position,    # Shape: (n_timepoints,) or (2, n_timepoints)
    'speed': speed,         # Shape: (n_timepoints,)
}

exp = load_exp_from_aligned_data(
    data_source='my_lab',
    data=data,
    static_features={'fps': 30.0},
    reconstruct_spikes='wavelet'
)

# Generate synthetic data for testing (canonical generator)
from driada.experiment.synthetic import generate_tuned_selectivity_exp

population = [
    {"name": "hd_cells", "count": 20, "features": ["head_direction"]},
    {"name": "place_cells", "count": 20, "features": ["position_2d"]},
    {"name": "event_cells", "count": 10, "features": ["event_0"]},
]

exp = generate_tuned_selectivity_exp(
    population=population,
    duration=600,
    fps=20.0,
    n_discrete_features=2,
    seed=42,
)

# Access ground truth for validation
ground_truth = exp.ground_truth
print(f"Expected pairs: {len(ground_truth['expected_pairs'])}")

# Access data
calcium_data = exp.calcium.data  # MultiTimeSeries object
head_dir = exp.dynamic_features['head_direction'].data
```

## Working with Experiments

```python
# Run INTENSE analysis
results = compute_cell_feat_significance(exp)

# Get selective neurons
significant = exp.get_significant_neurons()

# Apply dimensionality reduction directly
embedding = exp.calcium.get_embedding(method='umap', n_components=2)
```

## Data Format

The standard data format for neural recordings is:
- **Shape**: `(n_neurons, n_timepoints)`
- **Type**: Continuous (calcium) or discrete (spikes)
- Features are automatically classified as discrete/continuous based on unique values