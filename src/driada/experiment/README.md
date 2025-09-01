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

**Manifold Generators:**
- `generate_circular_manifold_exp()` - Head direction cells
- `generate_2d_manifold_exp()` - Place cells in 2D
- `generate_3d_manifold_exp()` - 3D spatial cells
- `generate_mixed_population_exp()` - Mixed selectivity populations

**Time Series Generators:**
- `generate_pseudo_calcium_signal()` - Realistic calcium dynamics
- `generate_binary_time_series()` - Discrete events
- `generate_fbm_time_series()` - Fractional Brownian motion

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

# Generate synthetic data for testing
from driada.experiment import generate_circular_manifold_exp

exp = generate_circular_manifold_exp(
    n_neurons=50,
    duration=600,
    fps=20.0,
    noise_std=0.1
)

# Access data
calcium_data = exp.calcium.data  # MultiTimeSeries object
position = exp.dynamic_features['position_circular'].data
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