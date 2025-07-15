# DRIADA

**Dimensionality Reduction for Integrated Activity Data** - A unified framework bridging single-neuron selectivity analysis with population-level dimensionality reduction for biological and artificial neural systems.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/driada.svg)](https://pypi.org/project/driada/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Vision

DRIADA creates a seamless bridge between understanding individual neurons and population-level neural dynamics. Our framework enables researchers to:

1. **Identify** which neurons encode specific variables (using INTENSE)
2. **Extract** collective latent variables from population activity
3. **Connect** single-cell selectivity to population manifolds
4. **Interpret** how neural populations represent information

### The DRIADA Workflow

DRIADA uniquely combines single-neuron and population-level analyses in one framework. While traditional methods analyze neurons in isolation OR populations as a whole, DRIADA reveals how individual neural selectivity gives rise to collective representations.

```
Dimensionality reduction  ←  Population Activity  ←  Single Neurons  →  INTENSE
         ↓                                                                ↓
Latent Variables                                                 Individual Selectivity
         ↓                                                                ↓
          → → → → → → → → →  Integration Analysis ← ← ← ← ← ← ← ← ← ← ← ← ←
                                       ↓
         Connect single-cell selectivity to population-level variables
```

## Overview

DRIADA provides a comprehensive toolkit for analyzing both individual neural selectivity and collective population dynamics:

- **🔍 Individual Analysis**: Discover which neurons encode specific behavioral variables using information theory
- **🌐 Population Analysis**: Extract latent variables and manifolds from neural population activity
- **🔗 Integrated Workflows**: Connect single-cell properties to population-level representations
- **🧪 Validation Tools**: Generate synthetic populations with known ground truth for algorithm testing

### Key Capabilities

🧠 **INTENSE Module** - Single Neuron Analysis
- Detect both linear and nonlinear relationships using mutual information
- Rigorous two-stage statistical testing with multiple comparison correction
- Handle temporal delays between neural activity and behavior
- Disentangle mixed selectivity when neurons respond to multiple variables

📊 **Population-Level Analysis** - Collective Neural Dynamics
- **Dimensionality Estimation**: Measure intrinsic dimensionality of neural manifolds
  - Linear methods: PCA-based dimension, effective rank
  - Nonlinear methods: k-NN dimension, correlation dimension
- **Dimensionality Reduction**: Extract latent variables from population activity
  - Classical: PCA, Factor Analysis
  - Manifold learning: Isomap, UMAP, diffusion maps
  - Specialized neural methods (coming soon)
- **Latent Variable Extraction**: Recover behavioral variables from neural populations
  - Extract circular variables (e.g., head direction)
  - Reconstruct spatial maps from place cell activity
  - Identify task-relevant population subspaces

🔗 **Integrated Analysis** - Bridging Scales
- Map single-cell selectivity to population manifolds
- Understand how individual neurons contribute to collective representations
- Visualize relationships between neural selectivity and population structure

🧪 **Synthetic Data Generation** - Algorithm Validation
- Generate populations with known ground truth:
  - Head direction cells on circular manifolds
  - Place cells on 2D/3D spatial manifolds
  - Mixed populations with manifold + feature-selective neurons
- Test and validate analysis methods before applying to real data
- Benchmark different algorithms on controlled datasets

**Perfect for:**
- 🧠 **Cognitive neuroscience**: Identify task-relevant neural subspaces and their dynamics
- 🤖 **AI interpretability**: Understand representations in artificial neural networks
- 🔬 **Systems neuroscience**: Bridge cellular and population-level descriptions

## Quick Start

### Installation

```bash
# Basic installation
pip install driada

# With GPU support (recommended for large datasets)
pip install driada[gpu]
```

### Getting Started with DRIADA

#### 1. Generate Synthetic Data for Testing

```python
import driada
import numpy as np

# Generate a population with head direction cells
exp = driada.generate_circular_manifold_exp(
    n_neurons=50,           # 50 head direction cells
    duration=600,           # 10 minutes of recording
    noise_level=0.1,        # 10% noise
    seed=42
)

# Or generate place cells in 2D environment
exp = driada.generate_2d_manifold_exp(
    n_neurons=64,           # 8x8 grid of place cells
    duration=900,           # 15 minutes of exploration
    environments=['env1']   # Single environment
)

# Or create mixed populations
exp = driada.generate_mixed_population_exp(
    n_neurons=100,
    manifold_type='circular',
    manifold_fraction=0.4,  # 40% manifold cells, 60% feature-selective
    duration=600
)
```

#### 2. Analyze Single-Neuron Selectivity (INTENSE)

```python
# Discover which neurons encode which variables
stats, significance, info, results = driada.compute_cell_feat_significance(
    exp,
    n_shuffles_stage1=100,    # Quick screening
    n_shuffles_stage2=1000,   # Rigorous validation
    verbose=True
)

# View results
significant_neurons = exp.get_significant_neurons()
print(f"Found {len(significant_neurons)} selective neurons")

# Visualize selectivity
if significant_neurons:
    neuron_id = list(significant_neurons.keys())[0]
    feature = significant_neurons[neuron_id][0]
    driada.intense.plot_neuron_feature_pair(exp, neuron_id, feature)
```

#### 3. Extract Population-Level Manifolds

```python
# Get neural activity matrix
neural_data = exp.calcium  # Shape: (n_neurons, n_timepoints)

# Estimate intrinsic dimensionality
from driada.dimensionality import nn_dimension, pca_dimension, effective_rank

intrinsic_dim = nn_dimension(neural_data.T, k=5)      # k-NN estimator
linear_dim = pca_dimension(neural_data.T, threshold=0.95)  # PCA 95% variance
eff_rank = effective_rank(neural_data.T)             # Effective rank

print(f"Intrinsic dimension: {intrinsic_dim:.2f}")
print(f"Linear dimension (95%): {linear_dim}")
print(f"Effective rank: {eff_rank:.2f}")

# Apply dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap

# Linear embedding
pca = PCA(n_components=2)
pca_embedding = pca.fit_transform(neural_data.T)

# Nonlinear manifold learning
isomap = Isomap(n_components=2, n_neighbors=10)
isomap_embedding = isomap.fit_transform(neural_data.T)

umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_embedding = umap_reducer.fit_transform(neural_data.T)
```

#### 4. Using Your Own Data

```python
# Load your neural recordings
calcium_traces = np.load('path/to/calcium_data.npy')  # Shape: (n_neurons, n_timepoints)

# Load behavioral variables
behavior_data = {
    'position_x': np.load('path/to/x_position.npy'),
    'position_y': np.load('path/to/y_position.npy'),
    'head_direction': np.load('path/to/head_direction.npy'),
    'speed': np.load('path/to/speed.npy')
}

# Create experiment object
exp = driada.Experiment(
    signature='MyExperiment',
    calcium=calcium_traces,
    dynamic_features=behavior_data,
    static_features={'fps': 20.0}  # 20 Hz sampling rate
)

# Follow steps 2-3 above for analysis
```

## Documentation & Examples

### 📚 Core Documentation
- **[INTENSE Module Guide](README_INTENSE.md)** - Complete neural selectivity analysis documentation
- **[API Reference](docs/)** - Detailed function and class documentation

### 🔬 Working Examples
- **[examples/basic_usage.py](examples/basic_usage.py)** - Minimal INTENSE analysis workflow
- **[examples/full_pipeline.py](examples/full_pipeline.py)** - Complete analysis with visualizations
- **[examples/mixed_selectivity.py](examples/mixed_selectivity.py)** - Advanced disentanglement analysis
- **[examples/extract_circular_manifold.py](examples/extract_circular_manifold.py)** - Population manifold extraction

### 📓 Interactive Notebooks
- **[notebooks/01_quick_start.ipynb](notebooks/01_quick_start.ipynb)** - Your first DRIADA analysis
- **[notebooks/02_understanding_results.ipynb](notebooks/02_understanding_results.ipynb)** - Interpreting INTENSE outputs
- **[notebooks/03_real_data_workflow.ipynb](notebooks/03_real_data_workflow.ipynb)** - Working with experimental data

### 🎯 Specialized Guides
1. **Single-Neuron Analysis**: Start with [README_INTENSE.md](README_INTENSE.md) for selectivity analysis
2. **Population Analysis**: Use [examples/extract_circular_manifold.py](examples/extract_circular_manifold.py) for manifold extraction
3. **Interactive Learning**: Explore [notebooks/](notebooks/) for hands-on tutorials
4. **Synthetic Data**: Generate test populations with `driada.generate_*_manifold_exp()` functions
5. **Real Data**: Follow the "Using Your Own Data" section above

## Requirements

- Python 3.8+
- NumPy, SciPy, scikit-learn
- numba (for performance optimization)
- matplotlib, seaborn (for visualization)
- See [pyproject.toml](pyproject.toml) for complete list

## Installation from Source

```bash
git clone https://github.com/iabs-neuro/driada.git
cd driada
pip install -e .  # Editable installation
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/iabs-neuro/driada.git
cd driada

# Create conda environment
conda create -n driada python=3.9
conda activate driada

# Install in development mode
pip install -e .[gpu]

# Run tests
pytest
```

## Citation

If you use DRIADA in your research, please cite:

```bibtex
@software{driada2024,
  title = {DRIADA: Dimensionality Reduction for Integrated Activity Data},
  author = {Pospelov, Nikita and contributors},
  year = {2025},
  url = {https://github.com/iabs-neuro/driada}
}
```

## Support

- 📧 **Email**: pospelov.na14@physics.msu.ru
- 🐛 **Issues**: [GitHub Issues](https://github.com/iabs-neuro/driada/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/iabs-neuro/driada/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: DRIADA is actively developed. We recommend using the latest stable release for production work.