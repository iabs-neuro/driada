# DRIADA

**Dimensionality Reduction for Integrated Activity Data** - A comprehensive Python library for analyzing neural activity patterns and their relationships with behavior.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/driada.svg)](https://pypi.org/project/driada/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

DRIADA is a powerful computational framework designed for neuroscientists and researchers working with neural recordings. It provides state-of-the-art tools for:

- **Neural Selectivity Analysis**: Discover which neurons encode specific behavioral variables using information theory
- **Dimensionality Reduction**: Apply cutting-edge techniques to understand population-level neural representations
- **Integrated Analysis**: Seamlessly work with calcium imaging, electrophysiology, and behavioral data

### Key Features

🧠 **INTENSE Module** - Information-Theoretic Evaluation of Neuronal Selectivity
- Detect both linear and nonlinear relationships between neurons and behavior
- Rigorous statistical testing with multiple comparison correction
- Handle temporal delays between neural activity and behavior
- Analyze mixed selectivity when neurons respond to multiple variables

📊 **Dimensionality Reduction Suite**
- Multiple algorithms: PCA, UMAP, diffusion maps, and more
- Specialized methods for neural data characteristics
- Interactive visualization tools

🔬 **Built for Real Research**
- Handle large-scale recordings (100s of neurons, hours of data)
- Parallel processing for computational efficiency
- Robust to common experimental artifacts
- Comprehensive validation and testing

## Why DRIADA?

Traditional correlation-based methods miss nonlinear relationships that are common in neural data. DRIADA's information-theoretic approach captures **any** statistical dependency between neural activity and behavior, making it ideal for:

- Place cell identification in navigation experiments
- Sensory encoding analysis
- Decision-making and cognitive studies
- Brain-computer interface development
- Computational model validation

## Quick Start

### Installation

```bash
# Basic installation
pip install driada

# With GPU support (recommended for large datasets)
pip install driada[gpu]
```

### Example: Analyzing Neural Selectivity

```python
import driada
import numpy as np

# Load your neural recordings
calcium_traces = np.load('path/to/calcium_data.npy')  # Shape: (n_neurons, n_timepoints)

# Load behavioral variables
behavior_data = {
    'position_x': np.load('path/to/x_position.npy'),
    'position_y': np.load('path/to/y_position.npy'),
    'speed': np.load('path/to/speed.npy'),
    'trial_type': np.load('path/to/trial_labels.npy')
}

# Create experiment object
exp = driada.Experiment(
    signature='MyExperiment',
    calcium=calcium_traces,
    dynamic_features=behavior_data,
    static_features={'fps': 20.0}  # 20 Hz sampling rate
)

# Discover neural selectivity using INTENSE
stats, significance, info, results = driada.compute_cell_feat_significance(
    exp,
    n_shuffles_stage1=100,    # Quick screening
    n_shuffles_stage2=1000,   # Rigorous validation
    verbose=True
)

# View results
significant_neurons = exp.get_significant_neurons()
print(f"Found {len(significant_neurons)} neurons with significant selectivity")

# Visualize a neuron's selectivity
if significant_neurons:
    neuron_id = list(significant_neurons.keys())[0]
    feature = significant_neurons[neuron_id][0]
    driada.intense.plot_neuron_feature_pair(exp, neuron_id, feature)
```

## Documentation

- 📚 **[Full Documentation](docs/)** - Comprehensive guides and API reference
- 🚀 **[Quick Start Guide](README_INTENSE.md#quick-start)** - Get running in 5 minutes
- 📓 **[Interactive Notebooks](notebooks/)** - Jupyter tutorials with examples
- 🔬 **[Example Scripts](examples/)** - Complete analysis workflows

### Key Resources

1. **[INTENSE Module Guide](README_INTENSE.md)** - Detailed documentation for neural selectivity analysis
2. **[Examples Directory](examples/)** - Real-world usage patterns:
   - `basic_usage.py` - Minimal working example
   - `full_pipeline.py` - Complete analysis workflow
   - `mixed_selectivity.py` - Advanced disentanglement analysis
3. **[Notebooks](notebooks/)** - Interactive tutorials:
   - `01_quick_start.ipynb` - Your first analysis
   - `02_understanding_results.ipynb` - Interpreting outputs
   - `03_real_data_workflow.ipynb` - Working with experimental data

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
  year = {2024},
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