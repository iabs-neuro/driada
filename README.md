# DRIADA

**Dimensionality Reduction for Integrated Activity Data** - A unified framework bridging single-neuron selectivity analysis with population-level dimensionality reduction for biological and artificial neural systems.

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/driada.svg)](https://pypi.org/project/driada/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/iabs-neuro/driada/actions/workflows/tests.yml/badge.svg)](https://github.com/iabs-neuro/driada/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/iabs-neuro/driada/branch/main/graph/badge.svg)](https://codecov.io/gh/iabs-neuro/driada)

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

## Installation

```bash
# Basic installation
pip install driada

# With GPU support (recommended for large datasets)
pip install driada[gpu]
```

## Quick Start

For complete code examples, tutorials, and API documentation, please visit the **[official documentation](https://driada.readthedocs.io)**.

## ⚠️ WARNING: Pre-Release Version

**DRIADA is currently in pre-release stage (v0.x.x) and will be finalized to v1.0 soon.**

Until the stable v1.0 release:
- 📚 **Documentation takes precedence** over example code
- 🔧 Examples and notebooks may be incomplete or broken
- 🚧 API may undergo changes
- 📖 Please refer to the [official documentation](https://driada.readthedocs.io) for the most up-to-date information

## Documentation

📖 **[Official Documentation](https://driada.readthedocs.io)** - Complete API reference, tutorials, and guides

### Additional Resources
- **[INTENSE Module Guide](README_INTENSE.md)** - Neural selectivity analysis documentation
- **[GitHub Issues](https://github.com/iabs-neuro/driada/issues)** - Report bugs or request features

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