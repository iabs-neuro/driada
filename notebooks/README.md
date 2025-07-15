# DRIADA INTENSE Interactive Tutorials

Welcome to the DRIADA INTENSE interactive notebook tutorials! These Jupyter notebooks will guide you through using INTENSE for neuronal selectivity analysis.

## 📚 Available Notebooks

### 1. [01_quick_start.ipynb](01_quick_start.ipynb) - Quick Start (5 minutes)
- **Goal**: Get you analyzing neuronal selectivity in 5 minutes
- **You'll learn**:
  - How to generate synthetic neural data
  - Running basic INTENSE analysis
  - Interpreting significance results
  - Creating your first visualizations
- **Perfect for**: First-time users who want immediate results

### 2. [02_understanding_results.ipynb](02_understanding_results.ipynb) - Understanding Results
- **Goal**: Deep understanding of INTENSE outputs and statistics
- **You'll learn**:
  - Navigating the results data structures
  - Understanding statistical measures (MI, p-values, delays)
  - Creating custom visualizations
  - Analyzing temporal dynamics
  - Exporting results for publication
- **Perfect for**: Users who want to fully understand their analysis

### 3. [03_real_data_workflow.ipynb](03_real_data_workflow.ipynb) - Real Data Workflow
- **Goal**: Apply INTENSE to your own neural recordings
- **You'll learn**:
  - Data formatting and preparation
  - Creating Experiment objects from various sources
  - Quality control and validation
  - Advanced analysis workflows
  - Performance optimization for large datasets
- **Perfect for**: Researchers ready to analyze their own data

## 🚀 Getting Started

### Prerequisites
```bash
# Ensure you have Jupyter installed
pip install jupyter notebook

# Optional: For better notebook experience
pip install jupyterlab
```

### Running the Notebooks

1. **Navigate to the notebooks directory**:
   ```bash
   cd path/to/driada2/notebooks
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   # or for JupyterLab:
   jupyter lab
   ```

3. **Open a notebook** and follow along!

### Important Notes

- These notebooks use relative imports to access DRIADA without installation
- All notebooks are self-contained and use synthetic data
- Run cells in order for best results
- Restart kernel between notebooks to avoid conflicts

## 📊 What You'll Accomplish

After completing these tutorials, you'll be able to:
- ✅ Run INTENSE analysis on any neural dataset
- ✅ Identify which neurons encode specific behaviors
- ✅ Understand and interpret all statistical outputs
- ✅ Create publication-quality visualizations
- ✅ Handle real-world data challenges
- ✅ Optimize analysis for large-scale datasets

## 💡 Tips for Success

1. **Start with notebook 01** even if you're experienced
2. **Run all code cells** to see outputs
3. **Experiment with parameters** to understand their effects
4. **Check the examples/ directory** for more advanced use cases
5. **Read docstrings** for detailed function information

## 🆘 Getting Help

- **Documentation**: See [README_INTENSE.md](../README_INTENSE.md)
- **Examples**: Check the [examples/](../examples/) directory
- **Issues**: Report problems at the project repository

## 🎯 Next Steps

After completing these notebooks:
1. Apply INTENSE to your own data
2. Explore advanced features like mixed selectivity analysis
3. Read the mathematical framework in README_INTENSE.md
4. Check out the Python examples for production code patterns

Happy analyzing! 🧠✨