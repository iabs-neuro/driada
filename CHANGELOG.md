# Changelog

All notable changes to DRIADA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-XX-XX

**🎉 First stable release with JOSS publication**

### Added
- **JOSS Submission Materials**
  - Academic paper (`paper.md`) with comprehensive literature review covering information-theoretic toolboxes (NIT, MINT, FRITES), calcium imaging tools (CaImAn, Suite2p), and dimensionality reduction frameworks
  - Complete bibliography (`paper.bib`) with 38 references spanning neuroscience methodology, information theory, and manifold learning
  - Publications documentation (`PUBLICATIONS.md`) listing 8 research papers using DRIADA across biological and artificial neural systems

- **Project Governance**
  - Contribution guidelines (`CONTRIBUTING.md`) with development setup, testing procedures, and code style standards
  - Community standards (`CODE_OF_CONDUCT.md`) establishing inclusive collaboration practices
  - Machine-readable citation metadata (`CITATION.cff`) with synchronized author database (`AUTHORS.yaml`)

- **Reproducibility**
  - Configurable random seed parameter for graph construction (`g_params['seed']`) ensuring reproducible manifold learning experiments across runs

### Changed
- **FFT Code Quality, Optimization, and Refactoring**
  - **New Modules**: Created dedicated FFT acceleration modules for better organization:
    - `info_fft.py` (1,130 lines): All FFT-accelerated MI estimators
    - `info_fft_utils.py` (256 lines): Shared utilities and helpers
  - **Performance Improvements**:
    - Replaced `fft`/`ifft` with `rfft`/`irfft` for real inputs → **50% speedup** in `mi_cd_fft` and `compute_mi_mts_discrete_fft`
    - Fixed memory waste in `compute_mi_mts_fft` and `compute_mi_mts_mts_fft` → **100-1000x memory reduction** when nsh << n
  - **Code Quality**:
    - Unified bias correction using `scipy.special.psi` across all functions
    - Standardized regularization thresholds (dimension-specific: 1e-10, 1e-15, 1e-20)
    - Eliminated code duplication via shared helpers (~40% reduction in FFT-related code)
    - Reduced `info_base.py` from 3,246 to 2,311 lines (~30% reduction)
    - Reduced `entropy.py` from 500 to 378 lines
  - **Backward Compatibility**: All public APIs unchanged via re-exports in `info_base.py` and `entropy.py`

- **Major Code Refactoring**
  - Extracted monolithic `neuron.py` (3,807 lines) into focused, maintainable modules:
    - `calcium_kinetics.py` (237 lines) - Double-exponential kernel modeling and spike-to-calcium reconstruction
    - `signal_preprocessing.py` (97 lines) - Signal validation, normalization, and quality checks
    - `event_detection.py` (536 lines) - Wavelet and threshold-based calcium transient detection
  - Maintained 100% backward compatibility via static method delegation pattern

- **Code Quality**
  - Applied Black formatter (line-length 100) and isort (Black profile) across entire codebase (177 files, 8,762 insertions/9,219 deletions)
  - Enhanced documentation consistency and fixed high-priority issues flagged by automated verification tools

### Fixed
- Documentation cross-reference validation issues
- Missing docstring parameters in several modules

---

## [0.7.2] - 2026-01-21

**🔧 Batch processing CLI and disentanglement analysis enhancements**

### Performance
- **Discrete-Discrete FFT Vectorization** - 50-100x speedup for `mi_dd_fft` computation via vectorized operations
- **FFT Memory Optimization** - Up to 200x memory reduction in FFT modules through improved buffer management
- **MI Value Caching** - Pre-computed MI values cached in FFT cache for 4-5x speedup in repeated lookups
- **Disentanglement Parallelization** - MI lookup tables and parallel FFT cache; parallelized per-neuron processing
- **IPC Overhead Reduction** - Split `fft_cache` per worker to reduce inter-process communication in parallel jobs
- **Feature Copula Pre-computation** - Pre-compute copula cache before parallel loop to avoid redundant computation
- **Calcium Signal Vectorization** - Vectorized convolution for spike-to-calcium signal reconstruction

### Features
- **Batch Processing CLI**
  - `--dir` option for processing multiple experiment files in a directory
  - `--skip-computed` flag to resume interrupted batch runs, loading existing stats from CSV
  - `--parallel-backend` option with global `PARALLEL_BACKEND` config (loky, threading, multiprocessing)
  - Backend-specific parallel config with `pre_dispatch` tuning for memory control
  - Timeout wrapper for autonomous batch processing using `subprocess.Popen` polling
  - Incremental batch summary saving after each file (fault tolerance)
  - Automatic cache clearing for memory management in long-running jobs
  - Upgraded to joblib 1.5.0 with `idle_worker_timeout` for worker cleanup

- **Disentanglement Analysis**
  - Composable pre-filters for disentanglement analysis (spatial, temporal, threshold-based)
  - Filter examples added to `run_intense_analysis` script
  - Disentangled stats and significance output tables for publication-ready results

- **I/O & Data Loading**
  - New INTENSE I/O module using NPZ format (removes HDF5 dependency)
  - Tool for loading synchronized experiment `.npz` files
  - Experiment configs for `selectivity_dynamics` package

- **Profiling**
  - Internal timing profiling for INTENSE pipelines
  - Timing instrumentation for FFT cache building

### Fixed
- **Windows Process Handling** - Changed joblib backend to `loky` to prevent CMD hanging; use `Popen` polling to prevent timeout wrapper freeze
- **Memory Leaks** - Added `gc.collect()` after each experiment to prevent memory accumulation in batch processing
- **Zero Column Errors** - Use `aggregate_multiple_ts` in exp_build to add noise and avoid degenerate feature matrices
- **Disentanglement Robustness** - Improved numerical stability with spatial filter fallbacks
- **Google API Warning** - Suppressed spurious authentication warning
- **Test Stability** - Relaxed flaky test tolerances for CI reliability

### Changed
- Consolidated IABS filename parsing into `driada.utils.naming` module
- Removed HDF5 dependency from INTENSE results (replaced with NPZ format)

---

## [0.7.1] - 2026-01-17

**🚀 Enhanced performance and data loading capabilities**

### Performance
- **MTS-MTS FFT Acceleration** - 3-200x speedup for mutual information computation between MultiTimeSeries pairs (d1+d2 ≤ 6). Uses FFT-based cross-covariance computation with block determinant formula via Schur complement. Enables efficient analysis of multivariate neural dynamics (e.g., 2D position vs 2D neural trajectory). Complexity: O(d₁ × d₂ × n log n + nsh × d³).

- **Memory Optimization for INTENSE** - New `store_random_shifts` parameter (default: False) reduces memory usage by ~400MB for typical datasets (N=500 neurons, M=20 features, nsh=10,000). Random shift indices are not persisted unless explicitly requested, while preserving all MI values for analysis. Added `memory_usage()` method to `IntenseResults` for diagnostics.

### Features
- **Experiment Building Enhancements**
  - `aggregate_features` parameter in `load_exp_from_aligned_data()` enables combining multiple 1D features into MultiTimeSeries before building Experiment objects (e.g., `{('x', 'y'): 'position'}`)
  - Reserved key filtering prevents neural data keys (`calcium`, `spikes`, `sp`, `asp`, `reconstructions`) and metadata (`_metadata`, `_sync_info`) from being treated as behavioral features
  - Improves workflow for analyzing relationships between position/velocity and neural activity

### Fixed
- **Critical FFT Conjugation Bug** - Fixed incorrect FFT cross-correlation direction in `compute_mi_mts_mts_fft` that caused wrong MI values at non-zero shifts. Discovered during comprehensive verification testing. Changed from `conj(fft_x1) * fft_x2` to `fft_x1 * conj(fft_x2)` to correctly match `np.roll` semantics.
- **Documentation** - Removed invalid cross-references for missing functions and user guide sections
- **Test Stability** - Fixed flaky CI tests by adding proper imports, fixed random seeds, and increased connectivity parameters in manifold tests

### Added
- Comprehensive test suite for MTS-MTS FFT (27 unit tests + 3 integration tests) verifying correctness at all shifts with rtol=1e-7
- Benchmark script (`tools/benchmark_mts_mts_fft.py`) demonstrating speedup characteristics across different parameter regimes (n=1,000 to n=60,000, nsh=100 to nsh=10,000)

---

## [0.7.0] - 2026-01-13

**🚀 Major performance and stability release**

### Performance
- **FFT Acceleration for INTENSE** - 10-100x speedup through FFT-based mutual information computation across shuffle loops, delay calculation, and discrete-continuous MI operations. Unified FFT pipeline with intelligent caching.

### Critical Fixes
- **Cache Collision Bug** - Fixed critical bug where all neurons returned identical MI values due to generic TimeSeries naming causing FFT cache collisions. Implemented comprehensive naming strategy ensuring unique cache keys.

### Added
- **Zero-Inflated Gamma (ZIG)** distribution for improved MI null hypothesis testing
- **Tuned Selectivity Generator** with disentanglement analysis metrics for INTENSE pipeline
- **Configurable Random Seeds** for graph construction ensuring reproducible manifold learning experiments
- **Publication Framework Enhancements** - Extended `StylePreset` with sizing parameters and accessor methods

### Changed
- **Synthetic Data Module** - Consolidated all generators into unified `generators.py` with standardized signatures, explicit kwargs, and improved reproducibility
- **Examples Updated** - Validated and improved 10+ examples including complete RSA rewrite using tuned selectivity generator, enhanced Network Analysis with spectral properties, and updates across INTENSE, dimensionality reduction, and manifold examples
- **Code Quality** - Applied Black formatter (line-length 100) across codebase

### Statistics
- 215 files changed, 20,762 insertions, 14,960 deletions
- 41 new tests added (FFT cache, multicomp correction, MTS FFT)
- Test coverage: `intense_base` 81.85%, `info_base` 88.48%
- All 1,681 tests passing

---

## [0.6.6] - 2025-12-XX

### Added
- **Dependency Management**
  - Lazy torch imports with conditional GPU support for wavelet detection
  - Reduces import overhead and enables CPU-only environments without requiring PyTorch installation
  - GPU toggle allows users to control whether torch-based wavelet detection uses CUDA acceleration

### Changed
- **Distribution Optimization**
  - Excluded images and data files from PyPI distribution package
  - Reduced tarball size to <1 MB (was >10 MB with example data)

### Fixed
- **Test Stability**
  - Increased shuffle iterations to 1000 for 2D manifold tests (previously failed intermittently with default 100 shuffles)
  - Ensures consistent statistical power in null hypothesis testing

---

## [0.6.5] - 2025-12-XX

### Added
- **Wavelet Optimization**
  - Batch optimization mode for wavelet-based event detection
  - Vectorized processing of multiple neurons simultaneously (10-50× speedup for large datasets)
  - New reconstruction workflow balancing accuracy and computational efficiency

### Fixed
- **Dimensionality Reduction**
  - Relaxed diffusion maps variance test to check total variance preservation trend
  - Previously failed due to strict per-eigenvalue thresholds; now validates overall manifold structure

### Changed
- Resolved merge conflicts from parallel feature development branches

---

## [0.6.4] - 2025-11-XX

### Added
- **Publication Tools**
  - Multi-panel figure framework for generating journal-quality visualizations
  - Precise physical sizing control (inches/cm) for manuscript submissions
  - Customizable subplot layouts and spacing

- **Testing**
  - Comprehensive test suites for quantum metrics, manifold analysis, wavelet detection, and neuron quality assessment
  - Improved coverage for edge cases in signal processing

### Changed
- **Spike Reconstruction**
  - Major refactor introducing iterative threshold detection algorithm
  - Alternates between threshold estimation and spike extraction until convergence
  - Balances sensitivity (detecting weak events) and specificity (avoiding false positives)

### Fixed
- **Statistical Testing**
  - Eliminated spurious correlations in INTENSE null hypothesis test
  - Proper surrogate generation now preserves temporal autocorrelation structure
  - Ensures accurate Type I error control during Holm-Bonferroni correction

- **Event Detection**
  - Unified SNR calculation between wavelet and threshold methods (previously reported inconsistent quality metrics)
  - Metrics (amplitude, correlation, SNR) now compatible with both detection approaches
  - Threshold computation uses original signal instead of residuals (prevents threshold drift during iteration)

- **Kinetics Optimization**
  - Fixed silent failures for signals with long decay time constants (t_off > 200 frames)
  - Added safety checks capping kernel length at 2000 frames
  - Warns users about suspicious t_off values incompatible with typical calcium indicators

### Improved
- Test tolerance relaxation for numerical edge cases
- .gitignore specificity for development artifacts

---

## [0.6.1] - 2025-10-XX

### Added
- **Frame Rate Adaptivity**
  - Complete FPS (frames per second) adaptivity implementation
  - Automatic time constant conversion for varying imaging frequencies
  - Handles datasets from 5 Hz (slow volumetric) to 100+ Hz (fast planar) imaging

- **Development Environment**
  - Conda environment specification (`environment.yml`) for reproducible setup
  - Pins exact versions of scientific Python stack (numpy, scipy, scikit-learn, etc.)

### Fixed
- **Kinetics Optimization**
  - Event regions now preserved during rise/decay time constant fitting
  - Previously could lose detected calcium transients when re-optimizing parameters
  - Added regression tests to prevent future breakage

### Changed
- Added Neuron class usage example demonstrating end-to-end calcium imaging workflow
- Cleaned up accidentally committed debug and analysis artifacts

---

## [0.6.0] - 2025-09-XX

### Added
- **Kinetics Analysis System**
  - Automated optimization of calcium indicator rise/decay time constants (τ_rise, τ_off)
  - Fits double-exponential kernel to observed calcium dynamics
  - Uses RMSE minimization between observed and reconstructed calcium traces

- **Event Quality Metrics**
  - Amplitude extraction from wavelet coefficients and threshold crossings
  - SNR (Signal-to-Noise Ratio) calculation for transient quality assessment
  - Template correlation metrics for validating event shape consistency
  - Deconvolution support for spike inference from calcium traces

### Fixed
- **Critical Reconstruction Bugs**
  - Amplitude extraction from wavelet coefficients (was systematically underestimating event strength)
  - SNR interface unification across wavelet and threshold detection methods
  - Kinetics parameter propagation through reconstruction pipeline

### Changed
- Completed major code quality fixes preparing for public pre-release
- Unified SNR calculation interface

---

## [0.5.1] - 2025-08-XX

### Added
- **Cross-Validation Framework**
  - Leave-one-out (LOO) dimensionality reduction analysis
  - Validates manifold structure stability when systematically excluding neurons
  - Integration with INTENSE for selectivity-guided validation

- **Visualization Enhancements**
  - `make_beautiful()` function for publication-quality plot styling
  - Customizable legend positioning (upper/lower × left/right, outside options)
  - Improved default aesthetics (font sizes, line widths, color schemes)

- **Data Management**
  - Google Drive share link support for collaborative data sharing
  - `router_source` parameter in experiment loading for flexible data backends
  - Support for multiple simultaneous data sources in IABS router

- **Analysis Features**
  - Downsampling support for LOO analysis on large datasets (reduces computation 10-100×)
  - Embedding visualization with automatic dimension detection

### Fixed
- Quality metrics validation for calcium imaging pipelines
- Test suite stability improvements

---

## [0.4.0] - 2025-07-XX

### Added
- **CI/CD Infrastructure**
  - GitHub Actions workflows for automated testing
  - Codecov integration for coverage tracking
  - Test coverage requirements: 80%+ for merging to main branch

- **Comprehensive Testing**
  - Coverage sprint achieving 80%+ across all major modules:
    - `information/`: 88.7% (INTENSE, GCMI, entropy functions)
    - `dim_reduction/`: 87.3% (PCA, Isomap, UMAP, Diffusion Maps)
    - `network/`: 96.2% (graph construction, topology metrics)
    - `experiment/`: extensive exp_base.py and neuron.py tests
  - Import tests ensuring all modules load without errors
  - Edge case handling for numerical stability

- **Dimensionality Reduction**
  - **Diffusion Maps**: complete eigen-decomposition implementation with heat kernel affinities
  - **Intrinsic dimension estimation**: correlation dimension, k-NN fractal dimension
  - **Graph preprocessing**: automatic defaults (e.g., UMAP requires connected graph, Isomap handles disconnected)
  - **API modernization**: `dr_sequence()` replacing legacy `dr_series` with better error messages

- **Information Theory**
  - **KSG estimator**: Kraskov-Stögbauer-Grassberger k-NN mutual information as alternative to GCMI
  - **Estimator selection**: `mi_estimator` parameter in INTENSE pipelines ('gcmi' or 'ksg')
  - **Automatic LNC**: local non-uniformity correction alpha selection for KSG
  - **JIT optimization**: Numba compilation for GCMI functions (10-100× speedup for large datasets)

- **Network Analysis**
  - **Gromov hyperbolicity**: curvature calculation for network geometry
  - **Partial directions**: construct graphs using subset of features
  - **Topology metrics**: clustering coefficient, path lengths, community structure

- **Neural Network Models**
  - **Flexible autoencoders**: modular loss composition system
  - **Loss functions**: reconstruction MSE, orthogonality penalties, correlation losses
  - **Architecture options**: skip connections, batch normalization, dropout
  - **Time series support**: advanced type detection for temporal vs. static data

### Changed
- **API Improvements**
  - Dimensionality reduction functions accept `verbose` parameter (suppresses progress print statements)
  - Matrix utilities automatically branch between dense numpy and sparse scipy implementations
  - Simplified parameter passing with better validation

- **RSA Module**
  - Major refactoring improving code clarity
  - Better integration with dimensionality reduction workflows
  - Simplified correlation distance calculations

### Fixed
- **Critical Bugs**
  - Manifold metric numerical stability (eliminated NaN propagation)
  - ProximityGraph edge weight doubling in k-NN construction (was creating 2× heavier graphs)
  - Lost nodes calculation in graph giant component extraction
  - Zero columns in discrete attribute handling (now raises informative error)

- **Precision Issues**
  - Floating-point formatting in axis labels and tick marks
  - Matrix condition number checks before inversion

### Removed
- **Deprecated code**
  - `signals` module (functionality migrated to utils)
  - Legacy `dr_series` function (use `dr_sequence` instead)

---

## [0.2.0] - 2025-03-XX

### Added
- **Synthetic Data Generators**
  - **Mixed selectivity populations**: neurons responding to multiple task variables with ground truth
  - **3D spatial manifolds**: volumetric place cell populations for navigation studies
  - **2D spatial manifolds**: planar place fields with adjustable coverage and overlap
  - **Circular manifolds**: head direction cells with von Mises tuning curves
  - **Validation framework**: compare dimensionality reduction against known ground truth geometry

- **Dimensionality Estimation**
  - **Linear methods**:
    - PCA-based effective dimension (proportion of variance threshold)
    - Effective rank (entropy of eigenvalue distribution)
  - **Nonlinear methods**:
    - k-NN fractal dimension (scales with neighbor distance)
    - Correlation dimension (box-counting on embedding)
  - **Circular extraction**: identify ring-like manifolds from population activity

- **Documentation**
  - Interactive Jupyter notebooks demonstrating end-to-end workflows
  - README overhaul explaining DRIADA's vision: bridging single-neuron selectivity and population dynamics
  - Example gallery with 15+ analysis pipelines

### Changed
- **Project Scope**
  - Unified framework integrating INTENSE (single-neuron selectivity) with dimensionality reduction (population structure)
  - Renamed internal documentation from `_intense_backlog.md` to `_backlog.md` reflecting broader mission

### Fixed
- 3D spatial manifold generation stability (optimized firing rate parameters)
- Test numerical tolerance for nonlinear dimension estimators

---

## [0.1.x] - 2024-2025

**Initial development releases establishing core functionality**

### INTENSE Module (Information-Theoretic Evaluation of Neuronal Selectivity)

#### Statistical Framework
- **Gaussian Copula Mutual Information (GCMI)**: continuous estimator handling arbitrary distributions
- **Two-stage testing**:
  - Stage 1: rank-based non-parametric test (fast, conservative)
  - Stage 2: parametric gamma distribution fitting (precise p-values for significant relationships)
- **Multiple comparison correction**: Holm-Bonferroni controlling family-wise error rate
- **Optimal shift search**: detect delayed relationships (e.g., sensory → motor lag)
- **Mixed selectivity disentanglement**: separate unique vs. shared information using interaction information

#### Analysis Workflows
- Cell-cell interaction information (synergy, redundancy)
- Feature-feature joint encoding
- Single-cell selectivity for behavioral variables
- Parallel processing for large-scale multi-neuron recordings

### Dimensionality Reduction Framework

#### Classical Methods
- **PCA**: principal component analysis with variance-based selection
- **Factor Analysis**: probabilistic latent variable model with noise modeling

#### Manifold Learning
- **Isomap**: geodesic distance preservation via graph shortest paths
- **UMAP**: uniform manifold approximation with fuzzy topology
- **t-SNE**: t-distributed stochastic neighbor embedding for visualization
- **Diffusion Maps**: spectral embedding based on random walk diffusion

#### Graph Construction
- Heat kernel affinities with adaptive bandwidth
- k-NN and ε-ball proximity graphs
- Giant component extraction for disconnected graphs
- Configurable distance metrics (Euclidean, correlation, etc.)

### Calcium Imaging Pipeline

#### Event Detection
- **Wavelet-based**: multi-scale decomposition with adaptive thresholding
- **Threshold-based**: iterative baseline-crossing detection
- **Quality metrics**: amplitude, SNR, shape correlation

#### Signal Reconstruction
- Double-exponential kernel convolution (rise + decay dynamics)
- Spike-to-calcium forward model
- Parameter optimization for indicator kinetics

### Experiment Framework

#### Data Management
- **Unified container**: neural activity + behavioral variables + metadata
- **Multi-cell support**: analyze 10-1000+ neurons simultaneously
- **HDF5 serialization**: efficient storage for large datasets (>1 GB recordings)
- **Flexible loading**: support for various acquisition systems (Inscopix, Miniscope, 2-photon)

#### Integration
- Seamless INTENSE → DR pipeline: map selectivity onto population manifolds
- Synthetic data validation: test pipelines on ground truth before real data

### Infrastructure

#### Development
- Python package structure (setuptools/pip installable)
- Modular architecture: independent modules for information theory, DR, experiment handling
- Public API via `__init__.py` exports

#### Testing
- pytest framework with fixtures
- Unit tests for mathematical functions
- Integration tests for full workflows
- Doctests in function docstrings

#### Documentation
- Sphinx-based HTML documentation
- NumPy-style docstrings
- Examples embedded in docstrings
- Reference API documentation

---

