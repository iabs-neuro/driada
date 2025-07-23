# DRIADA Production-Ready TODO List

**DRIADA = Dimensionality Reduction for Integrated Activity Data**

A comprehensive library bridging element-wise and population-wise analysis for both biological and artificial neural systems.

## 🎯 CORE VISION & ROADMAP TO COMPLETION (2025-01-14)

**Mission**: Create a unified framework that seamlessly connects single-neuron selectivity analysis (INTENSE) with population-level dimensionality reduction, enabling researchers to understand both individual neural coding and collective neural representations.

### 🏗️ MAJOR MILESTONES FOR COMPLETION

1. **Augment Synthetic Data Generation** - Add manifold-based neural activity
   - Generate neural activity lying on circular manifolds (e.g., head direction cells)
   - Generate activity on 2D/3D spatial manifolds (e.g., place cells on grids)
   - Create mixed populations with both selective neurons and manifold structure
   - Add noise models reflecting biological/artificial constraints

2. **Complete Test Coverage** - Ensure robustness of all modules
   - Test dimensionality reduction algorithms (PCA, UMAP, diffusion maps)
   - Test dimensionality estimation methods
   - Verify integration between modules
   - Add performance benchmarks for large-scale data

3. **Create Latent Variable Extraction Examples** - Demonstrate core capabilities
   - Show how to extract circular variables from head direction cells
   - Demonstrate 2D spatial map extraction from place cells
   - Extract task-relevant variables from mixed populations
   - Compare different dimensionality reduction methods

4. **Build INTENSE → Latent Variables Pipeline** - Bridge element to population
   - Map individual neuron selectivity to population manifolds
   - Show how single-cell MI relates to population structure
   - Create workflow: neurons → selectivity → manifold → interpretation
   - Demonstrate on both biological and artificial neural data

5. **Comprehensive Examples & Documentation** - Enable widespread adoption
   - End-to-end workflows for common use cases
   - Comparison with traditional methods
   - Best practices for different data types
   - Performance optimization guides

6. **Update README & Documentation** - Complete the vision
   - Show full DRIADA capabilities (not just INTENSE)
   - Include population-level analysis examples
   - Demonstrate biological + artificial neural system analysis
   - Clear value proposition for integrated analysis

## 📋 DETAILED IMPLEMENTATION PLAN

### 📅 PRIORITY TIMELINE

**Phase 1 (Weeks 1-2): Foundation**
- Complete Milestone 1 (Synthetic Data) - Enable testing and examples
- Complete Milestone 2 (Test Coverage) - Ensure reliability

**Phase 2 (Weeks 3-4): Core Integration**  
- Complete Milestone 3 (Latent Extraction Examples) - Demonstrate capabilities
- Complete Milestone 4 (Pipeline) - Build the bridge

**Phase 3 (Weeks 5-6): Polish & Release**
- Complete Milestone 5 (Full Examples) - Enable adoption
- Complete Milestone 6 (Documentation) - Complete the vision

### MILESTONE 1: Augment Synthetic Data Generation ✅ COMPLETED (2025-01-20)

### MILESTONE 2: Complete Test Coverage ✅ SUBSTANTIALLY COMPLETED (2025-01-15)

**Remaining Tasks:**
- [ ] **Complete MVU Implementation (POST-RELEASE)**
  - [ ] Finish MVU (Maximum Variance Unfolding) implementation
  - [ ] Add comprehensive tests for MVU method
  - [ ] Integrate with dimensionality reduction suite
- [ ] **Optimize Slow Tests (POST-RELEASE)**
  - [ ] Reduce test_circular_manifold_reconstruction time (currently 40s)
  - [ ] Consider parallel test execution for manifold tests
  - [ ] Cache more intermediate results to speed up test suite
- [ ] **Expand Noise Models for Testing (TIED TO MILESTONE 1)**
  - [ ] Add Poisson spiking noise models
  - [ ] Test calcium indicator dynamics effects
  - [ ] Add motion artifact simulation and testing
- [ ] **Add Missing DR Method Tests**
  - [ ] Complete diffusion maps (dmaps) implementation and testing
- [ ] **Performance Monitoring (POST-RELEASE)**
  - [ ] Create benchmark suite to track performance over time
  - [ ] Add regression testing for manifold quality metrics
  - [ ] Monitor test execution times for performance regressions
- [x] **Test dimensionality reduction module** ✅ SUBSTANTIALLY COMPLETED (2025-01-15)
- [ ] **Test dimensionality estimation module**
  - [ ] Test effective dimensionality estimators
  - [ ] Validate on synthetic data with known dimensionality
  - [ ] Edge cases (high noise, sparse data)
- [x] **Enhance DR Method Configuration (PRE-RELEASE PRIORITY)** ✅ COMPLETED (2025-01-20)
- [x] **Test integration between modules** ✅ COMPLETED (2025-01-22)
  - [x] INTENSE → DR pipeline tests ✅
  - [x] Data flow validation ✅
  - [x] Memory efficiency tests ✅
- [x] **Standardize test data generation using fixtures** ✅ COMPLETED (2025-01-22)
  - [x] Convert ~28 tests to use pytest fixtures instead of direct calls
  - [x] Add spike_reconstruction_experiment fixture
  - [x] Update all conftest.py imports to use consistent paths
  - [x] Achieve ~25% reduction in redundant experiment generation
  - [x] Fix test failures from fixture migration
  - [x] Document legitimate direct call cases (3 remaining)
- [ ] **Remaining TODO items**:
    - [ ] Implement eps graph construction method in ProximityGraph
    - [ ] Add more sophisticated noise models for synthetic data
    - [ ] Create comprehensive DR method comparison on various manifolds
    - [ ] Document best practices for DR method selection
    - [ ] Warning for too thread-like graphs (too many temporal neighbors)
    - [ ] **CRITICAL BUG: Investigate false positives in INTENSE with pure random noise**
      - [ ] Test shows significant neurons with p-values < 1e-10 for random data
      - [ ] All significant neurons have rval=1.0000 (MI higher than ALL shuffles)
      - [ ] Issue appears systematic - same neurons flagged across different random seeds
      - [ ] Could be related to shuffle mask handling or MI calculation
      - [ ] Need to verify shuffling procedure is working correctly
    - [ ] **BUG: Fix compute_feat_feat_significance self-comparison error**
      - [ ] Error: "MI computation of a TimeSeries with itself is not allowed"
      - [ ] Occurs when computing feature-feature significance matrix
      - [ ] Diagonal should be masked but scan_pairs still attempts self-comparison
      - [ ] Solution: Create copies of TimeSeries objects or handle in scan_pairs
      - [ ] Related test: test_compute_feat_feat_significance in test_intense_pipelines.py

### MILESTONE 3: Create Latent Variable Extraction Examples ✅ COMPLETED (2025-07-18)
- [x] **examples/extract_circular_manifold.py** ✅ COMPLETED (2025-01-14)
  - [ ] **Known Issues & TODOs**:
    - [ ] Need better correspondence metric for circular manifold alignment
    - [ ] Add Procrustes analysis for optimal rotation/reflection
    - [ ] Consider geodesic distance preservation metrics
    - [ ] Add example with denoising pipeline before dimension estimation
    - [ ] Document recommended preprocessing for accurate estimates
- [x] **examples/extract_spatial_map.py** ✅ COMPLETED (2025-01-15)
- [x] **examples/extract_task_variables.py** - ✅ COMPLETED (2025-07-16)
  - [x] **TODO**: Investigate why metric_distr_type='norm' works better than 'gamma' for MI distributions ✅ COMPLETED (2025-07-16)
- [x] **Fix low mixed selectivity detection in synthetic data** - URGENT - ✅ COMPLETED (2025-01-16)
- [x] **Fix eff_dim correction failing with NaN/inf errors** - URGENT ✅ COMPLETED (2025-01-17)
- [x] **examples/compare_dr_methods.py** ✅ COMPLETED (2025-01-17)

### MILESTONE 4: Build INTENSE → Latent Variables Pipeline ✅ COMPLETED (2025-07-18)

**Remaining tasks:**
- [x] **Enhance comprehensive example for SelectivityManifoldMapper** ✅ COMPLETED (2025-01-23)
  - [x] `examples/selectivity_manifold_mapper_demo.py` - Enhanced with visual utils and metrics
  - [x] Show full workflow: data → INTENSE → embeddings → component selectivity
  - [x] **IMPROVEMENTS COMPLETED:**
    - [x] Replaced Isomap with Laplacian Eigenmaps (LE) for better manifold capture
    - [x] Fixed component selectivity detection (now showing many selective neurons)
    - [x] Added manifold preservation metrics (k-NN, trustworthiness, continuity, etc.)
    - [x] Integrated visual utilities for consistent visualization
    - [x] Added manifold quality assessment with bar and radar plots
    - [x] Using plot_component_selectivity_heatmap from visual utils
    - [x] Enhanced documentation with new features description
  - [ ] **Future improvements (optional):**
    - [ ] Use `multicomp_correction='holm'` for more rigorous INTENSE analysis
    - [ ] Add 3D visualization option for first 3 components
    - [ ] Include temporal trajectory animations on manifold
  - [x] Compare different DR methods (PCA, UMAP currently working)
  - [x] Visualize neuron selectivity to embedding components (heatmap created)
  - [x] Show functional organization analysis
  - [x] Demonstrate how embedding components relate to behavioral features
  - [x] Include command-line interface like intense_dr_pipeline.py
- [ ] **Implement pipeline functions**
  - [ ] `analyze_population_structure()`
  - [ ] `map_selectivity_to_manifold()`
  - [ ] `identify_functional_subspaces()`
- [ ] **Create visualization tools**
  - [ ] Plot selectivity on manifold
  - [ ] Interactive 3D visualizations
  - [ ] Temporal dynamics on manifold

**MILESTONE 4 STATUS**: SelectivityManifoldMapper implementation completed with full test coverage. Core pipeline example (intense_dr_pipeline.py) completed and working. Synthetic data bug fixed. Ready for comprehensive example creation and visualization tools.

### MILESTONE 5: Comprehensive Examples & Documentation
- [ ] **examples/full_pipeline_biological.py**
  - [ ] Load real neural data
  - [ ] Complete INTENSE analysis
  - [ ] Extract population manifold
  - [ ] Relate single cells to population
- [ ] **examples/full_pipeline_artificial.py**
  - [ ] Analyze RNN representations
  - [ ] Compare to biological data
  - [ ] Interpretability insights
- [ ] **Create tutorial notebooks**
  - [ ] `04_population_analysis.ipynb`
  - [ ] `05_integrated_pipeline.ipynb`
  - [ ] `06_advanced_workflows.ipynb`
- [ ] **Write best practices guide**
  - [ ] When to use which method
  - [ ] Parameter selection
  - [ ] Computational considerations

### MILESTONE 6: Update README & Documentation
- [x] **Expand main README.md** ✅ PARTIALLY COMPLETED (2025-01-14)
  - [ ] **REMAINING**: Add comparison table (DRIADA vs other tools) - needs completion of all milestones
  - [ ] **FUTURE**: Final polish after all capabilities are implemented
- [ ] **Create module-specific READMEs**
  - [ ] README_DIMENSIONALITY.md
  - [ ] README_INTEGRATION.md
  - [ ] README_SYNTHETIC.md
- [ ] **Update API documentation**
  - [ ] Complete docstrings
  - [ ] Generate API reference
  - [ ] Add mathematical background

## PREVIOUS CONTENT (INTENSE-specific):

INTENSE = Information-Theoretic Evaluation of Neuronal Selectivity
A toolbox to analyze individual neuronal selectivity to external patterns using mutual information and other metrics.

## CURRENT STATE ANALYSIS FOR NEW USERS (2025-01-12)

**✅ STRENGTHS (What works well):**
- Comprehensive theoretical documentation in README_INTENSE.md (excellent mathematical background)
- Clean top-level API in `src/driada/__init__.py` with intuitive function exports
- Robust synthetic data generation via `generate_synthetic_exp()` function
- Strong test coverage (84%) and all tests passing
- Production-grade codebase with proper module structure
- Advanced features: disentanglement analysis, mixed selectivity, optimal delay detection

**❌ REMAINING GAPS (Still blocking some user adoption):**
- ~~**No practical getting-started guide**~~ ✅ FIXED: README_INTENSE.md now has comprehensive Quick Start
- **No examples/ directory** - Users cannot see working demonstrations
- **No notebooks/ directory** - No interactive tutorials for exploration
- **Minimal main README.md** - Only shows installation, no project overview or value proposition
- ~~**No quick-start path**~~ ✅ FIXED: 5-minute success path now available in README_INTENSE.md
- ~~**No result interpretation guidance**~~ ✅ FIXED: Clear output examples and Experiment methods shown

**📊 UPDATED USER JOURNEY ANALYSIS:**
1. **Discovery**: User finds DRIADA → README.md still inadequate (needs improvement)
2. **Understanding**: User reads README_INTENSE.md → ✅ **NOW HAS QUICK START** with immediate practical entry
3. **First attempt**: User copies Quick Start code → ✅ **SUCCESS IN 5 MINUTES** with working example
4. **Success**: ✅ **IMMEDIATE PATH** - no expert knowledge required for basic functionality

**🎯 DEMO STATUS UPDATE:**
- ✅ **5-minute path from import to meaningful results** - ACHIEVED in README_INTENSE.md
- ✅ **Copy-paste examples that work immediately** - ACHIEVED with tested Quick Start code
- ✅ **Clear value demonstration with synthetic data** - ACHIEVED with generate_synthetic_exp()
- ✅ **Visual outputs that showcase INTENSE capabilities** - ACHIEVED with plot_neuron_feature_pair()

**🚀 DEMO READINESS: PHASE 1 COMPLETE** - Library can now be demonstrated successfully!

## 0. NEW USER COMFORT & ONBOARDING - URGENT PRIORITY FOR DEMO

### CRITICAL PRIORITY (Must have before library demonstration)
- [x] **Create 5-minute quick-start guide** - Essential for library demo ✅ COMPLETED (2025-01-12)

- [x] **Create examples/ directory with working demos** - Critical for user confidence ✅ COMPLETED (2025-01-12)

- [x] **Fix and enhance full_pipeline.py example** - PARTIALLY COMPLETED (2025-01-12)
  - [ ] Investigate why no neurons are selective for continuous features
  - [ ] Investigate why no neurons are selective for MultiTimeSeries features

- [x] **Fix plot_selectivity_heatmap visualization issues** - ✅ COMPLETED (2025-01-12)

- [x] **Update mixed_selectivity.py to use visual.py functions** - ✅ COMPLETED (2025-01-13)

- [x] **Create notebooks/ directory with interactive tutorials** ✅ COMPLETED (2025-01-17)

- [ ] **Improve main README.md** - Currently inadequate for new users (PARTIALLY COMPLETED 2025-01-14)
  - [ ] Add population-level analysis examples and documentation
  - [ ] Add integration/dimensionality reduction examples
  - [ ] Create comprehensive overview of all DRIADA modules

- [x] **Create beginner-friendly API examples in README_INTENSE.md** ✅ COMPLETED (2025-01-12)
  - [x] Current README has excellent theory but zero practical guidance
  - [x] Add "Quick Examples" section before detailed mathematical framework
  - [x] Show synthetic data generation and analysis in 3-5 lines
  - [x] Demonstrate result interpretation with sample outputs
  - [x] Include common troubleshooting tips
  
  **Implementation**: Added Quick Start section with copy-paste working code, positioned before Mathematical Framework

### High Priority
- [ ] **Add installation verification script** - Reduce user setup friction
  - [ ] Create `verify_installation.py` script
  - [ ] Test all core functionality works correctly
  - [ ] Provide clear success/failure feedback
  - [ ] Include performance benchmarks on synthetic data

## 1. Documentation & API Reference

### High Priority
- [ ] Write comprehensive module overview documentation explaining INTENSE methodology
- [ ] Create API reference for all public functions and classes
- [ ] Add detailed docstrings for all functions (many are missing or incomplete)
- [ ] Create usage examples and tutorials for common workflows
- [ ] Document the two-stage statistical testing approach
- [ ] Explain the shuffling methodology and statistical significance testing
- [ ] Add mathematical formulations for all metrics (MI, correlation, etc.)
- [ ] Create a glossary of terms (selectivity, mutual information, shuffling, etc.)

### Medium Priority
- [ ] **Add circular feature type support** - Automatic cos/sin representation
  - [ ] Add `feature_type` parameter to TimeSeries (default='linear', options: 'linear', 'circular')
  - [ ] Automatically create cos/sin MultiTimeSeries for circular features in Experiment
  - [ ] Update INTENSE pipeline to detect and handle circular features automatically
  - [ ] Document best practices for circular variables (angles, phases, directions)
  - [ ] Add unit tests for automatic circular feature conversion
- [ ] **Create circular variable analysis example** - Demonstrate cos/sin superiority
  - [ ] Create `examples/circular_variables.py` showing head direction analysis
  - [ ] Compare linear vs circular representation detection rates
  - [ ] Show visualization of Von Mises tuning curves
  - [ ] Include other circular examples: phase, time-of-day, angles
  - [ ] Document the mathematical reasoning behind cos/sin representation
  - [ ] Show performance metrics: detection rate, MI values, computation time
- [ ] Add type hints to all function signatures
- [ ] Document performance characteristics and computational complexity
- [ ] Create best practices guide for parameter selection
- [ ] Add troubleshooting guide for common issues

## 2. Code Quality & Architecture

### High Priority
- [x] **Add proper __init__.py exports for public API** ✅ COMPLETED (2025-01-11)
- [ ] Refactor large functions (e.g., compute_me_stats is 400+ lines)
- [ ] Extract constants to configuration module (magic numbers like shift windows, etc.)
- [ ] Improve error handling with custom exceptions
- [ ] Add input validation for all public functions
- [x] **Remove or properly handle all TODO comments in code** ✅ COMPLETED (2025-01-12)
  - [ ] TODO 3: Automatic min_shifts from autocorrelation (skipped for now - requires complex implementation)
  - [ ] TODO 4: Deprecate joint_distr branch (in progress - evaluating MultiTimeSeries replacement)
- [ ] Implement proper logging instead of print statements

### Medium Priority
- [ ] Create abstract base classes for extensibility
- [ ] Separate concerns (statistics, visualization, IO operations)
- [ ] Implement builder pattern for complex parameter configurations
- [ ] Add configuration validation schemas

## 3. Testing

### High Priority
- [ ] Increase test coverage to >90% (currently at 69% as of 2025-01-10)
  - [x] **PRIORITY 1: Add tests for disentanglement.py (currently 19% coverage)** ✅ COMPLETED (2025-01-10)
  - [ ] **PRIORITY 2: Fix failing existing tests**
    - [x] **Fix test_intense_analysis_compatibility in test_3d_spatial_manifold.py** ✅ COMPLETED (2025-01-14)
    - [x] **Fix test_compute_cell_cell_significance** (from test_intense_pipelines.py) ✅ COMPLETED (2025-01-10)
  - [x] **PRIORITY 3: Improve visual.py coverage (currently 47%)** ✅ COMPLETED (2025-01-11)
- [x] **Add unit tests for individual statistical functions** ✅ COMPLETED (2025-01-11)
- [ ] Test edge cases (empty data, single point, etc.)
- [ ] Add integration tests for full pipeline
- [ ] Test parallel vs sequential execution consistency
- [ ] Add performance benchmarks
- [ ] Test all supported metrics (MI, correlation, etc.)
- [ ] Test all distribution types for p-value calculation
- [ ] **NEW: Test disentanglement analysis functions**
  - [x] Unit tests for `disentangle_pair` with various MI/CMI combinations ✅ COMPLETED
  - [x] Test redundancy detection (negative interaction information) ✅ COMPLETED
  - [x] Test synergy detection (positive interaction information) ✅ COMPLETED
  - [x] Test edge cases (zero MI, equal contributions) ✅ COMPLETED
  - [x] Integration tests for `disentangle_all_selectivities` ✅ COMPLETED
  - [x] Test multifeature mapping and aggregation ✅ COMPLETED
  - [x] Test error handling for missing features ✅ COMPLETED
- [ ] **NEW: Disentanglement Integration Tests (FROM ARCHIVE)**
  - [ ] End-to-end workflow with synthetic experiments
  - [ ] Test with known selectivity patterns (pure place cells, mixed selectivity)
  - [ ] Verify disentanglement detects correct patterns
  - [ ] Test with various noise levels
  - [ ] Compatibility tests with existing INTENSE pipeline
  - [ ] Test with different data types (calcium vs spikes)
- [x] **NEW: Test information theory additions** ✅ COMPLETED (2025-01-11)
- [x] **NEW HIGH PRIORITY: Fix interaction information sign issues** ✅ COMPLETED (2025-01-11)
- [ ] **NEW: Implement Partial Information Decomposition (PID) module**
  - [ ] Create new module `driada.information.pid` for Partial Information Decomposition
  - [ ] Implement Williams & Beer (2010) PID framework
  - [ ] Decompose mutual information into:
    - [ ] Unique information from each variable
    - [ ] Redundant (shared) information
    - [ ] Synergistic information
  - [ ] Support for 2-variable decomposition (standard case)
  - [ ] Extension to 3+ variables if feasible
  - [ ] Handle both discrete and continuous variables
  - [ ] Add comprehensive tests with known PID examples
  - [ ] Create visualization functions for PID results
  - [ ] Document theoretical background and usage
  - [ ] Note: This is different from interaction information (II) which only gives net redundancy/synergy
- [x] **NEW: Test visualization functions** ✅ COMPLETED
  - [x] Test `plot_disentanglement_heatmap` with various inputs ✅ COMPLETED
  - [x] Test `plot_disentanglement_summary` with single/multiple experiments ✅ COMPLETED
  - [x] Test colormap generation and masking ✅ COMPLETED
  - [x] Test parameter validation and error handling ✅ COMPLETED
- [ ] **NEW: Test compute_feat_feat_significance function**
  - [ ] Test with default 'all' features mode
  - [ ] Test with specific feature subsets
  - [ ] Test MultiTimeSeries handling
  - [ ] Verify diagonal elements are zero
  - [ ] Test two-stage significance testing
  - [ ] Test with different metrics (MI, correlation)
  - [ ] Verify matrix symmetry
  - [ ] Test parallel vs sequential execution

- [ ] **NEW: Disentanglement Performance Tests (FROM ARCHIVE)**
  - [ ] Time complexity for different population sizes
  - [ ] Memory usage with large datasets (>1000 neurons)
  - [ ] Stress tests with many features (>20)
  - [ ] Very long time series (>100k frames)
  - [ ] Create benchmark suite for disentanglement analysis
- [ ] **NEW: Mathematical Property Validation Tests (FROM ARCHIVE)**
  - [ ] Verify CMI chain rule: I(X;Y,Z) = I(X;Y) + I(X;Z|Y)
  - [ ] Test CMI inequality: CMI ≤ MI
  - [ ] Verify H(X,Y) ≥ max(H(X), H(Y))
  - [ ] Test H(Z|X) ≤ H(Z)
  - [ ] Validate entropy relationships
  - [ ] Test symmetry of interaction information

### Medium Priority
- [ ] Add property-based testing for statistical functions
- [ ] Create test fixtures for common data patterns
- [ ] Add tests for visualization functions
- [ ] Test memory usage with large datasets
- [ ] Add tests for reproducibility with seeds
- [ ] **Fix test_download_extension** - List ordering assertion issue in test_download.py
  - [ ] Currently fails due to os.listdir() returning files in different order
  - [ ] Should use set comparison or sorted lists instead of direct list comparison
  - [ ] Low priority - not related to INTENSE functionality

## 4. Features & Functionality

### High Priority
- [ ] **Implement Numba Configuration System (FROM ARCHIVE)**
  - [ ] Create driada/config.py with environment variable support
  - [ ] Implement conditional_njit decorator that respects configuration
  - [ ] Refactor all @njit decorators to use conditional version
  - [ ] Add DRIADA_DISABLE_NUMBA environment variable support
  - [ ] Create fallback pure Python implementations for JIT functions
  - [ ] Make Numba an optional dependency in pyproject.toml
  - [ ] Add tests for both Numba-enabled and disabled modes
  - [ ] Update documentation with usage instructions
- [ ] Implement missing MultiTimeSeries support in joint_distr mode
- [ ] Add support for custom similarity metrics
- [ ] Implement auto-correlation based min_shift calculation
- [ ] Add progress reporting for long computations
- [ ] Implement checkpoint/resume for long computations
- [ ] Add batch processing capabilities
- [ ] **Fix synthetic data generation for continuous/MultiTimeSeries selectivity**
  - [ ] Investigate why generate_synthetic_exp produces no selectivity for continuous features
  - [ ] Ensure MultiTimeSeries features can show selectivity in synthetic data
  - [ ] Add parameters to control continuous feature selectivity strength
  - [ ] Verify compute_cell_feat_significance handles all feature types equally

### Medium Priority
- [ ] Add support for different shuffling strategies
- [ ] Implement additional statistical tests beyond gamma/lognorm
- [ ] Add cross-validation for significance testing
- [ ] Support for temporal windowing analysis
- [ ] Add support for multi-GPU processing

### Low Priority
- [ ] Add support for streaming/online computation
- [ ] Implement adaptive sampling for efficiency
- [ ] Add Bayesian approaches for significance testing

## 5. Performance Optimization

### High Priority
- [ ] Profile and optimize mutual information calculations
- [ ] Optimize memory usage for large datasets
- [ ] Implement caching for repeated calculations
- [ ] Optimize shuffling operations
- [ ] Vectorize statistical computations where possible

### Medium Priority
- [ ] Add GPU acceleration for suitable operations
- [ ] Implement approximate algorithms for large-scale analysis
- [ ] Optimize KDTree operations for high-dimensional data
- [ ] Add data structure optimization for sparse data

## 6. Visualization

### High Priority
- [ ] Complete unfinished visualization functions
- [ ] Add interactive visualizations
- [ ] Implement heatmaps for pairwise significance
- [ ] Add statistical summary plots
- [ ] Create publication-ready figure exports
- [ ] **Fix examples to properly use visualization functions**
  - [ ] Update full_pipeline.py to show MI values in heatmap
  - [ ] Update mixed_selectivity.py to use plot_disentanglement_heatmap
  - [ ] Ensure all examples leverage existing visual.py functions

### Medium Priority
- [ ] Add 3D visualization capabilities
- [ ] Implement animation for temporal analysis
- [ ] Add customizable color schemes
- [ ] Support for multiple plot backends

## 7. Integration & Compatibility

### High Priority
- [ ] Ensure compatibility with latest scipy/numpy versions
- [ ] Add data format converters for common neuroscience formats
- [ ] Create integration examples with popular neuroscience tools
- [ ] Ensure thread-safety for parallel operations

### Medium Priority
- [ ] Add support for distributed computing frameworks
- [ ] Create plugins for common analysis platforms
- [ ] Add data validation for external inputs

## 8. User Experience

### High Priority
- [ ] Create user-friendly error messages
- [ ] Add verbose mode with detailed progress information
- [ ] Implement dry-run mode for parameter validation
- [ ] Add result summary reports
- [ ] Create result export functions (CSV, HDF5, etc.)

### Medium Priority
- [ ] Add interactive parameter selection tools
- [ ] Create GUI for common workflows
- [ ] Add result visualization dashboard

## 9. Scientific Validation

### High Priority
- [ ] Validate statistical methods against published benchmarks
- [ ] Add citations for all implemented methods
- [ ] Create validation datasets with known ground truth
- [ ] Document assumptions and limitations

### Medium Priority
- [ ] Implement method comparison tools
- [ ] Add sensitivity analysis tools
- [ ] Create synthetic data generators for testing

## 10. Code Cleanup

### High Priority
- [ ] Remove commented-out code
- [ ] Standardize variable naming conventions
- [ ] Remove duplicate code patterns
- [ ] Fix all linting issues
- [ ] Update deprecated function calls

### Medium Priority
- [ ] Refactor nested functions for clarity
- [ ] Simplify complex conditional logic
- [ ] Extract magic numbers to named constants
- [ ] Improve code organization and module structure

## 11. Documentation Gaps (from README analysis)

### High Priority
- [x] **Update README_INTENSE.md interaction information documentation** ✅ COMPLETED (2025-01-11)

### High Priority
- [ ] Add implementation details for shuffle mask handling
- [ ] Document how joint distributions are handled
- [ ] Explain edge effects in circular shifts
- [ ] Create parameter selection guide (gamma vs other distributions)
- [ ] Add guidance on choosing appropriate delay windows
- [ ] Document downsampling effects and recommendations

### Medium Priority
- [ ] Add validation section with synthetic data examples
- [ ] Create comparison table (INTENSE vs correlation, GLMs, etc.)
- [ ] Add troubleshooting section for common issues
- [ ] Provide guidance on interpreting borderline results
- [ ] Document effect sizes vs significance
- [ ] Add cross-validation options documentation
- [ ] Document caching mechanisms

### Low Priority
- [ ] Create interactive Jupyter notebook examples
- [ ] Add performance benchmarks for different dataset sizes
- [ ] Create FAQ section for interpretation questions
- [ ] Add flowchart for decision making

## Implementation Priority Order

**URGENT: Pre-Demo Phase (IMMEDIATE - before library demonstration)**
- [x] **Critical new user onboarding materials** (Section 0) - **PHASE 1 COMPLETED** ✅

1. **Phase 1 - Foundation (Weeks 1-2)** ✅ COMPLETED

2. **Phase 2 - Testing & Quality (Weeks 3-4)** ✅ COMPLETED (2025-01-09)

3. **Phase 3 - Documentation (Weeks 5-6)**
   - Write user guide and tutorials
   - Create API reference
   - Add scientific background documentation
   - Create example notebooks

4. **Phase 4 - Optimization (Weeks 7-8)**
   - Profile and optimize bottlenecks
   - Implement caching
   - Add parallel processing improvements
   - Optimize memory usage
   
   **NEW: Test Suite & Performance Optimizations (2025-01-09)** ✅ COMPLETED
   
   - [ ] Further test suite optimizations
     - [ ] Implement test data caching between runs
     - [ ] Add pytest-xdist for parallel test execution
     - [ ] Create tiered test suites (smoke, fast, full)
     - [ ] Add performance benchmarks to track regressions
   - [ ] Fix remaining test failures
     - [ ] Fix correlation detection in `test_compute_cell_cell_significance`
     - [ ] Handle empty feature list in `test_compute_feat_feat_significance_empty_features`
   - [ ] Memory optimization
     - [ ] Profile memory usage in large-scale computations
     - [ ] Implement streaming computation for large datasets
     - [ ] Add memory-efficient data structures
   - [ ] Computational optimizations
     - [ ] Vectorize shuffling operations
     - [ ] Implement adaptive shuffling (stop on convergence)
     - [ ] Add GPU acceleration for MI calculations
     - [ ] Optimize copula transformations

5. **Phase 5 - Features (Weeks 9-10)**
   - Complete MultiTimeSeries support
   - Add new metrics and statistical tests
   - Enhance visualization capabilities
   - Add batch processing

6. **Phase 6 - Polish (Weeks 11-12)**
   - User experience improvements
   - Final documentation review
   - Performance benchmarks
   - Release preparation

## CLEANUP TASKS - Need Investigation

### Test Files from Previous Sessions
- [ ] **Review test report files** - Determine if needed for documentation
  - [ ] final_test_report.py - May contain test results summary
  - [ ] final_timing_report.py - May contain performance benchmarks
  - [ ] test_examples.py - May be part of example testing suite
  
### Performance Test Files  
- [ ] **Review JIT performance tests** - Determine if part of optimization suite
  - [ ] test_fixed_jit.py
  - [ ] test_intense_jit_performance.py  
  - [ ] test_jit_final.py
  - [ ] test_jit_performance_isolated.py
  - [ ] test_performance_comparison.py

### Documentation Files
- [ ] **Review TODO documentation** - May contain important notes
  - [ ] TODO_DISENTANGLEMENT_TESTS.md
  - [ ] TODO_NUMBA_CONFIG.md
  - [ ] NUMBA_0.60_RIDGE_ISSUE.md - Known issue documentation

## RECENT ACHIEVEMENTS (2025-07-17)

### Extract Task Variables Example Updates ✅ COMPLETED (2025-07-17)

### Performance Optimization ✅ COMPLETED (2025-07-17)

## POST-RELEASE PLANS

### Disentangled Dimensionality Reduction Module - PLANNED

**Objective**: Implement disentangled representation learning methods that can discover interpretable latent factors from neural population data, complementing DRIADA's existing single-cell (INTENSE) and population-level (DR) analyses.

**Motivation**:
- Current DR methods (PCA, UMAP, etc.) find low-dimensional structure but don't guarantee interpretability
- Disentangled representations can isolate independent factors of variation (e.g., position, speed, reward)
- Natural extension of DRIADA's mission to bridge single-neuron selectivity and population structure
- Enables discovery of latent variables that individual neurons may not cleanly encode

**Core Components**:

1. **β-VAE Implementation**
   - Variational autoencoder with adjustable β parameter for disentanglement pressure
   - Architecture suitable for neural time series data (1D convolutions or RNNs)
   - Loss function: reconstruction + β * KL divergence
   - Latent traversal visualization to verify disentanglement
   - Integration with MVData for seamless use with other DR methods

2. **Information Bottleneck Methods**
   - Implement Variational Information Bottleneck (VIB) for supervised disentanglement
   - Deep Information Bottleneck with deterministic or stochastic encoders
   - Multi-task learning to discover task-relevant disentangled features
   - Connection to INTENSE: use MI to validate discovered latents

3. **Disentanglement Metrics**
   - SAP (Separated Attribute Predictability) score
   - MIG (Mutual Information Gap)
   - DCI (Disentanglement, Completeness, Informativeness)
   - Factor-VAE metric
   - Integration with existing INTENSE framework for validation

4. **Neural-Specific Adaptations**
   - Handle trial structure and repeated conditions
   - Incorporate known behavioral variables as weak supervision
   - Support for both calcium imaging and spike train data
   - Temporal consistency constraints for smoother latents

**Integration with DRIADA**:

1. **API Design**
   ```python
   from driada.disentangled import BetaVAE, VIB, DisentanglementMetrics
   from driada.dim_reduction import MVData
   
   # Use with existing infrastructure
   mvdata = MVData(neural_data)
   
   # Unsupervised disentanglement
   beta_vae = BetaVAE(latent_dim=10, beta=4.0)
   disentangled_embedding = mvdata.get_embedding({
       'e_method_name': 'beta_vae',
       'e_method': beta_vae
   })
   
   # Supervised disentanglement with behavioral variables
   vib = VIB(latent_dim=8, beta=0.01)
   vib.fit(neural_data, behavioral_labels)
   latents = vib.encode(neural_data)
   
   # Evaluate disentanglement
   metrics = DisentanglementMetrics()
   sap_score = metrics.sap(latents, ground_truth_factors)
   mig_score = metrics.mig(latents, ground_truth_factors)
   ```

2. **Workflow Integration**
   - Pre-processing: Use existing signal processing and normalization
   - Post-processing: Feed disentangled latents to INTENSE for validation
   - Visualization: Extend existing plotting functions for latent traversals
   - Validation: Compare discovered factors with known behavioral variables

3. **Example Workflows**
   - `examples/discover_latent_factors.py`: Unsupervised discovery of neural population factors
   - `examples/validate_disentanglement.py`: Use INTENSE to verify latent interpretability
   - `examples/compare_dr_methods.py`: β-VAE vs PCA vs UMAP for interpretability
   - `examples/temporal_disentanglement.py`: Discover slowly-changing vs fast dynamics

**Technical Implementation Plan**:

1. **Phase 1: Core Infrastructure** (4 weeks)
   - Create `driada/disentangled/` module structure
   - Implement basic β-VAE with PyTorch backend
   - Add to MVData embedding pipeline
   - Basic visualization tools

2. **Phase 2: Advanced Methods** (4 weeks)
   - Implement VIB and Deep Information Bottleneck
   - Add disentanglement metrics suite
   - Support for supervised/semi-supervised learning
   - Temporal consistency regularization

3. **Phase 3: Neural-Specific Features** (3 weeks)
   - Trial-aware architectures
   - Integration with Experiment class
   - Specialized preprocessing for neural data
   - Validation against known selectivity

4. **Phase 4: Integration & Testing** (3 weeks)
   - Full integration with INTENSE pipeline
   - Comprehensive test suite
   - Performance optimization
   - Documentation and examples

**Research Applications**:
- Discover latent task variables not directly measured
- Separate sensory, motor, and cognitive components
- Identify slow dynamical modes vs fast fluctuations
- Compare biological and artificial neural network representations
- Enable more interpretable brain-computer interfaces

**Success Metrics**:
- Disentanglement scores > 0.8 on synthetic data with known factors
- Discovered latents correlate with INTENSE-identified selectivity
- Computational efficiency comparable to existing DR methods
- Clear improvement in interpretability over PCA/UMAP for mixed populations

This module would position DRIADA as a comprehensive framework for interpretable neural data analysis, from single neurons to disentangled population factors.

## 🐛 KNOWN BUGS & ISSUES

### INTENSE Distribution Choice Sensitivity (Discovered 2025-01-21)
- **Issue**: INTENSE p-value calculation is highly sensitive to the distribution choice for fitting shuffled MI values
- **Symptoms**: 
  - With some distributions, INTENSE finds false positives even with pure random noise
  - With gamma distribution, results are more conservative and appropriate
- **Root Cause**: Different distributions fit the shuffled MI values differently, leading to vastly different p-values
- **Workaround**: Use `metric_distr_type='gamma'` for MI-based testing
- **Status**: ⚠️ POSTPONED - Needs further investigation on optimal distribution selection strategy
- **TODO**: 
  - Investigate why different distributions give such different results
  - Consider adaptive distribution selection based on data characteristics
  - Add warnings or auto-selection of appropriate distribution

## 🔧 DRIADA COMPLETION TASKS (2025-01-17)

### Task 1: Remove Signals Folder and Redistribute Functionality
- [ ] **Analyze current signals module usage**
  - [ ] Identify which modules depend on signals functionality
  - [ ] Map each signals function to its logical home
  - [ ] Ensure no breaking changes to public API
- [ ] **Redistribute functionality**
  - [ ] Move neural_filtering.py functionality to experiment module
  - [ ] Move complexity.py to utils or appropriate module
  - [ ] Move sig_base.py TimeSeries handling to experiment
  - [ ] Move theory.py functions to appropriate modules
- [ ] **Update all imports**
  - [ ] Fix all internal imports
  - [ ] Update __init__.py files
  - [ ] Ensure backward compatibility where needed
- [ ] **Test migration**
  - [ ] Run all tests to ensure nothing breaks
  - [ ] Update test imports as needed

### Task 2: Process Documentation from Archive ✅ COMPLETED (2025-01-22)
- [x] **Review archive documentation** ✅ COMPLETED
  - [x] Process TODO_DISENTANGLEMENT_TESTS.md ✅ COMPLETED
  - [x] Process TODO_NUMBA_CONFIG.md ✅ COMPLETED
  - [x] Process NUMBA_0.60_RIDGE_ISSUE.md ✅ COMPLETED
  - [x] Extract actionable items and implementation plans ✅ COMPLETED
- [x] **Integrate findings** ✅ COMPLETED
  - [x] Update relevant modules based on archive insights ✅ COMPLETED
  - [x] Document any unresolved issues in appropriate places ✅ COMPLETED
  - [x] Move completed items to main documentation ✅ COMPLETED

**Implementation Details:**
- Created ARCHIVE_PROCESSING_SUMMARY.md documenting all findings
- Identified that many disentanglement tests are already implemented
- Numba configuration system remains unimplemented (added to Features)
- NUMBA_0.60_RIDGE_ISSUE is resolved (no action needed)
- Added missing test items to Testing section
- Added Numba configuration to Features & Functionality section

### Task 3: Verify Global Code Coverage and Create Missing Tests
- [ ] **Run comprehensive coverage analysis**
  - [ ] Generate full coverage report with pytest-cov
  - [ ] Identify modules with <90% coverage
  - [ ] Prioritize critical modules for testing
- [ ] **Create missing tests**
  - [ ] Test dimensionality estimation module (currently incomplete)
  - [ ] Test integration between modules
  - [ ] Add edge case tests for low-coverage modules
  - [ ] Ensure all public API functions have tests
- [ ] **Achieve coverage targets**
  - [ ] Target: >90% overall coverage
  - [ ] 100% coverage for public API
  - [ ] Document any intentionally untested code

### Task 4: Create Network Module Example
- [ ] **Design example showcasing Network module**
  - [ ] Create example using graph analysis on neural connectivity
  - [ ] Show spectral analysis capabilities
  - [ ] Demonstrate community detection
  - [ ] Include visualization with drawing module
- [ ] **Implement example**
  - [ ] Create examples/network_analysis.py
  - [ ] Use synthetic or real neural connectivity data
  - [ ] Show practical neuroscience applications
  - [ ] Add comprehensive documentation
- [ ] **Test and validate**
  - [ ] Ensure example runs without errors
  - [ ] Generate meaningful visualizations
  - [ ] Add to examples README

### Task 4b: Remove e_method Redundancy (PRE-RELEASE PRIORITY) ✅ COMPLETED (2025-01-21)
**Note**: Discovered during analysis that this was already completed in commit "refactor: update all DR code to use new simplified API"
- [x] **Simplify DR method specification** ✅ COMPLETED
  - [x] e_method can be reconstructed from e_method_name using METHODS_DICT
  - [x] Update MVData.get_embedding() to auto-construct method objects
  - [x] Deprecate e_method parameter in favor of e_method_name only
  - [x] Maintain backward compatibility with deprecation warning
- [x] **Update all examples and tests** ✅ COMPLETED
  - [x] Remove e_method parameter from all calls
  - [x] Simplify DR method usage throughout codebase
  
**Implementation Details:**
- Simplified API allows `mvdata.get_embedding(method='pca')` instead of passing both e_method and e_method_name
- merge_params_with_defaults() automatically sets both parameters
- All examples updated to use new simplified format
- Backward compatibility maintained - old code still works

### Task 5: Enhance TimeSeries and MultiTimeSeries Architecture (PRE-RELEASE PRIORITY) ✅ COMPLETED (2025-01-21)
**Note**: Critical for improving user experience and enabling direct DR on neural data
- [x] **Refactor MultiTimeSeries class hierarchy** ✅ COMPLETED
  - [x] Make MultiTimeSeries inherit from MVData
    - [x] MultiTimeSeries IS-A multi-dimensional dataset
    - [x] Enables direct DR without conversion
    - [x] Maintains time series specific functionality
  - [x] Add support for discrete MultiTimeSeries
    - [x] Update _check_input to allow discrete=True
    - [x] Handle mixed discrete/continuous components (prohibited for clarity)
    - [x] Update entropy calculations for discrete case
- [x] **Create filtered TimeSeries interface** ✅ COMPLETED
  - [x] Add .filter() method to TimeSeries
    - [x] Returns new filtered TimeSeries object
    - [x] Support multiple filter types (gaussian, savgol, wavelet)
    - [x] Preserve metadata and shuffle_mask
  - [x] Add .filter() to MultiTimeSeries
    - [x] Apply same filter to all components
    - [x] Maintain temporal alignment
  - [ ] Integrate wavelet filtering as primary method
    - [ ] Use existing wavelet_event_detection module
    - [ ] Add to neural_filtering options as main filtering approach
    - [ ] Support multiple wavelet families
    - [ ] Auto-select wavelet based on signal characteristics
- [x] **Convert exp.calcium/spikes to MultiTimeSeries** ✅ COMPLETED
  - [x] Created MultiTimeSeries directly from neurons
    - [x] Preserves individual neuron shuffle masks
    - [ ] Each neuron is a TimeSeries component
    - [ ] Support both continuous (calcium) and discrete (spikes)
  - [ ] Update Experiment class
    - [ ] exp.calcium returns CellularMultiTimeSeries
    - [ ] exp.spikes returns discrete CellularMultiTimeSeries
    - [ ] Maintain backward compatibility with array access
  - [ ] Enable direct DR on neural data
    - [ ] exp.calcium.get_embedding() should work
    - [ ] Automatic MVData conversion through inheritance

### Task 6: Create Spatial Analysis Utilities (PRE-RELEASE PRIORITY) ✅ COMPLETED (2025-01-21)
**Note**: Many spatial metrics already exist in manifold_metrics.py - focus on integration
- [x] **Review existing spatial metrics** ✅ COMPLETED
  - [x] manifold_metrics.py already has many metrics
  - [x] Identify gaps from intense_dr_pipeline.py
  - [x] Plan integration strategy
- [x] **Create utils.spatial module** ✅ COMPLETED
  - [x] Spatial-specific metrics not in manifold_metrics
    - [x] Place field analysis functions
    - [x] Grid score computation
    - [x] Spatial information rate
    - [x] Speed/direction filtering
  - [x] High-level spatial analysis functions
    - [x] analyze_spatial_coding()
    - [x] extract_place_fields()
    - [x] compute_spatial_metrics()
- [x] **Update examples to use library functions** ✅ COMPLETED
  - [x] Replace inline implementations
  - [x] Ensure consistent metric usage
  - [x] Add comparison plots

**Implementation Details:**
- Created comprehensive spatial.py module with 759 lines
- Supports continuous calcium signals as primary input
- Implements occupancy maps, rate maps, place field detection
- Added spatial information rate (Skaggs et al. 1993)
- Grid score computation with rotational analysis
- Position decoding using Random Forest
- Speed and direction filtering utilities
- Full integration with TimeSeries/MultiTimeSeries
- 100% test coverage with 34 tests
- Updated intense_dr_pipeline.py to use library functions
- Added visualization example for spatial maps

### Task 7: Refactor Signal Module (PRE-RELEASE PRIORITY) ✅ COMPLETED (2025-01-21)
**Note**: Signal class is barely used but refactoring improves code organization
- [x] **Analyze Signal module usage** ✅ COMPLETED
  - [x] Signal class barely used (only in ts_wavelet_denoise)
  - [x] neural_filtering.py actively used
  - [x] brownian() and ApEn() unused
- [x] **Implement refactoring plan** ✅ COMPLETED
  - [x] Move brownian() to utils/signals.py
  - [x] Move ApEn() to utils/signals.py (renamed to approximate_entropy)
  - [x] Move all filtering functions to utils/signals.py
  - [x] Delete entire signals module (Signal class was unused)
  - [x] Add approximate_entropy method to TimeSeries class
- [x] **Update module structure** ✅ COMPLETED
  - [x] Create utils/signals.py with all functionality
  - [x] Fix all imports throughout codebase
  - [x] Update documentation
  - [x] Ensure backward compatibility with deprecation warning

**Implementation Details:**
- Consolidated all signal processing in utils/signals.py
- Renamed ApEn to approximate_entropy for clarity
- Removed manifold_preprocessing wrapper (redundant)
- Added comprehensive tests for all functionality
- Created backward compatibility module with deprecation warning
- Simplified API by removing unnecessary abstractions

## 🚀 POST-RELEASE ENHANCEMENTS

### Enhancement 1: Advanced Disentangled Representations
- [ ] Implement β-VAE and variants
- [ ] Add disentanglement metrics
- [ ] Create examples for neural data
- [ ] Integration with INTENSE pipeline

### Enhancement 2: Factor Analysis Module
- [ ] Classical Factor Analysis
- [ ] Probabilistic Factor Analysis
- [ ] Sparse variants
- [ ] Integration with DR framework

### Enhancement 3: State Space Models
- [ ] Gaussian Process Factor Analysis (GPFA)
- [ ] Linear Dynamical Systems
- [ ] Trial-structured analysis
- [ ] Smooth trajectory extraction

### Enhancement 4: Additional Dimensionality Reduction Methods (POST-RELEASE)
- [ ] **Add random projections and PPCA**
  - [ ] Implement Random Projection methods
    - [ ] Gaussian random projections
    - [ ] Sparse random projections
    - [ ] Johnson-Lindenstrauss guarantee validation
  - [ ] Implement Probabilistic PCA (PPCA)
    - [ ] Full PPCA with EM algorithm
    - [ ] Missing data handling
    - [ ] Automatic dimensionality selection
  - [ ] Integration with existing DR framework
  - [ ] Add comprehensive tests and examples

### 2025-01-20: Visual Utilities Module and Examples Organization ✅ COMPLETED

