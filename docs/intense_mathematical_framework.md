# INTENSE: Information-Theoretic Evaluation of Neuronal Selectivity

INTENSE is a computational framework for analyzing neuronal selectivity to behavioral and environmental variables using information theory. It quantifies the relationship between neural signals (calcium imaging, spike trains) and external variables through mutual information analysis with rigorous statistical testing.

## Overview

INTENSE addresses a fundamental question in neuroscience: which neurons encode specific behavioral or environmental variables? By using mutual information (MI) as the core metric, INTENSE can detect both linear and nonlinear relationships between neural activity and external variables.

The key advantage of MI over traditional correlation measures is its ability to capture any statistical dependency:
- Linear relationships (captured by Pearson correlation)
- Monotonic relationships (captured by Spearman correlation)  
- Complex nonlinear dependencies (only captured by MI)

## Key features

- **Mutual Information Analysis**: Quantifies statistical dependencies between neural signals and behavioral variables
- **Two-Stage Statistical Testing**: Efficient hypothesis testing with multiple comparison correction
- **Optimal Delay Detection**: Accounts for temporal delays between neural activity and behavior
- **Mixed Selectivity Analysis**: Disentangles relationships when neurons respond to multiple correlated variables
- **Multiple Metrics Support**: MI, Spearman correlation, and other similarity metrics
- **Parallel Processing**: Efficient computation for large-scale neural datasets

## Quick start

Analyze neuronal selectivity with synthetic data:

```python
import driada

# Generate synthetic experiment with place cells and behavioral features
exp = driada.generate_synthetic_exp(
    n_dfeats=2,      # discrete features (e.g., trial type)
    n_cfeats=2,      # continuous features (e.g., x, y position)  
    nneurons=20,     # number of neurons
    duration=300,    # 5 minutes recording
    seed=42
)

# Discover significant neuron-feature relationships
stats, significance, info, results = driada.compute_cell_feat_significance(
    exp,
    mode='two_stage',
    n_shuffles_stage1=50,   # preliminary screening
    n_shuffles_stage2=1000, # validation (use 10000+ for publication)
    verbose=True
)

# Use Experiment method to get significant neurons
significant_neurons = exp.get_significant_neurons()
print(f"Found {len(significant_neurons)} significant neuron-feature pairs")

# Examine results using Experiment methods
for cell_id in list(significant_neurons.keys())[:3]:  # Show first 3 neurons
    for feat_name in significant_neurons[cell_id]:
        pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_name)
        
        print(f"Neuron {cell_id} - Feature '{feat_name}':")
        print(f"  Mutual Information: {pair_stats['pre_rval']:.4f}")
        if 'pval' in pair_stats:
            print(f"  P-value: {pair_stats['pval']:.2e}")
        print(f"  Optimal delay: {pair_stats.get('opt_delay', 0):.2f}s")
```

**Expected output:**
```
Found 8 significant neuron-feature pairs
Neuron 3 - Feature 'c_feat_0':
  Mutual Information: 0.2847
  P-value: 2.45e-05
  Optimal delay: 0.10s
```

**Visualization:**
```python
import matplotlib.pyplot as plt

# Plot neuron-feature relationship for first significant pair
if significant_neurons:
    cell_id = list(significant_neurons.keys())[0]
    feat_name = significant_neurons[cell_id][0]
    fig, ax = plt.subplots(figsize=(8, 5))
    driada.intense.plot_neuron_feature_pair(exp, cell_id, feat_name, ax=ax)
    plt.title(f"Neuron {cell_id} selectivity to {feat_name}")
    plt.show()
```

## Using your own data

The `Experiment` class is the main data container throughout DRIADA. It holds neural recordings, behavioral variables, and analysis results in a unified structure.

### Creating an Experiment from your data

```python
import numpy as np
import driada

# Your neural data
calcium_traces = np.random.randn(100, 12000)  # 100 neurons, 12000 timepoints
spike_times = None  # Optional: spike trains (same shape as calcium)

# Your behavioral variables (dynamic features)
x_position = np.random.randn(12000)           # x coordinate over time
y_position = np.random.randn(12000)           # y coordinate over time  
trial_type = np.random.randint(0, 2, 12000)   # discrete: 0=left, 1=right trials
speed = np.abs(np.random.randn(12000))        # continuous: movement speed

dynamic_features = {
    'x': x_position,
    'y': y_position, 
    'trial_type': trial_type,
    'speed': speed
}

# Experimental metadata (static features)
static_features = {
    'fps': 20.0,        # sampling rate (frames per second)
    't_rise_sec': 0.5,  # calcium rise time
    't_off_sec': 2.0    # calcium decay time
}

# Experiment identifiers
exp_identificators = {
    'session': 'session_001',
    'animal': 'mouse_123'
}

# Create Experiment object
exp = driada.Experiment(
    signature='MyExperiment',
    calcium=calcium_traces,
    spikes=spike_times,  # Can be None
    exp_identificators=exp_identificators,
    static_features=static_features,
    dynamic_features=dynamic_features
)

print(f"Created experiment with {exp.n_cells} neurons and {exp.n_frames} timepoints")

# Now analyze with INTENSE
stats, significance, info, results = driada.compute_cell_feat_significance(exp)
```

**Key points:**
- **calcium**: Neural activity traces (numpy array, shape: n_neurons × n_timepoints)
- **dynamic_features**: Behavioral variables that change over time (dict of numpy arrays)
- **static_features**: Experimental parameters (dict with 'fps', 't_rise_sec', 't_off_sec')
- **spikes**: Optional spike trains (can be None to auto-detect from calcium using wavelets)
- The Experiment automatically converts numpy arrays to TimeSeries objects internally
- All subsequent DRIADA analysis uses this unified Experiment container

## Mathematical framework

### Mutual information computation

INTENSE uses Gaussian Copula Mutual Information (GCMI) for robust MI estimation. The mutual information between two random variables X and Y is defined as:

```
I(X;Y) = ∫∫ p(x,y) log[p(x,y)/(p(x)p(y))] dx dy
```

Equivalently, using entropy notation:
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

where H(X) = -∫ p(x) log p(x) dx is the differential entropy.

#### Gaussian copula transformation

The GCMI method leverages the copula representation of joint distributions. The key insight is that MI is invariant under monotonic transformations, allowing us to transform to a convenient representation:

1. **Empirical CDF transformation**: For each variable, compute the empirical cumulative distribution function:
   ```
   F_X(x) = (rank(x) + 0.5) / N
   ```

2. **Gaussian transformation**: Apply the inverse normal CDF:
   ```
   X̃ = Φ^(-1)(F_X(X))
   Ỹ = Φ^(-1)(F_Y(Y))
   ```

3. **MI estimation**: For the Gaussian copula, MI has a simple form:
   ```
   I_GCMI(X;Y) = -0.5 log|R|
   ```
   where R is the correlation matrix of (X̃, Ỹ).

#### Handling different variable types

**Continuous-Continuous Case**: Direct application of GCMI formula above.

**Discrete-Continuous Case**: For discrete X with K states and continuous Y:
```
I(X;Y) = Σ_{k=1}^K p(X=k) · H(Y) - Σ_{k=1}^K p(X=k) · H(Y|X=k)
```
Each conditional entropy H(Y|X=k) is computed using GCMI on the subset where X=k.

### Statistical significance testing

Significance is assessed by comparing observed MI against a null distribution generated from time-shuffled signals:

```
MI_shuffle^(i) = I(X(t); Y(t + τᵢ))
```

where τᵢ represents random circular shifts preserving the temporal structure of individual signals while destroying their correlation.

#### Parametric null distribution modeling

The null distribution of shuffled MI values is modeled using a parametric distribution. The gamma distribution is the default choice due to:
- Non-negativity of MI values
- Right-skewed nature of MI distributions
- Theoretical support from information theory

The gamma distribution parameters are estimated via maximum likelihood:
```
MI_shuffle ~ Γ(α, β)
p(MI|α,β) = (β^α/Γ(α)) · MI^(α-1) · exp(-β·MI)
```

The p-value is computed as:
```
p-value = P(MI_shuffle ≥ MI_observed) = 1 - F_Γ(MI_observed; α̂, β̂)
```

where F_Γ is the cumulative distribution function of the fitted gamma distribution.

#### Noise addition for numerical stability

A small amount of noise (default: 10^-3) is added to MI values to:
- Prevent numerical issues with identical values
- Improve parametric distribution fitting
- Avoid degeneracies in rank-based statistics

#### Rank values (r-vals) and non-parametric testing

INTENSE employs a dual approach combining parametric and non-parametric methods:

**Rank Values (r-vals)**: For each neuron-feature pair, INTENSE computes the rank of the observed MI value within the null distribution:
```
r-val = rank(MI_observed) / (n_shuffles + 1)
```

The r-val represents the proportion of shuffled MI values that are **lower** than the observed MI. This is equivalent to computing an empirical p-value:
```
empirical_p_value = 1 - r-val
```

**Advantages of r-vals**:
- Distribution-free: No assumptions about the shape of the MI distribution
- Robust: Unaffected by outliers or distribution fitting failures
- Conservative: The (n_shuffles + 1) denominator prevents p-values of exactly 0

**Dual Criterion for Significance**: INTENSE requires both criteria to be met:
1. **Non-parametric criterion**: r-val > (1 - k/(n_shuffles + 1)), typically k=5
2. **Parametric criterion**: p-value < threshold (after multiple comparison correction)

This dual approach provides robustness:
- The r-val criterion ensures the observed MI is genuinely in the extreme tail
- The parametric p-value provides a continuous measure of significance
- Both must pass, preventing false positives from poor distribution fits

### Two-stage testing procedure

The two-stage approach balances computational efficiency with statistical rigor:

#### Stage 1: Screening (100 shuffles)
- **Criterion**: MI_observed > max{MI_shuffle^(i)}_{i=1}^{100}
- **Purpose**: Early rejection of non-significant pairs
- **False negative rate**: < 1% for truly significant pairs
- **Computational savings**: ~100x for rejected pairs

#### Stage 2: Validation (10,000 shuffles)
- **Applied to**: Pairs passing Stage 1
- **Criteria**: 
  1. Rank criterion: MI_observed > 99.95th percentile of shuffles
  2. Parametric test: p-value < threshold (after multiple comparison correction)

#### Multiple comparison correction

The Holm-Bonferroni method is used to control family-wise error rate (FWER):

1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. For the k-th hypothesis, use threshold:
   ```
   α_k = α / (m - k + 1)
   ```
   where α = 0.01 (FWER) and m = number of hypotheses from Stage 1

3. Reject hypotheses while p_k < α_k; stop at first failure

This provides strong FWER control while being more powerful than standard Bonferroni correction.

### Optimal delay detection

Neural activity often shows temporal delays relative to behavior due to:
- Calcium indicator dynamics (~1-2s delay for GCaMP)
- Predictive coding (activity preceding behavior)
- Sensory processing delays

INTENSE finds the optimal delay by maximizing MI:

```
δ_optimal = argmax_{δ} I(X(t); Y(t + δ))
MI_optimal = I(X(t); Y(t + δ_optimal))
```

where:
- δ ∈ [-W, +W], typically W = 2 seconds
- Resolution: Δδ = 0.05 seconds (or 1/fps)
- Search space: 80 delay values for 2s window

The significance testing uses the optimally delayed signals, accounting for the multiple comparison issue in delay selection.

### FFT-accelerated permutation testing

The two-stage procedure requires evaluating MI at thousands of circular shifts per neuron-behavior pair. A naive implementation recomputes MI independently for each shift, yielding a per-pair cost of $O(n_{\text{sh}} \cdot n)$ where $n$ is the number of time frames and $n_{\text{sh}}$ the number of shuffles. For a typical experiment with $N$ neurons, $M$ behavioral features, $D$ delay steps, and $n_{\text{sh}} = 10{,}000$ shuffles, this becomes prohibitive.

INTENSE exploits the fact that circular time shifts correspond to circular cross-correlations, which can be computed for *all* $n$ possible shifts simultaneously via the convolution theorem:

$$C_{XY}[s] = \sum_{i=0}^{n-1} X[i] \, Y[(i+s) \bmod n] = \mathcal{F}^{-1}\!\bigl[\mathcal{F}[X] \cdot \mathcal{F}[Y]^{*}\bigr][s]$$

where $\mathcal{F}$ denotes the discrete Fourier transform and $^{*}$ complex conjugation. This reduces the cost from $O(n_{\text{sh}} \cdot n)$ to $O(n \log n)$ for computing MI at all shifts, after which any specific shift is an $O(1)$ array lookup. The particular form of MI at each shift depends on the variable types:

**Continuous--continuous case.** For copula-normalized Gaussian variables, MI reduces to $I(X; Y_s) = -\tfrac{1}{2}\log_2(1 - r(s)^2)$ where $r(s)$ is the Pearson correlation at shift $s$. The correlation for all shifts is obtained as $r(s) = C_{XY}[s] / [(n-1)\,\sigma_X \sigma_Y]$, requiring a single pair of real-valued FFTs.

**Continuous--discrete case.** For a continuous variable $Z$ and a discrete variable $X$ with $K$ classes, class-conditional sums at every shift are circular correlations with indicator functions:

$$\sum_i Z[i] \cdot \mathbb{1}[X_{(i+s)} = k] = \mathcal{F}^{-1}[\mathcal{F}[Z] \cdot \mathcal{F}[\mathbb{1}_{X=k}]^{*}][s]$$

From these sums and the analogous sums of $Z^2$, class-conditional variances and Gaussian entropies are computed for all shifts simultaneously, yielding MI via $I = H(Z) - \sum_k p(k)\,H(Z | X\!=\!k)$. This requires $2K+2$ FFTs (for $Z$, $Z^2$, and $K$ indicator functions), independent of the number of shuffles.

**Discrete--discrete case.** Contingency tables for all shifts are obtained via indicator cross-correlations: $n_{ij}(s) = \mathcal{F}^{-1}[\mathcal{F}[\mathbb{1}_{X=i}] \cdot \mathcal{F}[\mathbb{1}_{Y=j}]^{*}][s]$, requiring $K_X \cdot K_Y$ FFT pairs.

**Multivariate case.** For $d$-dimensional behavioral features (e.g., 2D spatial position), the joint covariance matrix separates into shift-invariant within-variable blocks and shift-dependent cross-covariance blocks computed via $d$ FFTs. MI is then evaluated through closed-form vectorized determinant formulas ($d \leq 3$).

#### FFT cache strategy

INTENSE precomputes MI for all $n$ shifts once per neuron-feature pair and stores the result in an FFT cache. This cache is reused across delay optimization, Stage 1 screening, and Stage 2 validation without recomputation. The total cost is $O(N \cdot M \cdot n \log n)$ for the cache build plus $O(N \cdot M \cdot n_{\text{sh}})$ for shift lookups, compared to $O(N \cdot M \cdot n_{\text{sh}} \cdot n)$ without acceleration.

### Mixed selectivity analysis

Many neurons respond to multiple, often correlated behavioral variables. INTENSE disentangles these relationships using information-theoretic decomposition.

#### Conditional mutual information

For neural activity A and behavioral variables X, Y:

```
I(A;X|Y) = H(A|Y) - H(A|X,Y) = I(A;X,Y) - I(A;Y)
```

This quantifies information about A in X that is not present in Y.

#### Interaction information

The three-way interaction (synergy/redundancy) is measured using the Williams & Beer (2010) convention:

```
II(A;X;Y) = I(A;X|Y) - I(A;X) = I(A;Y|X) - I(A;Y)
```

- II < 0: Redundancy (X and Y provide overlapping information about A)
- II > 0: Synergy (X and Y together provide more information than separately)

Note: Interaction information (II) provides the net redundancy or synergy but does not decompose mutual information into unique, redundant, and synergistic components. A future Partial Information Decomposition (PID) module will provide this full decomposition.

#### Variable type handling

1. **Both continuous**: Use multivariate GCMI with Cholesky decomposition
2. **X continuous, Y discrete**: Compute I(A;X|Y=y) for each y, then average
3. **X discrete, Y continuous**: Use the identity I(A;X|Y) = I(A;X) - [I(A;Y) - I(A;Y|X)]
4. **Both discrete**: Standard discrete MI formulas with empirical probabilities

#### Interpretation framework

For negative II (common in neural data), we identify the "weakest" link:
- If |I(A;X)| < |II|: X is redundant given Y
- If |I(A;Y)| < |II|: Y is redundant given X
- Otherwise: Both variables contribute independently

## Usage example

```python
from driada.intense.pipelines import compute_cell_feat_significance

# Analyze neuronal selectivity in an experiment
stats, significance, info, results = compute_cell_feat_significance(
    exp,                    # Experiment object
    cell_bunch=None,        # Analyze all neurons
    feat_bunch=['speed', 'head_direction', 'x', 'y'],  # Behavioral variables
    metric='mi',            # Use mutual information
    mode='two_stage',       # Two-stage testing
    n_shuffles_stage1=100,  # Stage 1 shuffles
    n_shuffles_stage2=10000,# Stage 2 shuffles
    pval_thr=0.01,         # Significance threshold
    find_optimal_delays=True,
    shift_window=2          # ±2 second delay window
)
```

## Output

INTENSE provides:
- **Statistical measures**: MI values, p-values, optimal delays
- **Significance results**: Which neuron-behavior pairs are significant
- **Diagnostic information**: Shuffle distributions, test statistics

## Applications

- Place cell identification in navigation experiments
- Feature selectivity in sensory neurons
- Task-relevant neural encoding in cognitive experiments
- Population-level analysis of neural representations
- Validation of computational models against neural data

## Limitations and considerations

### Statistical power
- Stage 1 screening may miss weak but real effects (1% false negative rate)
- Multiple comparison correction reduces power for large neuron populations
- Minimum recommended recording length: 1000 timepoints per behavioral state

### Computational considerations
- Memory usage scales as O(N × M × S) for N neurons, M behaviors, S shuffles
- Optimal delay search adds 40-80x computational overhead
- Parallel processing recommended for >100 neurons

### Methodological assumptions
- Stationarity: Assumes stable neuron-behavior relationships
- Temporal resolution: Limited by sampling rate (calcium imaging: ~10-30 Hz)
- Shuffle validity: Assumes temporal shifts destroy all dependencies

## References

Key publications underlying INTENSE methodology:

1. **Gaussian Copula MI**: Ince, R. A., Giordano, B. L., Kayser, C., Rousselet, G. A., Gross, J., & Schyns, P. G. (2017). A statistical framework for neuroimaging data analysis based on mutual information estimated via a gaussian copula. Human brain mapping, 38(3), 1541-1573.

2. **Information Theory in Neuroscience**: Timme, N., Alford, W., Flecker, B., & Beggs, J. M. (2014). Synergy, redundancy, and multivariate information measures: an experimentalist's perspective. Journal of computational neuroscience, 36(2), 119-140.

3. **Interaction Information**: Ghassami, A., Kiyavash, N., Huang, B., & Zhang, K. (2017). Multi-domain causal structure learning in linear systems. Advances in neural information processing systems, 30.

4. **MI Distribution Theory**: Hutter, M. (2001). Distribution of mutual information. Advances in neural information processing systems, 14.

## Performance considerations

- Parallel processing supported for large datasets
- Downsampling available to reduce computational load
- Caching of precomputed statistics for iterative analysis
- Memory-efficient implementations for long recordings