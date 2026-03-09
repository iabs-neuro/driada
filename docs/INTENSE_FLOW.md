# INTENSE Method: How Selectivities Are Determined and Disentangled

## 1. Overview

INTENSE (Information-Theoretic Evaluation of Neuronal Selectivity) determines which neurons in a population encode specific behavioral or environmental variables. Given simultaneously recorded neural activity and behavioral measurements, it answers two questions: (1) which neuron-feature pairs carry statistically significant mutual information, and (2) when a neuron appears selective to multiple correlated features, which selectivities are genuine versus artifacts of behavioral correlation.

The pipeline combines shuffle-based significance testing with information-theoretic disentanglement, producing a cleaned map of neuron-to-feature relationships suitable for downstream population analysis.

## 2. Input Data

INTENSE operates on an Experiment object containing two types of time-aligned data:

**Neural signals** are stored as a MultiTimeSeries with shape (n_neurons, n_frames). Typically these are calcium imaging traces (deconvolved or raw fluorescence), though spike trains are also supported. Each neuron's trace is accessed as a univariate TimeSeries.

**Behavioral features** are stored as a dictionary of TimeSeries, each with n_frames timepoints. Features fall into several types that affect how mutual information is computed:

- **Discrete** features (e.g., trial type, locomotion state) have integer-valued data accessed via `.int_data`. MI is computed using class-conditional entropy decomposition.
- **Continuous** features (e.g., speed) have real-valued data. Before MI computation, values undergo copula normalization: rank-transform to uniform marginals, then inverse-normal CDF to Gaussian marginals. The transformed data is stored in `.copula_normal_data`.
- **Circular** features (e.g., head direction) are automatically replaced by their 2D (cos, sin) representation, creating a two-dimensional MultiTimeSeries. This avoids the discontinuity at 0/2pi that would corrupt MI estimation. The substitution is handled by `substitute_circular_with_2d` at the start of the pipeline.
- **Multivariate** features (e.g., 2D position from x,y coordinates) are represented as MultiTimeSeries with shape (d, n_frames) where d is typically 2 or 3. Individual components like x and y are aggregated into a single "place" feature via a multifeature map.
- **Spatial discrete** features (e.g., corners, walls, center zones) are binary indicators marking when the animal occupies a specific region.

## 3. Two-Stage Significance Testing

The central operation of INTENSE is determining whether the mutual information between a neuron's activity and a behavioral feature is statistically significant. This is done by comparing the observed MI against a null distribution generated from temporally shuffled data. A two-stage procedure balances computational cost against statistical rigor.

### Data Flow

```
Experiment
    |
    v
[Circular 2d substitution] -- headdirection -> headdirection_2d (cos, sin)
    |
    v
[FFT cache build] -- precompute MI for all circular shifts per pair
    |
    v
[Optimal delay finding] -- scan shifts in [-W, +W], pick max MI per pair
    |
    v
[Stage 1: Screening] -- 100 shuffles, rank-based filter
    |
    v
[Stage 2: Validation] -- 10,000 shuffles, only for Stage 1 survivors
    |                      p-value via gamma-ZI fit + multiple comparison correction
    v
[Anti-selectivity filter] -- remove pairs with signal ratio <= 1
    |
    v
[Feature-feature significance] -- which behavioral features are correlated?
    |
    v
[Disentanglement] -- for neurons with 2+ features: pre-filter, CMI, post-filter
    |
    v
stats, significance, disentanglement results
```

### FFT Cache

Before any scanning begins, INTENSE builds an FFT cache that precomputes MI between each neuron-feature pair for all possible circular shifts simultaneously. This exploits the fact that circular time shifts correspond to circular cross-correlations, which can be computed for all n shifts in O(n log n) via the convolution theorem. The cache is built once and reused across delay optimization, Stage 1, and Stage 2, eliminating redundant computation.

The cache supports different pair types (continuous-continuous, continuous-discrete, discrete-discrete, multivariate, and mixed) with specialized FFT routines for each. Cache building can be parallelized across CPU cores.

### Stage 1: Screening

**Purpose:** Quickly reject neuron-feature pairs that have no chance of being significant, reducing the number of expensive Stage 2 tests.

**Procedure:** For each neuron-feature pair, INTENSE computes MI at the optimal delay, then computes MI at 100 random circular shifts (the number is configurable via `n_shuffles_stage1`). Each shuffle circularly shifts the feature time series by a random offset, preserving the temporal autocorrelation structure of both signals while destroying their cross-correlation.

**Criterion:** The observed MI must be the top-ranked value among all 101 values (observed + 100 shuffles). Specifically, the rank value `pre_rval = (count of shuffles below observed MI + 1) / (n_shuffles + 1)` must exceed `1 - topk/(n_shuffles + 1)`. With default `topk=1` and 100 shuffles, this means the observed MI must be strictly higher than all 100 shuffled values.

**Outcome:** Pairs failing Stage 1 are excluded from Stage 2. No p-values are computed in Stage 1 because the screening criterion uses only ranks, which is computationally cheaper than distribution fitting.

### Stage 2: Validation

**Purpose:** Provide rigorous statistical assessment of pairs that survived Stage 1.

**Procedure:** Each surviving pair undergoes 10,000 shuffles (configurable via `n_shuffles_stage2`). The shuffled MI values form a null distribution.

**P-value computation:** The null distribution is modeled with a zero-inflated gamma (gamma_zi) distribution, which is the default and recommended choice. MI null distributions naturally contain a point mass at zero (when shuffled signals happen to be independent), and the gamma_zi model handles this explicitly:

- The zero-inflation parameter pi is estimated as the fraction of shuffled MI values near zero.
- A gamma distribution is fit (via maximum likelihood) to the non-zero shuffled values.
- The p-value is computed as `(1 - pi) * Gamma.sf(MI_observed)`.

Alternative distribution models (standard gamma, lognormal, normal) are supported but not recommended.

**Dual criterion:** A pair is declared significant only if both conditions hold:

1. **Rank criterion:** The observed MI ranks in the top 5 among all shuffles (configurable via `topk2`). This non-parametric check ensures the observed MI is genuinely extreme.
2. **Parametric criterion:** The fitted p-value falls below the multiple-comparison-corrected threshold.

### Stage Merging

After both stages complete, their statistics and significance results are merged. Each pair retains its Stage 1 rank (`pre_rval`) alongside Stage 2 statistics (`rval`, `pval`, `me`). The final significance is determined by Stage 2 results alone; Stage 1 serves only as a filter.

### Multiple Comparison Correction

INTENSE tests many neuron-feature pairs simultaneously, requiring correction for multiple comparisons. The correction is applied to Stage 2 p-values only (Stage 1 uses a rank criterion, not p-values). Supported methods:

- **Holm-Bonferroni** (default, `multicomp_correction='holm'`): Controls family-wise error rate (FWER). More powerful than Bonferroni while maintaining the same error guarantee. Orders p-values and tests each against a progressively relaxed threshold: `alpha / (m - k + 1)` for the k-th smallest p-value among m hypotheses.
- **Bonferroni** (`'bonferroni'`): Divides the significance threshold by the number of hypotheses. Conservative.
- **Benjamini-Hochberg** (`'fdr_bh'`): Controls false discovery rate instead of FWER. More permissive, appropriate when some false positives are tolerable.
- **None**: No correction. Uses the raw p-value threshold.

The number of hypotheses for correction is the number of pairs that entered Stage 2, not all possible neuron-feature pairs. This is a consequence of the two-stage design: Stage 1 reduces the hypothesis space before correction is applied.

## 4. Mutual Information Estimation

### Available Estimators

INTENSE supports two MI estimators, selected via `mi_estimator`:

**GCMI** (Gaussian Copula Mutual Information, default): Transforms both variables to Gaussian marginals via copula normalization, then computes MI from the correlation matrix. For two univariate variables, this reduces to `MI = -0.5 * log(1 - rho^2)` where rho is the Pearson correlation after copula normalization. Fast and robust for large samples, but captures only monotonic dependencies because the copula transformation preserves ranks.

**KSG** (Kraskov-Stoegbauer-Grassberger): A k-nearest-neighbor estimator that does not assume any functional form for the dependency. Captures non-monotonic relationships (e.g., bell-shaped tuning curves) that GCMI misses. Substantially slower because it requires building a k-d tree for each shuffle, and the per-shuffle loop cannot be accelerated by FFT.

### MI Computation by Feature Type

- **Continuous-continuous** (e.g., calcium vs speed): Both variables are copula-normalized. MI is computed as `MI = -0.5 * log(1 - rho^2)` where rho is the Pearson correlation of copula-normalized data (equivalent to Spearman rank correlation).
- **Continuous-discrete** (e.g., calcium vs trial type): MI is decomposed as `I(X;Y) = H(Y) - sum_k p(k) * H(Y|X=k)`, where class-conditional entropies are estimated on copula-normalized subsets.
- **Discrete-discrete** (e.g., two binary features): MI is computed from contingency tables using empirical probabilities.
- **Circular features**: Replaced by their (cos, sin) 2D representation before analysis. The resulting MultiTimeSeries is treated as a multivariate continuous variable.
- **Multivariate** (e.g., 2D position): MI is computed from the joint covariance matrix using closed-form determinant formulas for dimensions up to 3.

### FFT Acceleration

The FFT cache computes MI at all possible circular shifts in a single pass. For the continuous-continuous case, this means computing the cross-correlation function via FFT, converting to Pearson r at each shift, and applying the MI formula. For other pair types, analogous FFT-based batch computations handle class-conditional sums, contingency tables, and multivariate covariance blocks. The key insight is that any quantity that depends on a circular shift can be expressed as a cross-correlation, which FFT computes for all shifts simultaneously.

## 5. Anti-Selectivity Filtering (Signal Ratio)

After significance testing, INTENSE optionally filters out anti-selective neurons -- those whose activity is suppressed rather than elevated when a feature is active. This is controlled by `remove_anti_selective` (default: True).

### What Signal Ratio Measures

Signal ratio quantifies the direction of a neuron's response: `SR = mean(calcium | feature active) / mean(calcium | feature inactive)`. A ratio above 1 indicates the neuron fires more when the feature is present; a ratio at or below 1 indicates suppression or no change.

### How It Is Computed

**Binary discrete features** (e.g., locomotion, freezing): "Active" means the feature value is 1, "inactive" means 0. The calcium trace is aligned using the optimal delay before computing the ratio.

**Linear continuous features** (e.g., speed): The feature values are median-split. "Active" means above the median, "inactive" means at or below the median. This extends the signal ratio concept to continuous variables that have a natural "high" direction.

**Circular and multivariate features**: Signal ratio is not computed (set to None). These features do not have a natural "active/inactive" split, so anti-selectivity filtering does not apply to them.

### Why Pairs with SR <= 1 Are Removed

MI is symmetric and captures any statistical dependency, including suppression. A neuron that is inhibited by a feature will show significant MI just as readily as one that is excited. However, in most neuroscience contexts, "selectivity" implies excitatory encoding. Removing anti-selective pairs prevents them from inflating selectivity counts and confounding downstream disentanglement, where the assumption is that a neuron's response to its preferred feature should be positive.

## 6. Optimal Delay Finding

### Why Temporal Delays Matter

Neural responses to behavioral events are not instantaneous. Calcium indicators introduce 1-2 seconds of delay due to their binding kinetics. Predictive coding may cause neural activity to precede behavior. Sensory processing adds further latency. If these delays are not accounted for, MI between simultaneous neural and behavioral signals will be artificially reduced, causing real selectivities to be missed.

### How Optimal Delays Are Found

INTENSE scans a window of temporal shifts (default: +/-2 seconds, converted to frames using the recording's sampling rate) and selects the shift that maximizes MI for each neuron-feature pair. The function `calculate_optimal_delays` evaluates MI at every shift within the window. When the FFT cache is available, this is an O(1) lookup per shift rather than a fresh MI computation, making the delay scan essentially free.

The resulting optimal delay matrix has shape (n_neurons, n_features) and is stored in the Experiment object. All subsequent significance testing (Stages 1 and 2) uses the delay-aligned signals. Certain features can be excluded from delay optimization via `skip_delays` (e.g., features known to have zero latency).

## 7. Disentanglement

### The Problem

In natural behavior, many features are correlated. An animal's speed correlates with its locomotion state; its position correlates with which spatial zones it occupies. When a neuron is found to be significantly selective to multiple correlated features, some of those selectivities may be spurious -- arising from behavioral correlation rather than independent neural encoding. Disentanglement resolves which features a neuron truly encodes.

### Three-Phase Approach

Disentanglement runs inside `disentangle_all_selectivities` and operates only on neurons with significant selectivity to at least two features. It proceeds in three phases:

#### Phase 1: Pre-Filter Chain (Serial, Population-Level)

Before any information-theoretic computation, a composed filter chain resolves obvious cases using domain knowledge. Filters operate on three mutable dictionaries shared across all neurons:

- `neuron_selectivities`: which features each neuron responds to
- `pair_decisions`: pre-computed outcomes for specific feature pairs
- `renames`: merged feature names (e.g., "place-corners")

The filter chain typically includes:

**Priority rules** (`build_priority_filter`): Declarative rules encoding known feature hierarchies. For example, "bodydirection beats headdirection" or "freezing beats rest." When both features in a rule appear in a neuron's selectivities, the primary feature wins without computation.

**Experiment-specific rules**: Custom filters for particular experimental paradigms. For example, in NOF (Novel Object Field) experiments, a specific object identity feature (object1) takes priority over the general "objects" category, which in turn takes priority over "center."

**Spatial filter**: For experiments with discrete spatial zones (corners, walls, center), this data-driven filter checks whether a neuron's high-activity frames correspond to a specific zone. It computes the correspondence (fraction of top-2% activity frames where the zone is active). If correspondence exceeds a threshold (default: 0.4), the continuous place feature and the discrete zone are merged into a combined feature (e.g., "place-corners"). This captures the common case where a place cell's field overlaps with a named spatial region.

Filters are composable via `compose_filters` and execute in order. Later filters can override earlier decisions.

#### Phase 2: CMI Analysis (Parallel, Per-Neuron)

For each neuron, all undecided feature pairs are tested using conditional mutual information. This phase runs in parallel across neurons via joblib.

**Skip conditions:** A pair is automatically assigned result 0.5 (both features contribute) if:
- It was already decided by Phase 1.
- The two features are not significantly correlated in the behavioral data. Feature-feature significance is determined by running the same two-stage INTENSE procedure on feature pairs (via `compute_feat_feat_significance`). If two features are behaviorally independent, both selectivities are genuine by definition.

**MI lookups:** Rather than recomputing MI from scratch, the disentanglement phase looks up pre-computed values from the significance testing results: MI(neuron, feature1), MI(neuron, feature2), and MI(feature1, feature2). Optimal delays from the significance analysis are also reused.

**Conditional MI computation:** For each undecided pair (feature1, feature2) and the neuron's activity, INTENSE computes:

- `CMI(neuron, feature1 | feature2)` -- how much information feature1 provides about the neuron after accounting for feature2.
- `CMI(neuron, feature2 | feature1)` -- the reverse.

For continuous-continuous-continuous triplets, CMI is computed via the GCMI copula method (`cmi_ggg`) using pre-cached copula-normalized data with optimal delay alignment. For mixed discrete/continuous cases, a fallback path through `conditional_mi` handles the type-specific computation.

**Interaction information:** The interaction information is computed as:

```
I_av = mean(CMI(N, F1|F2) - MI(N, F1),  CMI(N, F2|F1) - MI(N, F2))
```

This quantity determines whether the features provide redundant or synergistic information about the neuron:

**Negative I_av (redundancy):** The features share overlapping information. INTENSE identifies the "weakest link" -- the feature whose MI is mostly explained by the other:

1. If MI(neuron, F1) is less than the absolute interaction information, but CMI(neuron, F2|F1) is not, then F1 is redundant and F2 is primary.
2. If neither strict criterion fires, a CMI ratio test checks whether conditioning removes more than 90% of a feature's MI. If so, that feature is redundant.
3. If both features are highly redundant, the one with higher MI is kept.
4. If no criterion resolves the pair, the result is 0.5 (undistinguishable).

**Positive I_av (synergy):** The features together provide more information than separately. INTENSE resolves these cases through dominance tests:

1. If one feature has negligible MI (below epsilon) while the other does not, AND its conditional MI direction confirms dominance (`cmi123 > cmi132` or `cmi132 > cmi123`), the strong feature is primary.
2. If one feature's MI exceeds twice the other's, AND the CMI direction confirms dominance, the dominant feature is primary.
3. Otherwise, the result is 0.5.

#### Phase 3: Post-Filter (Serial, Population-Level)

After all neurons are processed, an optional post-filter can modify results. This is used for tie-breaking in specific experimental contexts. For example, in 3D maze experiments, when disentanglement cannot distinguish between 2D place and 3D place (result 0.5), the post-filter awards the decision to the simpler 2D model.

### Decision Values

| Value | Meaning |
|-------|---------|
| 0 | Feature 1 is primary; feature 2 is redundant and removed |
| 1 | Feature 2 is primary; feature 1 is redundant and removed |
| 0.5 | Both features contribute independently (true mixed selectivity) |

Each decision also records its source: `'pre_filter'` (Phase 1), `'not_significant'` (features not behaviorally correlated), or `'standard'` (Phase 2 CMI analysis).

### The Multifeature Map

Some behavioral variables are meaningful only in combination. The x and y coordinates individually are uninterpretable, but together they define "place." The multifeature map (default: `{('x', 'y'): 'place'}`) aggregates component features into a MultiTimeSeries that is analyzed as a single entity. This aggregation happens before disentanglement: the components (x, y) are excluded from INTENSE, and only the combined feature (place) enters the pipeline.

### How Feature-Feature Correlations Inform the Process

Before disentanglement, `compute_feat_feat_significance` runs the same two-stage INTENSE procedure on all feature pairs (treating features as both "bunch1" and "bunch2"). This produces a binary significance matrix indicating which feature pairs are statistically correlated. Only correlated pairs need disentanglement. Uncorrelated pairs are automatically assigned 0.5 (both contribute), since their selectivities cannot be confounded by behavioral correlation.

The feature-feature similarity matrix (raw MI values) is also passed to the disentanglement phase for MI lookups, avoiding redundant computation.

## 8. Output Structure

### Return Values of compute_cell_feat_significance

The function returns a 5-tuple:

1. **stats** (`dict`): Nested dictionary `stats[cell_id][feat_name]` containing per-pair statistics:
   - `me`: observed MI value at optimal delay
   - `pval`: p-value from Stage 2 distribution fitting (None if pair did not reach Stage 2)
   - `rval`: rank of observed MI among Stage 2 shuffles
   - `pre_rval`: rank of observed MI among Stage 1 shuffles
   - `pre_pval`: None (not computed in Stage 1 for performance)
   - `opt_delay`: optimal delay in frames
   - `rel_me_beh`: MI normalized by feature entropy
   - `rel_me_ca`: MI normalized by neural signal entropy
   - `signal_ratio`: ratio of mean activity when feature is active vs inactive (or None)
   - `data_hash`: hash for validating cached results

2. **significance** (`dict`): Nested dictionary `significance[cell_id][feat_name]` with boolean significance at each stage:
   - `stage1`: True if pair passed Stage 1 screening
   - `stage2`: True if pair passed Stage 2 validation (this is the final significance call)

3. **info** (`dict`): Additional information including:
   - `optimal_delays`: delay matrix of shape (n_neurons, n_features)
   - `timings`: execution time breakdown (if `profile=True`)
   - `me_total1`, `me_total2`: raw MI arrays with shuffles (if stored)

4. **intense_res** (`IntenseResults`): Container object holding all of the above in a single serializable structure, plus the INTENSE parameter configuration.

5. **disentanglement_results** (`dict` or `None`): Present only when `with_disentanglement=True`. Contains:
   - `feat_feat_similarity`: MI matrix between all feature pairs
   - `feat_feat_significance`: binary matrix of significant feature correlations
   - `disent_matrix`: matrix where element [i,j] counts how many neurons had feature i as primary over feature j
   - `count_matrix`: matrix counting how many neurons were tested for each feature pair
   - `per_neuron_disent`: detailed per-neuron results (see below)
   - `feature_names`: list of feature names corresponding to matrix indices
   - `summary`: aggregate statistics (redundancy rate, true mixed selectivity rate, per-pair breakdowns)

### Per-Neuron Disentanglement Detail

Each entry in `per_neuron_disent[neuron_id]` contains:

- `pairs`: dictionary mapping `(feat_i, feat_j)` to `{'result': 0/0.5/1, 'source': str}`
- `renames`: dictionary mapping merged feature names to their component tuples
- `final_sels`: list of surviving features after removing redundant ones
- `errors`: list of any errors encountered during processing

## 9. Cross-Session Analysis

### How run_intense_analysis Wraps the Pipeline

The `run_intense_analysis` function in `tools/selectivity_dynamics/analysis.py` provides the operational wrapper used for batch processing of experimental sessions. It performs the following steps:

1. Fixes circular features erroneously normalized to [0,1] range.
2. Builds the feature list from the experiment, excluding specified skip features.
3. Filters out multi-dimensional features when using non-MI metrics.
4. Calls `compute_cell_feat_significance` with configuration from a standardized config dictionary (default: 100 Stage 1 shuffles, 10,000 Stage 2 shuffles, p-value threshold 0.001, no multiple comparison correction, downsampling factor 5).
5. Returns the full result tuple plus timing information.

The filter system (`tools/selectivity_dynamics/filters.py`) provides per-experiment-type configurations. Each experiment type (NOF, LNOF, 3DM, FOF, etc.) has a registered configuration specifying which features to aggregate, which to skip, which priority rules to apply, and which spatial zones to consider for merging. The function `get_filter_for_experiment` composes the appropriate filter chain for a given experiment type.

### How Results Flow to Population Analysis

After INTENSE runs on individual sessions, `build_disentangled_stats` applies the disentanglement decisions to produce cleaned stats and significance dictionaries. Redundant features are removed, merged features receive combined statistics (max MI, min p-value). These cleaned results can be stored in a NeuronDatabase for population-level queries across sessions, enabling questions like "what fraction of neurons in the population are place-selective after disentanglement."
