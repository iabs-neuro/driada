# Disentanglement Pipeline: End-to-End Guide

How INTENSE resolves mixed selectivity when a neuron responds to multiple correlated behavioral features.

## The Problem

A neuron selective to both "place" and "walls" might truly encode both, or it might only encode place — the correlation between place and walls makes them statistically entangled. Disentanglement uses conditional mutual information (CMI) and interaction information to determine which features carry independent information about the neuron.

## Pipeline Overview

```
run_intense_analysis.py
  │
  ├─ load_experiment_from_npz()          ← Load data, build aggregates (e.g. x,y → place)
  ├─ get_filter_for_experiment()         ← Build composed pre-filter chain
  ├─ extract_filter_data()               ← Pre-extract calcium/feature/position arrays for the place-vs-zone filter
  │
  └─ compute_cell_feat_significance()    [pipelines.py]
       │
       ├─ INTENSE Stage 1 + Stage 2      ← MI(neuron, feature) at optimal delays
       │   → cell_feat_stats[neuron][feature] = {me, pval, opt_delay, ...}
       │
       ├─ compute_feat_feat_significance()
       │   → feat_feat_similarity[i,j] = MI(feat_i, feat_j)
       │   → feat_feat_significance[i,j] = 0 or 1
       │
       └─ disentangle_all_selectivities()  [disentanglement.py]
            │
            ├─ PHASE 1: Pre-filter chain (population-level, serial)
            │   ├─ general_filter           ← Priority rules (e.g. headdirection > bodydirection)
            │   ├─ experiment-specific       ← e.g. nof_filter: object1 > objects > center
            │   └─ zone_share_filter        ← Place vs discrete zone by the zone's information share
            │   → pair_decisions[neuron][(feat_i, feat_j)] = 0/0.5/1
            │   → renames[neuron][new_name] = (old1, old2)
            │
            ├─ PHASE 2: Parallel processing (per-neuron, joblib)
            │   └─ _process_neuron_disentanglement()
            │       for each feature pair:
            │       ├─ Skip if pre-decided or feat-feat not significant
            │       ├─ Look up MI values from cell_feat_stats
            │       ├─ Look up optimal delays from cell_feat_stats
            │       └─ _disentangle_pair_with_precomputed()
            │           ├─ Apply delay shifts (np.roll on copnorm arrays)
            │           ├─ Compute CMI via cmi_ggg (aligned copnorm)
            │           ├─ Compute interaction information I_av
            │           └─ Classify: 0 (feat1 primary), 1 (feat2 primary), 0.5 (both)
            │
            └─ PHASE 3: Post-filter (population-level, serial)
                └─ e.g. tdm_post_filter: tie-break place > 3d-place
```

## Phase 1: Pre-Filter Chain

Before any MI-based disentanglement runs, a composed filter chain pre-decides obvious cases. Filters mutate three shared dictionaries:

- **`neuron_selectivities`**: `{neuron_id: [feat1, feat2, ...]}`
- **`pair_decisions`**: `{neuron_id: {(feat_i, feat_j): 0/0.5/1}}`
- **`renames`**: `{neuron_id: {new_name: (old1, old2)}}`

### Filter 1: General Priority Rules

Declarative rules based on domain knowledge. When both features are in a neuron's selectivities, the primary wins without computation.

```
bodydirection > headdirection
bodydirection_2d > headdirection_2d
freezing > rest
locomotion > speed
rest > speed
freezing > speed
walk > speed
```

Applied via `build_priority_filter()`. Sets `pair_decisions[nid][(primary, redundant)] = 0`.

> **Note:** Circular features (e.g., head direction, body direction) are substituted
> with their `_2d` (cos, sin) representation by the main pipeline before disentanglement.
> Priority rules include both original and `_2d` variants to handle both naming conventions.

### Filter 2: Experiment-Specific Rules

Each experiment type (NOF, LNOF, 3DM, FOF) has custom rules:

- **NOF**: `object1 > objects > center` — specific objects beat general categories
- **3DM**: `3d-place > z`, `start_box > 3d-place`, `speed > speed_z`

### Filter 3: Place-vs-Zone Filter

For experiments with discrete spatial features (corners, walls, center, objects), decides between place and each discrete zone a neuron is selective to. The rule is chosen by `--zone-rule` (`zone_rule` in `get_filter_for_experiment`).

**`place_field` (default, `place_field_filter`)** — a neuron is a zone cell when its main place field lies in the zone. For every (neuron, zone) pair:

1. Shift calcium by the INTENSE optimal delay of the pair
2. Build the neuron's activity map (20×20 bins, occupancy-normalised, Gaussian smoothing 1.5 bins); never-visited bins are left out
3. Fields = connected regions above half of the peak, the map minimum being the baseline; the main field holds the peak
4. The zone holds the main field if the peak bin or one of its neighbours belongs to the zone (more than half of the time spent in that bin was spent in the zone) and the fields lying in the zone (their own peak is in the zone) together carry more than half of the activity of all fields
5. If so → merge into a combined feature (e.g. `place-corners`); of several such zones the one covering the largest part of the main field is merged and the others lose to place and are removed from the neuron together with the merge (a pair decision naming place or the merged zone could no longer be applied). Otherwise place wins, `(place, zone) = 0`

The map is normalised by occupancy, so the decision does not depend on how long the animal stays in the zone. Fields in the zone are counted together because a zone can be a set of separate places (four corners) that one cell covers with several fields.

**`information_share` (`zone_share_filter`)** — for every (neuron, zone) pair:

1. Shift calcium by the INTENSE optimal delay of the pair and copula-normalise it
2. Build the neuron's activity map (20×20 bins, Gaussian smoothing 1.5 bins, occupancy-normalised), cut the frames into 8 equal-duration levels of the map value at the visited bin; position label = level × zone (16 classes)
3. `I_pos` = MI(calcium, position label), `I_zone` = MI(calcium, zone), Gaussian class-conditional estimator with bias correction
4. Subtract from both the mean over 100 circular shifts of calcium (at least 20 s, fixed seed), with the map rebuilt for every shift
5. share = corrected `I_zone` / corrected `I_pos`
6. If share ≥ 0.5 and mean calcium inside the zone is above the mean outside → the zone wins, `(place, zone) = 1`. Otherwise place wins, `(place, zone) = 0`

Nothing is merged; several zones of one neuron are decided independently.

**`top_activity` (`spatial_filter`)** — checks whether a neuron's high-activity frames (top 2%) correspond to the discrete zone:

1. Compute correspondence = fraction of high-activity frames where zone is active
2. If correspondence > 0.4 → merge into combined feature (e.g. `place-corners`)
3. Set pair decisions and renames accordingly

All three run only when `discrete_place_features` is non-empty (NOF, LNOF experiments).

## Phase 2: Information-Theoretic Disentanglement

For each neuron with ≥ 2 significant features, all undecided feature pairs go through MI-based analysis. This runs in parallel across neurons via joblib.

### Step 1: Skip Checks

A pair is skipped (result = 0.5) if:
- Already pre-decided by Phase 1
- `feat_feat_significance[i,j] == 0` — features are not behaviorally correlated, so both selectivities are independent (true mixed selectivity)

### Step 2: Gather Pre-Computed Values

From INTENSE Stage 2 results (`cell_feat_stats`):
- `mi12 = MI(neuron, feat1)` — delay-optimized
- `mi13 = MI(neuron, feat2)` — delay-optimized
- `delay1 = opt_delay(neuron, feat1)` — in original frame units
- `delay2 = opt_delay(neuron, feat2)` — in original frame units

From feat-feat analysis (`feat_feat_similarity`):
- `mi23 = MI(feat1, feat2)` — at zero delay (features are synchronous)

### Step 3: Apply Optimal Delays

INTENSE computes MI at per-neuron-per-feature optimal circular shifts. The same delays must be applied when computing CMI, otherwise the interaction information mixes delay-optimized MI with zero-delay CMI.

For the copula-normalized (CCC) path:
```python
delay_ds = delay_frames // ds                     # Convert to downsampled units
ts2_shifted = np.roll(ts2_copnorm, delay_ds)      # Align feat1 with neural activity
ts3_shifted = np.roll(ts3_copnorm, delay_ds)      # Align feat2 with neural activity
```

For the MI fallback path (when MI is not pre-computed):
```python
mi12 = get_mi(neural, feat1, ds=ds, shift=delay_ds)
```

Feature-feature MI (`mi23`) stays at zero delay — behavioral features are measured synchronously.

### Step 4: Compute Conditional MI

Two CMI values via GCMI copula method (`cmi_ggg`):
- `CMI(neuron, feat1 | feat2)` — feat1's unique information after accounting for feat2
- `CMI(neuron, feat2 | feat1)` — feat2's unique information after accounting for feat1

### Step 5: Interaction Information

```
I_av = mean(CMI(N,F1|F2) - MI(N,F1),  CMI(N,F2|F1) - MI(N,F2))
```

- **I_av < 0 → Redundancy**: Features share information about the neuron. Identify which is the "weakest link":
  - If `MI(N,F1) < |I_av|` but `CMI(N,F2|F1)` isn't: F1 is redundant → result = 1 (F2 primary)
  - Fallback: CMI ratio test — if conditioning removes > 90% of MI, that feature is redundant
  - Fallback: if both are highly redundant, keep the one with higher MI

- **I_av ≥ 0 → Synergy or Independence**: Features provide complementary information.
  - If one feature has negligible MI but the other doesn't → the strong one is primary
  - If one MI is > 2x the other → the dominant one is primary
  - Otherwise → 0.5 (both contribute, undistinguishable)

### Decision Values

| Value | Meaning |
|-------|---------|
| 0 | Feature 1 (ts2) is primary, Feature 2 is redundant |
| 1 | Feature 2 (ts3) is primary, Feature 1 is redundant |
| 0.5 | Both features contribute — true mixed selectivity |

## Phase 3: Post-Filter

After all neurons are processed, an optional post-filter can modify results. Currently used only for 3DM experiments:

- **`tdm_post_filter`**: When `(place, 3d-place)` gets result 0.5 (undistinguishable), change to 0 (place wins). Rationale: prefer the simpler 2D model when information theory can't distinguish.

## Output

`disentangle_all_selectivities()` returns:

```python
{
    'disent_matrix':     np.array,   # [i,j] = count of times feat_i was primary over feat_j
    'count_matrix':      np.array,   # [i,j] = number of neurons tested for this pair
    'per_neuron_disent': {           # Per-neuron detail
        neuron_id: {
            'pairs': {(feat_i, feat_j): {'result': 0/0.5/1, 'source': str}},
            'renames': {new_name: (old1, old2)},
            'final_sels': [surviving features after removing redundant ones],
            'errors': [(neuron_id, pair, error_msg), ...],
        }
    }
}
```

The `source` field tracks how each decision was made:
- `'pre_filter'` — decided by Phase 1 filter chain
- `'not_significant'` — features not behaviorally correlated (auto 0.5)
- `'standard'` — decided by interaction information analysis

## Configuration

Key parameters in `run_intense_analysis.py`:

| Parameter | Default | Source |
|-----------|---------|--------|
| `ds` | 5 | CLI `--ds` |
| `pval_thr` | 0.001 | CLI `--pval` |
| `feat_feat_pval_thr` | 0.01 | Hardcoded in pipeline |
| `find_optimal_delays` | True | Hardcoded |
| `mi_ratio_threshold` | 1.5 | `filter_kwargs` |
| `correspondence_threshold` | 0.4 | `filter_kwargs` |
| `n_jobs` | -1 | All cores |

## Example: NOF Experiment

A NOF neuron is selective to `[place, corners, object1, speed]`:

1. **General filter**: `rest > speed` — no match (no rest). No decisions.
2. **NOF filter**: `object1 > center` — no match (no center). No decisions.
3. **Place-vs-zone filter**: neuron has place + corners + object1. Corners carry 70% of its position information and it is more active in the corners → `(place, corners) = 1`. object1 carries 10% → `(place, object1) = 0`.
4. **Phase 2**: Remaining undecided pairs go through CMI analysis:
   - `(corners, object1)`: feat-feat significant? If yes → compute interaction information at optimal delays.
   - `(place, speed)`, `(corners, speed)`, `(object1, speed)`: feat-feat not significant → auto 0.5 (true mixed selectivity).
5. **Result**: `final_sels = [corners, speed]` (place lost to corners, object1 lost to place).
