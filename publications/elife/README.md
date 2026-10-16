# DRIADA eLife paper: figure and analysis scripts

Scripts that produce the figures and the numbers quoted in the text of the
DRIADA paper submitted to eLife. Each figure has one script; computations that
feed several figures live in `compute/`. All paths come from `elife_paths.py`.

## Requirements

- DRIADA 1.2 or newer (`pip install -e .` from the repository root). Version 1.2
  brings the `place_field` rule for neurons selective to both place and a zone
  and the fix for aggregate feature types; older versions give different
  selectivity tables.
- The `tools/` folder of the repository (`run_intense_analysis.py`,
  `selectivity_dynamics`, `neuron_database`, `load_synchronized_experiments.py`).
  It is found automatically when the scripts run from a checkout; otherwise set
  `DRIADA_TOOLS`.
- Extra packages: `colorstamps` (Figures 3-5), `PyMuPDF` and `Pillow`
  (Figure 1), `psutil` (`compute/run_intense.py`), `torch` (autoencoders of
  Figure 5), `umap-learn` (Figure 3A when its cache is absent).

## Where things are

| Variable | Meaning | Default |
|---|---|---|
| `DRIADA_ELIFE_DATA` | root of the experimental data | `publications/elife/data` |
| `DRIADA_ELIFE_CACHE` | intermediate results | `publications/elife/cache` |
| `DRIADA_ELIFE_OUTPUT` | figures, tables, logs | `publications/elife/output` |
| `DRIADA_ELIFE_INTENSE` | output folder of the INTENSE run | `<data root>/INTENSE` |
| `DRIADA_TOOLS` | `tools` folder of the library | `<repository>/tools` |

Expected layout of the data root:

```
synchronized/<session>_aligned.npz     39 sessions of the free open field experiment
Mice_info.csv                          animal metadata (cross-session figure)
INTENSE/by_type/results/               written by compute/run_intense.py
INTENSE/by_type/tables/
INTENSE/by_type/tables_disentangled/
```

### Data availability

The calcium imaging dataset (synchronized activity and behaviour of the free
open field sessions) will be published together with a separate data paper.
Until then it is available from the authors on request. Figures 1, 2 and 5
and the supplementary figure on autoencoder initializations need no
experimental data.

## Figures

Times are rough, on a laptop, with the caches in place.

| Figure | Script | Inputs | Output | Time |
|---|---|---|---|---|
| 1 | `fig1/fig1_architecture.py`, `fig1_objects.py`, `fig1_network.py`, `fig1_traces.py`, then `fig1/fig1_compose.py` | `fig1/fig1_assembled.pdf` (see "Not reproduced by scripts") | `fig1.png/pdf` and the four component drawings | 3 min |
| 2 | `fig2/fig2_data.py`, then `fig2/fig2_synthetic.py` | none (synthetic population, seeded) | `cache/fig2_*.npz`, `fig2.png/pdf` | 3 min + 1 min |
| 3 | `fig3/fig3_composite.py` (panels: `panel_a.py`, `panel_b.py`, `panel_e.py`) | synchronized file of session `FOF_F28_1D`, INTENSE results and disentangled tables of all sessions, `cache/FOF_F28_1D_umap_cache.npz` | `fig3.png/pdf` | 6 min |
| 4 | `fig4/fig4_composite.py` (network panels: `panel_c_network.py`) | INTENSE results of `FOF_F28_1D`, `cache/fig4_ab_cache_nn40.npz`, `cache/fig4_multisession_sessions.csv`, `cache/network_*.npz` | `fig4.png/pdf` | 2 min |
| 5 | `fig5/fig5_data.py`, then `fig5/fig5_advanced.py` | none (simulated attractor network, seeded); `cache/fig5_ae_sweep.npz` from `supplementary/dci_sweep.py` | `cache/fig5_*.npz`, `fig5.png/pdf` | data: not timed end to end (minutes to tens of minutes); drawing 1 min |
| Supp. (cross-session stability) | `supplementary/supp_cross_session_stability.py` | disentangled tables of all sessions, `Mice_info.csv` | `supp_cross_session_stability.png/pdf` | 2 min |
| Supp. (initializations) | `supplementary/dci_sweep.py`, then `supplementary/supp_dci_initializations.py` | none | `cache/fig5_ae_sweep.npz`, `supp_dci_initializations.png/pdf` | sweep: about an hour or more; drawing seconds |

## Upstream computations (`compute/`)

| Script | What it does | Time |
|---|---|---|
| `run_intense.py` | INTENSE selectivity for every session: type-based feature representations, `place_field` zone rule, seed 42, frame rate 30. Pass session names to process a subset. | many hours for 39 sessions |
| `compute_cell_cell_network.py` | cell-cell significance matrices of `FOF_F28_1D`, real and shuffled (`cache/network_real_*.npz`, `network_shuffled_*.npz`) | hours |
| `run_multisession.py` (runs `multisession_fig4_controls.py` in three parallel groups) | Figure 4 analyses on every session with random-subset and shuffle controls; writes `output/multisession/results_all/sessions.csv` | hours |
| `paired_stats.py` | paired comparisons over sessions and mice from the cohort table | seconds |
| `network_assortativity.py` | assortativity of the networks by selectivity label with a label-permutation null | 4 min |
| `assortativity_degree_controls.py` (helpers in `assortativity_common.py`) | the same with permutation inside groups of similar degree and with degree-preserving rewiring | 4 min |
| `compute_procrustes_protest.py` | permutation test of the Procrustes fit of Figure 3A | seconds |
| `trace_fig3.py`, `trace_fig4.py` | every number of Figures 3 and 4 quoted in the text | seconds |
| `set_high_qos.py` | Windows only: lifts power throttling from background processes | - |

## Order

1. `compute/run_intense.py`
2. `compute/compute_cell_cell_network.py`, `compute/run_multisession.py`; copy
   `output/multisession/results_all/sessions.csv` to
   `cache/fig4_multisession_sessions.csv`
3. `fig2/fig2_data.py`, `fig5/fig5_data.py`, `supplementary/dci_sweep.py`
4. the figure scripts, in any order
5. `compute/trace_fig3.py`, `compute/trace_fig4.py`, `compute/paired_stats.py`,
   the assortativity scripts

Steps 2 and the heavy part of step 3 can be skipped: their results are in
`cache/`.

## Shipped caches

`cache/` holds the small intermediate results the manuscript figures were drawn
from: the UMAP embedding of Figure 3A, the network matrices and layout, the
panel A/B results and the cohort table of Figure 4, the Figure 5 results and
the autoencoder sweep. Two kinds of cache are too large to ship and are written
on demand:

- `fig5_rnn.npz` (simulated activity): `python fig5/fig5_data.py --simulation-only`
  recreates it in seconds, bit for bit.
- `fig2_*.npz`: `python fig2/fig2_data.py`.

## What is exactly reproducible

With the shipped caches and the same INTENSE results, Figures 3, 4 and 5 and
the initialization figure are redrawn pixel for pixel. Not everything
recomputes to the same bits:

- **Figure 3A**: the UMAP embedding is not seeded; without the cache the cloud
  differs in detail from run to run.
- **Isomap** (Figure 4A-B and the multisession control): repeated embeddings
  of the same data are not bit-identical, so a rerun of
  `multisession_fig4_controls.py` gives close but not identical values, with
  the largest spread in sessions where the embedding fits position poorly.
- **Cross-session stability, panel B**: the enrichment test draws unseeded
  shuffles; the bars move slightly between runs. Panel A (counts) is exact.
- **Figure 2**: the population and the detection scores (precision, recall,
  F1) recompute exactly; mutual information values, the Isomap coordinates and
  the network partition scores change slightly with the library version.
- **Network matrices** in `cache/network_*.npz` were computed with an earlier
  version of the library; the script is provided as it was run.

## Not reproduced by scripts

Figure 1 was arranged by hand: the four component drawings were placed and
annotated in a slide deck, and the deck was exported to `fig1/fig1_assembled.pdf`
(two pages, panels A and B). `fig1_compose.py` only joins these two pages and
adds the panel letters. The slide deck itself is not part of the repository.
