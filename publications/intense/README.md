# INTENSE paper: scripts behind the analyses, numbers and figures

Scripts of the paper "INTENSE: Information-theoretic discovery of behavioral selectivity of
neurons in calcium imaging data" (Pospelov et al.). They compute the numbers, tables and figures of
the main text, Methods, Extended Data and Supplementary Information.
[`numbers_v12.csv`](numbers_v12.csv) names, for each of them, the script, its input and the output
file and key that hold the value. Run every command from this folder; on Windows use `python -X utf8`, so
that a script that prints names with non-ASCII characters does not stop on the console encoding.

| Folder | What it reproduces |
|---|---|
| `compute/` | session files from DANDI, INTENSE with disentanglement, the databases, the runs with correlation |
| `numbers/` | numbers of Results and Methods and Supplementary Table 7 |
| `fig1/`, `fig3/`, `fig2/` | caches of Figs 1 and 3; `fig2/` and `fig1/cache_examples.py` only document inputs made from an earlier export (see Figures) |
| `ed_calibration/` | calibration of false discoveries (Methods), Extended Data Fig. 1 |
| `classic/` | classic place-cell method of Fig. 2 |
| `synthetic/` | synthetic benchmark of Fig. 4 |
| `fft/` | speed-up of the FFT cache (Methods) |
| `glm/` | Supplementary Section S1 without MEC: Supplementary Tables 3, 4, Figs 1-3; `glm/saved/`: saved outcomes |
| `mec/` | Supplementary Section S1.5: Supplementary Table 5 and the data of Figs 5-7 |
| `s2/` | Supplementary Section S2: Supplementary Table 6 and the data of Fig. 8 |
| `figure_chain/` | Figs 1-4, Supplementary Figs 4-8 |
| `source_data_builders/` | Source Data workbooks |

In `numbers_v12.csv` and below, `[section] key` is an entry of `config.toml`, and `<db_dir>`, `<cache_dir>`, `<numbers_dir>`,
`<synthetic_dir>`, `<calibration_dir>` are the entries of that name.

## Software

* **DRIADA 1.2.2.** Release `v1.2.2-intense` of the repository is release 1.2.2 with this folder added;
  `config.toml` points at `../../src` and `../../tools` of the same checkout, and every script stops
  if it imports another version. To use an installed package instead (`pip install driada==1.2.2`),
  set `[driada] src = ""` and `tools` to the `tools/` folder of the release.
* **Python 3.11 or newer** (`tomllib`). The paper was computed with Python 3.13.5, numpy 2.1.3,
  scipy 1.15.3, pandas 2.2.3, matplotlib 3.10.0, scikit-learn 1.6.1, statsmodels 0.14.4, numba 0.61.0,
  joblib 1.5.3, h5py 3.12.1, ssqueezepy 0.6.6 (wavelets of Fig. 2a), seaborn 0.13.2,
  matplotlib-venn 1.1.2, starbars 3.1.1 (brackets of Fig. 3c), scikit-image 0.25.0 (map similarity of
  Fig. 2e), Pillow 11.1.0, PyMuPDF 1.27.1, openpyxl 3.1.5, psutil 5.9.0.
* **MEC, the steps that unpack arrays of the dump:** Python 3.8.20, datajoint 0.12.7 (the version
  that wrote the dump), numpy 1.24.4, tomli.
* Arial installed (the journal font); a figure script warns when it falls back to another font.

## Settings that decide the numbers

| Setting | Value | Where it is set |
|---|---|---|
| random seed | 42 | `[analysis] seed`: INTENSE shuffles, the tie-breaking noise added when an experiment is built, calibration nulls, synthetic data of Fig. 4 |
| significance of a neuron-feature pair | p < 0.001, no correction for multiple comparisons | `[analysis] pval_thr`; DRIADA 1.2.2 `tools/selectivity_dynamics/config.py` |
| excess information | more than 0.04 bits, applied before disentanglement | `[analysis] mi_excess_thr`; not applied in Fig. 4, Supplementary Section S1 and the MEC data |
| shuffles | 100 in stage 1; 10,000 requested in stage 2, capped by DRIADA at the number of distinct circular shifts (n frames // ds: 2,369 or 3,571-3,572 per session) | DRIADA 1.2.2 defaults |
| downsampling `ds` | 5 (MEC: 2) | DRIADA 1.2.2 defaults; `mec/run_xyfix.py` |
| frame rate | each session its own: 26 sessions at about 19.8 frames/s, 38 at about 29.8 ([`fps_by_session.csv`](fps_by_session.csv)); the delay window, the excluded shifts and the delay categories follow it. MEC: [`mec/si_sessions_fps.csv`](mec/si_sessions_fps.csv), [`mec/fps_by_session.csv`](mec/fps_by_session.csv) | `[v11] fps_table`, written into the session files by `compute/make_v2_data.py`; `[mec] fps_table`, `[mec] fps_all` |
| feature representation | type-based (`by_type`) | `[analysis] representation` |
| zone rule | `place_field` | `[analysis] zone_rule` |
| conditional MI of disentanglement | at the optimal delays of the pair | `[v11] cmi_delays = "aligned"` (`compute/run_disent_pair_delays.py`); DRIADA alone gives 872 instead of 868 |
| delay categories | negative, near zero, positive at round(0.75 s x frame rate) frames | `compute/build_db_fps.py --round-boundary` |
| Supplementary Section S1, synthetic | seeds 42, 123, 7, 11, 19, 23, 31, 44, 57, 64, 73, 88, 91, 101, 113, 127, 139, 151, 163, 177 (20 for Table 3 and Fig. 1, the first 15 for three-feature mixtures, the first 10 for Fig. 2, matched FPR and the GLM controls); INTENSE seed 42, p < 0.001 per pair in Table 3, p < 0.05 with Holm correction in Figs 1, 2; no delay search; bootstrap seed 0 | `glm/synth_common.py`, `glm/synth_compute.py`, `glm/synth_numbers.py` |
| GLM | 24 head-direction bins, 10 quantile speed bins, 7 x 7 position bins, ridge alpha 1, 10 block folds, one-sided Wilcoxon test, alpha 0.05 with Bonferroni correction | `glm/glm_variants.py`, `glm/ca1_glm.py`, `mec/analyze_cohort.py` |
| Supplementary Section S2 | calibration seed 20261102 (B = 2,000), S7 seed 1102 (B = 1,000), test seed 20261104; **run `s2/16_significance.py 2000`: the paper used 2,000 draws, the default is 10,000** | scripts of `s2/` |
| MEC | fps of each session, no 0.04-bit threshold, no disentanglement, x and y declared linear, seed 42, `n_jobs = 1` | `mec/run_xyfix.py` |

**What a rerun gives.** DRIADA 1.2.2 draws a small part of its noise without a fixed seed. On the recorded
data this touches no number: rerun from the data, the second stage of a session gives the results of the paper
byte for byte except `rel_me_ca` of the statistics tables, which nothing uses; the databases and the tables of
numbers built on them are identical; and of the figures only the density curve of Fig. 1d moves slightly between
renders (noise of amplitude 1e-8 added to the calcium when a neuron is built). On the synthetic cells of
Supplementary Section S1 the generator adds tie-breaking noise without a seed, so a rerun from scratch gives F1
values within 0.02 of the printed ones (in a repeat of three seeds 122 of 135 values were identical, the others
differed by one cell). The printed values come from the saved outcomes per cell, which are in this folder:
`glm/saved/`, about 4 MB (see Supplementary Section S1 below).

## Data

| Input | Where to get it | Where the scripts read it |
|---|---|---|
| CA1 recordings: 64 sessions of 16 mice (calcium traces, behavior, zone masks) | DANDI, Dandiset 002047 (one NWB file per session) | `python compute/data_from_dandi.py <dandiset folder>` writes the session files to `[v11] nof_v1` and the zone masks to `[data] zone_masks` |
| frame rate of each session | `fps_by_session.csv` in this folder | `[v11] fps_table` |
| MEC recordings (Obenhaus et al., 2022) | NIRD Research Data Archive, doi:10.11582/2022.00005, file `dump.sql` (101,479,145,429 bytes, sha256 in `config.toml`); the scripts read 17 of its 178 tables, about 19.2 GB | `[mec] dump` |
| synthetic data (Fig. 4, Supplementary Sections S1, S2) | generated by the scripts | - |
| small inputs of four example panels, saved outcomes of Section S1 | in this folder: `figure_chain/saved_inputs/`, `glm/saved/` | see Figures and Section S1 below |

Everything is located by `config.toml`. By default the inputs go under `data/` and the results under
`results/` next to this file; set the environment variables `INTENSE_DATA` and `INTENSE_RESULTS` to move
them, or edit `config.toml` (or point `INTENSE_PAPER_CONFIG` at a copy).

## Order of the steps

Times are for one machine with 8 cores, from the logs of the runs that produced the paper unless marked as
an estimate. Most steps skip what is already computed, so an interrupted step can be started again; the figure
scripts refuse to overwrite their outputs.

### CA1: INTENSE, disentanglement and the numbers of Results and Methods

| Step | Command | Reproduces | Time |
|---|---|---|---|
| 0a | `python compute/data_from_dandi.py <dandiset folder>` | session files, zone masks | 3 min |
| 0b | `python compute/make_v2_data.py` | session files with the frame rate of `fps_by_session.csv` in `_metadata['fps']` (`[data] nof_aligned`) | minutes |
| 1 | `python compute/run_intense_interaction.py --output-dir <results>/stage2 --shard 0/2` and `--shard 1/2` side by side | INTENSE of the 64 sessions with the interaction information of the pairs tested by conditional MI (`[v11] stage2_results`) | 1.6 h for the two shards |
| 2 | `python compute/run_disent_pair_delays.py [--shard k/N]` | disentanglement with conditional MI at the delays of the pair, replayed from step 1 (with `--full` the second stage is computed instead and step 1 can be skipped) | 45 s per session (replay); 1.5-4 min per session (`--full`) |
| 3 | `python compute/build_db_fps.py --round-boundary` | `<db_dir>/NOF_pre_disent.csv`, `NOF_post_disent.csv` | 1 min |
| 4 | `python numbers/recount_v8.py`, `python numbers/ii_numbers.py` | 25.8%, 19.0%, 1,480 to 868, 41.4%; 260 pairs, 92.7%, 54.2% | 1 min |
| 5 | `python fig1/fig1_data.py`, `python fig3/fig3_data.py`, then `python numbers/fig3c_by_mouse.py` | caches of Figs 1e, 3c, 3f; 90.0%, 92-100%, 89-94%; the Wilcoxon tests of Fig. 3c over sessions and mice | minutes |
| 6 | `python compute/run_pearson.py`; `python compute/corr_by_type.py --variant rmult --out <results>/corr/rmult` | correlation instead of MI on the same sessions | 40 min (all cores); 2.2 h (one core) |
| 7 | `python numbers/overlap_mi_corr.py`, `python numbers/gamma_tail_mi_corr.py` | 54%, 82%, 30%, 22,402, 19,040, 75%, 80%, 5,297; 1.1%, 2.7% | 5-15 min |
| 8 | `python ed_calibration/calib_run_part_fps.py k/6` for k = 0..5 side by side, then `python ed_calibration/analyze.py` | false discoveries (Methods): 5.1, 6.4, 3.4 per 1000, 0.45, 0.60, 0.06 per 1000, 2-3% | 1.9 h |
| 9 | `python classic/run_classic_all.py --n_jobs 4` | events and spatial-information z of every neuron (classic method of Fig. 2), `[v11] classic_dir` | 15-45 min (estimate) |

### Synthetic benchmark (Fig. 4, Results "Synthetic validation")

| Step | Command | Reproduces | Time |
|---|---|---|---|
| 10 | `python synthetic/run_benchmark.py --workers 4`, `python synthetic/naive_pvalues.py`, `python synthetic/naive_common_threshold.py --src <synthetic_dir>/full --out <synthetic_dir>/full_thr001` | `<synthetic_dir>/full/numbers_text.csv`, `full_thr001/summary_table.csv`: 47 of 54, 0.99, 0.92, 2-8%, 0.58, 0.61, 6-13% | 8-11 h with 4 workers, 5 GB of memory each (estimate from a timed quick run) |
| 11 | `python numbers/fig4_numbers.py` | skip probability 0.3 (F1 0.949 at 0.4), 0.98-0.99, 6-13% | seconds |

### FFT speed-up (Methods, 130 to 220 times)

| Step | Command | Time |
|---|---|---|
| 12 | `python fft/d_fft_timing2.py prepare`; `python fft/d_fft_core.py c 5` and `python fft/d_fft_core.py d 5` side by side, needs step 10 | 2.3 min; 9.7 min each |

The ratio is `summary.ratio_median.core_gcycles` of `<[fft] out_dir>/fft_core_{c,d}.json` (134.0; 224.3). It
counts CPU cycles with a Windows call and depends on the machine and its load: a rerun reaches the same
decision and a ratio of this order, not the same digits.

### Supplementary Section S1: comparison with the GLM on synthetic data and on CA1

| Step | Command | Reproduces | Time |
|---|---|---|---|
| 13 | `python glm/synth_compute.py --worker K` for K = 1..4 side by side | per-cell outcomes, `<[glm] synth_dir>/parts` | 42 min |
| 14 | `python glm/synth_numbers.py`, `python glm/synth_figures.py` | Supplementary Table 3, Figs 1, 2; 0.004-0.07 bits, 88%; 0.40, 0.12, 0.27; 0.26-0.30; nearly four, 2.7 | seconds |
| 15 | `python glm/mixtures_breakdown.py`, `python glm/mixtures_components.py --workers 4`, `python glm/mixtures_summary.py` | 1.5-1.9 times; half of the modulation | 2 min |
| 16 | `python glm/matched_fpr.py`, `python glm/matched_fpr_numbers.py`, `python glm/matched_fpr_figure.py` | Supplementary Fig. 3; 0.76, 0.88, 0.48; 0.67, 0.85, 0.34; 1/201 | 19 min |
| 17 | `python glm/ca1_glm.py --shard k/2` for k = 0, 1 side by side, then `python glm/ca1_glm_numbers.py` | GLM columns of Supplementary Table 4, `[v11] glm_ca1_per_neuron` (needed by steps 18, 19 and Supplementary Fig. 4) | 90 min |
| 18 | `python glm/ca1_intense_zero_lag.py --delays off --shard k/3 --n-jobs 1` for k = 0, 1, 2 side by side, then `python glm/ca1_numbers.py` | INTENSE columns of Supplementary Table 4; more than ten times; 65 against 86; 393 | 90 min |
| 19 | `python numbers/speed_direction_ca1.py` | 45 of 72 | seconds |

**The exact printed numbers of Sections S1.1-S1.3 and S1.6.** The generator of the synthetic cells adds
noise without a seed (see above), so step 13 lands within 0.02 of the printed F1, and step 16 likewise. To
get the printed values, skip the computation and use the saved outcomes:

    mkdir -p results/glm/synth results/glm/matched_fpr
    cp -r glm/saved/synth/* results/glm/synth/
    cp glm/saved/matched_fpr/* results/glm/matched_fpr/
    python glm/synth_numbers.py && python glm/synth_figures.py && python glm/mixtures_breakdown.py
    python glm/mixtures_components.py --workers 4 && python glm/mixtures_summary.py
    python glm/matched_fpr_numbers.py && python glm/matched_fpr_figure.py

On the saved outcomes `glm/synth_numbers.py` writes the tables of the paper byte for byte, and Supplementary
Figs 1-3 come out byte for byte as printed.

### Supplementary Section S2: disentanglement on synthetic neurons with known selectivity

| Step | Command | Reproduces | Time |
|---|---|---|---|
| 20 | `python s2/select_sessions.py` | the 16 sessions, one per mouse | seconds |
| 21 | `python s2/s2_synth_fps_16.py make`, `python s2/7_s7_make.py` | synthetic neurons on the recorded behavior | 8 min; 4 min |
| 22 | `python s2/s2_synth_fps_16.py run`, `python s2/7_s7_run.py` | their INTENSE and disentanglement | about 1 h; 2-3 min per session |
| 23 | `python s2/s2_synth_fps_16.py score`, `s2/3_ca1_table.py`, `s2/2_rescore_ca1rule.py`, `s2/table6.py`, `s2/8_s7_score.py` | Supplementary Table 6; 335 of 338; 640; 330 of 330 | seconds |
| 24 | `python s2/4_calibration.py`, `python s2/11_s3_with_s7.py`, `python s2/16_significance.py 2000` with one BLAS thread each | 4.5%, 11.2%, 0 of 2,000 draws, p = 1.5e-5 | 11 min each |
| 25 | `python s2/15_exact_intervals.py`, `python s2/5_composition.py`, `python numbers/s2_numbers.py`, `python numbers/wilcoxon_table.py` | 3.7-5.4%, 544 of 868; Supplementary Table 7 | seconds |
| 26 | `python numbers/place_cell_agreement.py` (after step 30) | 60%, 94%, 40%, 84%; 13,191, 3,370, 1,036, 347 | seconds |

### Supplementary Section S1.5: medial entorhinal cortex (MEC)

The main folder of this analysis is `mec/`. Steps marked A need the MEC environment above, the others the
main one. The comparison with the authors' labels uses 213 of the 215 sessions: we have no copies of the authors'
scores for the two added sessions, the two that are not in `mec/si_sessions_fps.csv`, so the cell list of step 27d
leaves them out.

| Step | Command | Reproduces | Time |
|---|---|---|---|
| 27a | A: `python -B mec/extract_tables.py validate`, `python -B mec/extract_dataset_map.py` | small tables of the dump, the signal dataset the authors scored in each session | 2 min for the map |
| 27b | A: `python -B mec/decode_cohort_primary.py` (after an interruption: `mec/find_resume_offset.py`, `mec/finish_decode_then_recompute.py`, `mec/decode_cohort_primary_resume.py`) | 215 session files in `[mec] sessions_dir` | about 1 h |
| 27c | `python -X utf8 -B mec/run_recompute_primary.py`: runs `mec/run_xyfix_primary.py` three times for INTENSE and three times for the GLM, all at once | INTENSE and GLM of every session, `<[mec] recompute_dir>/parts` | 2-2.5 h with 6 processes; about 5.6 h per pass in one process |
| 27d | `python -X utf8 -B mec/numbers_primary.py`, which while there are no labels only writes the analysed cells, `<[mec] recompute_dir>/variants/new_on/cell_sets.jsonl`, and stops; then `python -B mec/arbiter_labels.py` (reads 3.3 GB of score tables of the dump) | the authors' labels at the 95th and 99th shuffle percentiles | 21 min |
| 27e | `python -X utf8 -B mec/numbers_primary.py`: the numbers of the MEC analysis, from the parts of step 27c, the session files and the labels of step 27d. It writes into `<[mec] recompute_dir>`: `numbers_primary.json` (four variants: with and without the delay search and the 0.04-bit threshold; on all analysed sessions and on the 213 of `mec/si_sessions_fps.csv`, whole and by imaging configuration; the comparison with the labels on the sessions that have them); the parts joined (`cohort_results_primary.csv`, `cohort_cell_sets_primary.jsonl`, `pairs_primary.csv`); `variants/new_on/` and likewise `new_on_mi04/`, `new_off/`, `new_off_mi04/` with `results.csv`, `cell_sets.jsonl` on the sessions with labels and what `mec/arbiter_join.py`, `mec/arbiter_stats.py`, `mec/confident_by_config.py` write from them; `variants/new_on/results_all.csv`, `cell_sets_all.jsonl` on all analysed sessions | Supplementary Table 5; the numbers of Sections S1.5, S1.5.1 and of the MEC paragraph of Results | minutes |
| 27f | optional check, A: `python -B mec/sync_fps.py`, `python -B mec/join_fps.py > <[mec] fps_dir>/summary.txt`, `python -B mec/list_over_1pct.py` | must rebuild `mec/si_sessions_fps.csv`, `mec/fps_by_session.csv` | 3 min |

### Figures

Everything below writes under `<[figures] output_dir>/figure_chain` (`results/figures/figure_chain` by
default). `figure_chain/` keeps the version layout of the manuscript: a figure script of the last version,
`figure_chain/figures_v11.1/work/<figure>/`, imports what it does not change from the earlier ones.

| Step | Command | Reproduces | Time |
|---|---|---|---|
| 28 | `python figure_chain/explore.py`; `python -X utf8 -B figure_chain/figures_v10.4/work/events_wvt.py`; `python -X utf8 -B figure_chain/figures_v10.3/work/fig2a_data_v103.py` | path lengths of the sessions; events of the neurons of Figs 1a, 2a; data of Fig. 2a | 10 min |
| 29 | `python -X utf8 -B figure_chain/figures_v11.1/work/fig1/make_fig1_v111.py` | **Fig. 1**, needs step 5 | 4 min |
| 30 | `python figure_chain/fig2_v10.4/work/ssim_v104.py sync`, `python figure_chain/figures_v10.4/work/fig2_src/fig2_data_v104.py --classic B`, `python figure_chain/fig2_v10.4/work/panel_bd_v104.py`; then, in `figure_chain/figures_v11.1/work/fig2/`, `panel_b_ticks.py`, `panel_f_data.py` and `make_fig2_v111.py final`, each with `python -X utf8 -B` | **Fig. 2**, needs steps 1, 9, 28 | 10 min (estimate); 4 min for the figure |
| 31 | `python -X utf8 -B figure_chain/compute_pack_v11.1/scripts/f_candidates.py`, then `python -X utf8 -B figure_chain/figures_v11.1/work/fig3/make_fig3_v111.py final` | **Fig. 3** (the example neuron of d, e is the third candidate), needs steps 3, 5, 28 | 6 min |
| 32 | `python -X utf8 -B figure_chain/figures_v11.1/work/fig4/make_fig4_v111.py` | **Fig. 4**, needs step 10 | 3 min |
| 33 | `python ed_calibration/ed_calibration_fps.py` | **Extended Data Fig. 1** and its workbook, needs step 8 | seconds |
| 34 | `python -X utf8 -B figure_chain/figures_v11.1/work/glm_examples/make_glm_examples_v111.py` | **Supplementary Fig. 4**, needs steps 3, 17, 28 | seconds |
| 35 | `python -X utf8 -B figure_chain/figures_v11.1/work/mec/make_mec_v111.py final` (Fig. 5 on all analysed sessions, into `figures_v11.1/si/`) or `... final 213` (Fig. 5 on the 213 sessions with labels, into `figures_v11.1/si_213/`) | **Supplementary Figs 5-7**, needs step 27e | seconds |
| 36 | `python -X utf8 -B figure_chain/figures_v11.1/work/figx/fig_x_v111.py` | **Supplementary Fig. 8**, needs step 25 | 1 min |

Supplementary Figs 1-3 are drawn in steps 14 and 16.

**Example panels of an earlier export.** Figs 1a, 1b, 1d and 2b were drawn from an earlier export of the same
sessions, which is not part of the deposit. Its calcium is the deposited calcium with one extra leading row;
its behavior went through another synchronization: positions in 0-1 (correlation with the deposited
coordinates above 0.999), the object-4 label drawn in Figs 1b and 1d different in up to 0.2% of the frames. Head
direction, drawn in Fig. 1b, is in that export the angle from head center to the nose, and in the analysed files the
angle from head center to the miniscope (recomputed from the tracking of two other sessions of the same mouse). So
that these panels come out as printed, their small inputs are in `figure_chain/saved_inputs/` (150 KB): `fig1a_data.npz`,
`fig1b_data.npz`, `NOF_H01_2D_object4_earlier_export.npz`, `fig2b_data_v104_onsets.npz`. The scripts that made
them need the earlier export (`[data] nof_legacy`) and are kept as the record: `fig1/cache_examples.py`,
`fig2/cache_examples.py`, `fig2/fig2_loader.py`, `legacy_loader.py`, `figure_chain/fig2_v10.4/work/fig2b_onsets.py`.
Everything else in these figures is computed from the deposit.

The scheme of Fig. 1c is a raster image without an editable source: `assets/fig1c_scheme.png`, with the two
stage labels set again by `figure_chain/figures_v11.1/work/fig1/make_scheme_v111.py` (needs the Roboto font) into
`assets/fig1c_scheme_v111.png`, which the figure reads.

### Source Data workbooks

    python -X utf8 -B source_data_builders/build_fig1.py     # also build_fig2.py, build_fig3.py, build_fig4.py
    python -X utf8 -B source_data_builders/build_ed1.py
    python -X utf8 -B source_data_builders/check_source_data.py [--tex MANUSCRIPT.tex]

Each builder re-draws its figure (about 5 min) and writes `<[figures] output_dir>/source_data/*_source_data.xlsx`
with the values of the drawn objects; the check reads the input of every panel again and compares it with the
sheet.

## What cannot be rerun from the deposit

* `compute/make_fps_table.py`: the frame rates come from the CaImAn metadata of the BEARMIND processing and the
  raw pickles, not deposited; the resulting table is in this folder (`fps_by_session.csv`).
* The inputs named above under "Example panels of an earlier export": their scripts need that export.
* The behavior analysis (Sphynx) and the processing of the videos (BEARMIND) are other repositories, named in
  the Code availability of the paper.
