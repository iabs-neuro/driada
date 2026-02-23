#!/usr/bin/env python
"""
Generate Notebook 02: Detecting Neuron Selectivity with INTENSE
================================================================

Assembles a Colab-ready Jupyter notebook covering information theory
fundamentals, a basic INTENSE workflow, and a complete pipeline with
feature-feature analysis, disentanglement, and ground truth validation.
"""

import os
import nbformat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md_cell(source):
    """Create a markdown cell."""
    return nbformat.v4.new_markdown_cell(source.strip())


def code_cell(source):
    """Create a code cell."""
    return nbformat.v4.new_code_cell(source.strip())


# ---------------------------------------------------------------------------
# Build cells
# ---------------------------------------------------------------------------

cells = []

# ===== HEADER + SETUP =====================================================

cells.append(md_cell(
"# Detecting neuron selectivity with INTENSE\n"
"\n"
"Which neurons encode which behavioral variables -- and how do you tell\n"
"real selectivity from chance?  [**DRIADA**](https://driada.readthedocs.io)\n"
"answers this with **INTENSE**, an information-theoretic significance\n"
"testing pipeline.  This notebook walks through the method from first\n"
"principles to a full production run.\n"
"\n"
"| Step | Notebook | What it does |\n"
"|---|---|---|\n"
"| **Overview** | [00 -- DRIADA overview](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/00_driada_overview.ipynb) | Core data structures, quick tour of INTENSE, DR, networks |\n"
"| Neuron analysis | [01 -- Neuron analysis](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/01_data_loading_and_neurons.ipynb) | Spike reconstruction, kinetics optimization, quality metrics, surrogates |\n"
"| **Single-neuron selectivity** | **02 -- this notebook** | Detect which neurons encode which behavioral variables |\n"
"| Population geometry | [03 -- Dimensionality reduction](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/03_population_geometry_dr.ipynb) | Extract low-dimensional manifolds from population activity |\n"
"| Network analysis | [04 -- Networks](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/04_network_analysis.ipynb) | Build and analyze cell-cell interaction graphs |\n"
"| Putting it together | [05 -- Advanced](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/05_advanced_capabilities.ipynb) | Combine INTENSE + DR, leave-one-out importance, RSA, RNN analysis |\n"
"\n"
"**What you will learn:**\n"
"\n"
"1. **Information theory fundamentals** -- mutual information estimation\n"
"   (GCMI vs KSG), similarity metrics, time-delayed MI, conditional MI,\n"
"   and interaction information.\n"
"2. **Basic INTENSE workflow** -- generate a synthetic population, run\n"
"   two-stage significance testing, and extract results.\n"
"3. **Complete pipeline with ground truth** -- feature-feature correlations,\n"
"   all feature types, Holm correction, disentanglement, delay optimization,\n"
"   and validation against known selectivity."
))

cells.append(code_cell(
"# TODO: revert to '!pip install -q driada' after v1.0.0 PyPI release\n"
"!pip install -q git+https://github.com/iabs-neuro/driada.git@main\n"
"%matplotlib inline\n"
"\n"
"import os\n"
"import time\n"
"import tempfile\n"
"\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"\n"
"import driada\n"
"from driada.information import (\n"
"    TimeSeries,\n"
"    get_mi,\n"
"    get_sim,\n"
"    get_tdmi,\n"
"    conditional_mi,\n"
"    interaction_information,\n"
")\n"
"from driada.experiment.synthetic import generate_tuned_selectivity_exp\n"
"from driada.intense import compute_feat_feat_significance\n"
"from driada.intense.io import save_results, load_results"
))

# ===== SECTION 1: INFORMATION THEORY FUNDAMENTALS =========================

cells.append(md_cell(
"## 1. Information theory fundamentals\n"
"\n"
"Before running INTENSE, understand the building blocks: mutual information\n"
"estimation, similarity metrics, temporal lags, and conditional dependencies."
))

# --- 1.1 Creating TimeSeries ---

cells.append(md_cell(
"### Creating TimeSeries\n"
"\n"
"Wrap numpy arrays as [`TimeSeries`](https://driada.readthedocs.io/en/latest/api/information/core.html#driada.information.info_base.TimeSeries) with type hints (`linear`, `categorical`,\n"
"`circular`). The type determines which MI estimator DRIADA uses:\n"
"\n"
"| Pair type | GCMI (fast, copula-based) | KSG (accurate, nearest-neighbor) |\n"
"|---|---|---|\n"
"| continuous–continuous | [`mi_gg`](https://driada.readthedocs.io/en/latest/api/information/mutual_information.html#mi-gg) | [`nonparam_mi_cc`](https://driada.readthedocs.io/en/latest/api/information/mutual_information.html#nonparam-mi-cc) |\n"
"| continuous–discrete | [`mi_model_gd`](https://driada.readthedocs.io/en/latest/api/information/mutual_information.html#mi-model-gd) | [`nonparam_mi_cd`](https://driada.readthedocs.io/en/latest/api/information/mutual_information.html#nonparam-mi-cd) |\n"
"| discrete–discrete | exact MI from joint distribution | — |"
))

cells.append(code_cell(
"rng = np.random.default_rng(42)\n"
"n = 5000\n"
"\n"
"print(\"[1] Creating TimeSeries objects\")\n"
"print(\"-\" * 40)\n"
"\n"
"continuous = rng.normal(size=n)\n"
"ts_cont = TimeSeries(continuous, ts_type=\"linear\", name=\"continuous\")\n"
"print(f\"  Continuous: type={ts_cont.type_info}, len={len(ts_cont.data)}\")\n"
"\n"
"discrete = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])\n"
"ts_disc = TimeSeries(discrete, ts_type=\"categorical\", name=\"discrete\")\n"
"print(f\"  Discrete:   type={ts_disc.type_info}, len={len(ts_disc.data)}\")\n"
"\n"
"circular = rng.uniform(0, 2 * np.pi, size=n)\n"
"ts_circ = TimeSeries(circular, ts_type=\"circular\", name=\"circular\")\n"
"print(f\"  Circular:   type={ts_circ.type_info}, len={len(ts_circ.data)}\")"
))

# --- 1.2 Pairwise MI: GCMI vs KSG ---

cells.append(md_cell(
"### Pairwise MI: GCMI vs KSG\n"
"\n"
"[`get_mi`](https://driada.readthedocs.io/en/latest/api/information/core.html#driada.information.info_base.get_mi)`()` estimates mutual information between two `TimeSeries`.\n"
"[**GCMI**](https://driada.readthedocs.io/en/latest/api/information/mutual_information.html#gaussian-copula-mi-gcmi)\n"
"(Gaussian Copula MI) is a fast, copula-based estimator but can underestimate\n"
"non-monotonic dependency.\n"
"[**KSG**](https://driada.readthedocs.io/en/latest/api/information/mutual_information.html#ksg-estimators)\n"
"(Kraskov-Stögbauer-Grassberger) is nearest-neighbor-based and captures\n"
"arbitrary dependency but is slower."
))

cells.append(code_cell(
"print(\"[2] Pairwise mutual information (get_mi)\")\n"
"print(\"-\" * 40)\n"
"\n"
"x = rng.normal(size=n)\n"
"noise = rng.normal(size=n)\n"
"y_corr = x + 0.5 * noise          # correlated with x\n"
"y_indep = rng.normal(size=n)       # independent of x\n"
"\n"
"ts_x = TimeSeries(x)\n"
"ts_y_corr = TimeSeries(y_corr)\n"
"ts_y_indep = TimeSeries(y_indep)\n"
"\n"
"mi_corr = get_mi(ts_x, ts_y_corr)\n"
"mi_indep = get_mi(ts_x, ts_y_indep)\n"
"print(f\"  MI(X, Y_correlated)  = {mi_corr:.4f} bits\")\n"
"print(f\"  MI(X, Y_independent) = {mi_indep:.4f} bits\")\n"
"print(f\"  Correlated MI >> independent MI: {mi_corr > 5 * mi_indep}\")\n"
"\n"
"mi_gcmi = get_mi(ts_x, ts_y_corr, estimator=\"gcmi\")\n"
"mi_ksg = get_mi(ts_x, ts_y_corr, estimator=\"ksg\")\n"
"print(f\"\\n  Monotonic relationship (y = x + noise):\")\n"
"print(f\"    GCMI: {mi_gcmi:.4f} bits\")\n"
"print(f\"    KSG:  {mi_ksg:.4f} bits\")\n"
"print(f\"    (agree because relationship is monotonic)\")\n"
"\n"
"# Spearman rho ~ 0 due to exact symmetry, so GCMI ~ 0.\n"
"x_sym = rng.uniform(-3, 3, size=n)\n"
"y_quad = x_sym ** 2 + 0.3 * rng.normal(size=n)\n"
"ts_x_sym = TimeSeries(x_sym)\n"
"ts_y_quad = TimeSeries(y_quad)\n"
"mi_gcmi_q = get_mi(ts_x_sym, ts_y_quad, estimator=\"gcmi\")\n"
"mi_ksg_q = get_mi(ts_x_sym, ts_y_quad, estimator=\"ksg\")\n"
"print(f\"\\n  Non-monotonic relationship (y = x^2 + noise):\")\n"
"print(f\"    GCMI: {mi_gcmi_q:.4f} bits  (underestimates symmetric dependency)\")\n"
"print(f\"    KSG:  {mi_ksg_q:.4f} bits  (captures it)\")\n"
"print(f\"    KSG >> GCMI: {mi_ksg_q > 3 * mi_gcmi_q}\")"
))

# --- 1.3 Similarity metrics ---

cells.append(md_cell(
"### Similarity metrics\n"
"\n"
"[`get_sim`](https://driada.readthedocs.io/en/latest/api/information/mutual_information.html#driada.information.get_sim)`()` wraps MI, Pearson r, and Spearman rho in a unified interface.\n"
"Available metrics include `mi`, `pearsonr`, `spearmanr`, `kendalltau`,\n"
"`fast_pearsonr`, `av` (activity ratio for binary-gated signals), and any\n"
"scipy.stats correlation function by name."
))

cells.append(code_cell(
"print(\"[3] Similarity metrics comparison (get_sim)\")\n"
"print(\"-\" * 40)\n"
"\n"
"metrics = [\"mi\", \"pearsonr\", \"spearmanr\"]\n"
"for metric in metrics:\n"
"    val = get_sim(ts_x, ts_y_corr, metric=metric)\n"
"    print(f\"  {metric:12s}(X, Y_corr) = {val:.4f}\")"
))

# --- 1.4 Time-delayed MI ---

cells.append(md_cell(
"### Time-delayed MI\n"
"\n"
"[`get_tdmi`](https://driada.readthedocs.io/en/latest/api/information/core.html#driada.information.info_base.get_tdmi)`()` sweeps temporal lags to find the shift that maximizes MI.\n"
"This is useful for detecting delayed neural responses to behavior."
))

cells.append(code_cell(
"# Noisy sine wave — TDMI reveals the underlying periodicity\n"
"period_samples = 30\n"
"t_demo = np.arange(n)\n"
"clean_sine = np.sin(2 * np.pi * t_demo / period_samples)\n"
"signal = clean_sine + 0.4 * rng.normal(size=n)\n"
"\n"
"max_shift = 60  # lag window in samples (= 3 sec at 20 fps)\n"
"tdmi_values = np.array(get_tdmi(signal, max_shift=max_shift))\n"
"lags = np.arange(1, max_shift)\n"
"best_lag = lags[np.argmax(tdmi_values)]\n"
"\n"
"fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4),\n"
"                                gridspec_kw={'height_ratios': [1, 1.3]})\n"
"\n"
"show_n = 4 * period_samples\n"
"ax1.plot(t_demo[:show_n], signal[:show_n], alpha=0.6, label='noisy')\n"
"ax1.plot(t_demo[:show_n], clean_sine[:show_n], 'k--', alpha=0.4, label='clean')\n"
"ax1.set_ylabel('Amplitude')\n"
"ax1.set_title(f'Noisy sine wave (period = {period_samples} samples)')\n"
"ax1.legend(fontsize=8)\n"
"\n"
"ax2.plot(lags, tdmi_values, 'k-')\n"
"ax2.axvline(best_lag, color='red', ls='--',\n"
"            label=f'Peak lag = {best_lag} samples')\n"
"ax2.set_xlabel('Lag (samples)')\n"
"ax2.set_ylabel('MI (bits)')\n"
"ax2.set_title('Time-delayed MI')\n"
"ax2.legend(fontsize=8)\n"
"\n"
"plt.tight_layout()\n"
"plt.show()\n"
"\n"
"print(f'Signal period: {period_samples} samples')\n"
"print(f'TDMI peak:     {best_lag} samples')"
))

# --- 1.5 Conditional MI ---

cells.append(md_cell(
"### Conditional MI and interaction information\n"
"\n"
"**Conditional MI** ([`conditional_mi`](https://driada.readthedocs.io/en/latest/api/information/core.html#driada.information.info_base.conditional_mi)) `I(X;Y|Z)` removes shared variance with Z.\n"
"**Interaction information** ([`interaction_information`](https://driada.readthedocs.io/en/latest/api/information/core.html#driada.information.info_base.interaction_information)) distinguishes synergy (>0) from\n"
"redundancy (<0)."
))

cells.append(code_cell(
"print(\"[5] Conditional MI: I(X;Y|Z)\")\n"
"print(\"-\" * 40)\n"
"\n"
"z = rng.normal(size=n)\n"
"x_from_z = z + 0.3 * rng.normal(size=n)\n"
"y_from_z = z + 0.3 * rng.normal(size=n)\n"
"\n"
"ts_xz = TimeSeries(x_from_z)\n"
"ts_yz = TimeSeries(y_from_z)\n"
"ts_z = TimeSeries(z)\n"
"\n"
"mi_xy = get_mi(ts_xz, ts_yz)\n"
"cmi_xy_z = conditional_mi(ts_xz, ts_yz, ts_z)\n"
"\n"
"print(f\"  I(X;Y)   = {mi_xy:.4f} bits  (shared via Z)\")\n"
"print(f\"  I(X;Y|Z) = {cmi_xy_z:.4f} bits  (residual after conditioning)\")\n"
"print(f\"  Conditioning reduces MI: {cmi_xy_z < mi_xy * 0.5}\")"
))

# --- 1.6 Interaction information ---

cells.append(code_cell(
"print(\"[6] Interaction information: synergy vs redundancy\")\n"
"print(\"-\" * 40)\n"
"\n"
"# Redundancy: Y and Z provide overlapping info about X\n"
"x_r = rng.normal(size=n)\n"
"y_r = TimeSeries(x_r + 0.2 * rng.normal(size=n))\n"
"z_r = TimeSeries(x_r + 0.2 * rng.normal(size=n))\n"
"ts_xr = TimeSeries(x_r)\n"
"\n"
"ii_redund = interaction_information(ts_xr, y_r, z_r)\n"
"print(f\"  Redundancy example: II = {ii_redund:.4f} (expected < 0)\")\n"
"\n"
"# Synergy: XOR-like relationship\n"
"a = rng.choice([0, 1], size=n).astype(float)\n"
"b = rng.choice([0, 1], size=n).astype(float)\n"
"xor_signal = (a + b + 0.1 * rng.normal(size=n))\n"
"\n"
"ts_xor = TimeSeries(xor_signal)\n"
"ts_a = TimeSeries(a, ts_type=\"binary\")\n"
"ts_b = TimeSeries(b, ts_type=\"binary\")\n"
"\n"
"ii_synergy = interaction_information(ts_xor, ts_a, ts_b)\n"
"print(f\"  Synergy example:    II = {ii_synergy:.4f} (expected > 0)\")\n"
"print(f\"  Redundancy is negative: {ii_redund < 0}\")"
))

# ===== SECTION 2: BASIC INTENSE WORKFLOW ==================================

cells.append(md_cell(
"## 2. Basic INTENSE workflow\n"
"\n"
"These information-theoretic tools are the building blocks. DRIADA's INTENSE\n"
"pipeline combines them into a two-stage significance test that scales to\n"
"hundreds of neurons and features.\n"
"\n"
"The minimal pipeline: [`generate_tuned_selectivity_exp`](https://driada.readthedocs.io/en/latest/api/experiment/synthetic.html#driada.experiment.synthetic.generators.generate_tuned_selectivity_exp) creates a synthetic\n"
"population with known selectivity, [`compute_cell_feat_significance`](https://driada.readthedocs.io/en/latest/api/intense/pipelines.html#driada.intense.pipelines.compute_cell_feat_significance) runs\n"
"two-stage significance testing, and [`plot_neuron_feature_pair`](https://driada.readthedocs.io/en/latest/api/intense/visual.html#driada.intense.visual.plot_neuron_feature_pair) visualizes\n"
"individual neuron-feature relationships.\n"
"\n"
"For loading your own experimental data into an `Experiment` object, see\n"
"[Notebook 00](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/00_driada_overview.ipynb).\n"
"The workflow below is identical for real recordings."
))

cells.append(code_cell(
"print(\"Generating synthetic experiment...\")\n"
"\n"
"population = [\n"
"    {\"name\": \"speed_cells\", \"count\": 3, \"features\": [\"speed\"]},\n"
"    {\"name\": \"event_cells\", \"count\": 3, \"features\": [\"event_0\"]},\n"
"    {\"name\": \"nonselective\", \"count\": 4, \"features\": []},\n"
"]\n"
"\n"
"exp = generate_tuned_selectivity_exp(\n"
"    population=population,\n"
"    duration=600,\n"
"    fps=20,\n"
"    seed=47,\n"
"    n_discrete_features=1,\n"
"    verbose=True,\n"
")"
))

cells.append(code_cell(
"print(\"Running INTENSE analysis...\")\n"
"\n"
"stats, significance, info, results = driada.compute_cell_feat_significance(\n"
"    exp,\n"
"    mode=\"two_stage\",\n"
"    n_shuffles_stage1=100,   # fast screening\n"
"    n_shuffles_stage2=10000,  # precise p-values via gamma fit\n"
"    pval_thr=0.001,\n"
"    multicomp_correction=None,\n"
"    ds=5,\n"
"    verbose=True,\n"
")"
))

cells.append(code_cell(
"significant_neurons = exp.get_significant_neurons()\n"
"total_pairs = sum(len(features) for features in significant_neurons.values())\n"
"\n"
"print(f\"Significant neurons: {len(significant_neurons)}/{exp.n_cells}\")\n"
"print(f\"Total neuron-feature pairs: {total_pairs}\")\n"
"\n"
"for cell_id, features in significant_neurons.items():\n"
"    for feat_name in features:\n"
"        pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_name)\n"
"        mi = pair_stats.get('me', 0)\n"
"        pval = pair_stats.get('pval', 1.0)\n"
"        print(f\"  Neuron {cell_id} -> {feat_name}: MI={mi:.4f}, p={pval:.2e}\")"
))

cells.append(code_cell(
"# Visualize an event neuron — shaded regions show when the event is active\n"
"event_neurons = [cid for cid, feats in significant_neurons.items()\n"
"                 if 'event_0' in feats]\n"
"if event_neurons:\n"
"    plot_id = event_neurons[0]\n"
"    fig = driada.intense.plot_neuron_feature_pair(exp, plot_id, 'event_0')\n"
"    plt.show()"
))

# ===== SECTION 3: COMPLETE PIPELINE WITH GROUND TRUTH =====================

cells.append(md_cell(
"## 3. Complete pipeline with ground truth validation\n"
"\n"
"The basic workflow above used a small population and default settings. Next,\n"
"we set up a realistic pipeline with ground truth to evaluate detection\n"
"accuracy.\n"
"\n"
"Production workflow: feature-feature correlation analysis, all feature types\n"
"(circular, spatial, linear, discrete), Holm correction, disentanglement,\n"
"delay optimization, and ground truth validation."
))

cells.append(code_cell(
"# Population configuration - defines neuron groups and their selectivity\n"
"POPULATION = [\n"
"    {\"name\": \"hd_cells\", \"count\": 4, \"features\": [\"head_direction\"]},\n"
"    {\"name\": \"place_cells\", \"count\": 4, \"features\": [\"position_2d\"]},\n"
"    {\"name\": \"speed_cells\", \"count\": 4, \"features\": [\"speed\"]},\n"
"    {\"name\": \"event_cells\", \"count\": 4, \"features\": [\"event_0\"]},\n"
"    {\"name\": \"mixed_cells\", \"count\": 4, \"features\": [\"head_direction\", \"event_0\"]},\n"
"    {\"name\": \"nonselective\", \"count\": 4, \"features\": []},\n"
"]\n"
"\n"
"# Analysis parameters\n"
"CONFIG = {\n"
"    # Recording parameters\n"
"    \"duration\": 900,        # seconds\n"
"    \"fps\": 20,              # sampling rate\n"
"    \"seed\": 42,\n"
"    # Tuning parameters\n"
"    \"kappa\": 4.0,           # von Mises concentration (HD cells)\n"
"    # Calcium dynamics\n"
"    \"baseline_rate\": 0.02,  # baseline firing rate\n"
"    \"peak_rate\": 2.0,       # peak response\n"
"    \"decay_time\": 1.5,      # calcium decay time\n"
"    \"calcium_noise\": 0.01,  # noise level\n"
"    # Discrete event parameters\n"
"    \"n_discrete_features\": 2,\n"
"    \"event_active_fraction\": 0.08,  # ~8% active time per event\n"
"    \"event_avg_duration\": 0.8,      # seconds\n"
"    # INTENSE analysis parameters\n"
"    \"n_shuffles_stage1\": 100,   # stage 1 screening shuffles\n"
"    \"n_shuffles_stage2\": 10000,  # stage 2 confirmation (FFT makes this fast)\n"
"    \"pval_thr\": 0.05,           # p-value threshold after correction\n"
"    \"multicomp_correction\": \"holm\",  # multiple comparison correction\n"
"}\n"
"\n"
"# Custom tuning defaults based on config\n"
"tuning_defaults = {\n"
"    \"head_direction\": {\"kappa\": CONFIG[\"kappa\"]},\n"
"}\n"
"\n"
"exp3 = generate_tuned_selectivity_exp(\n"
"    population=POPULATION,\n"
"    tuning_defaults=tuning_defaults,\n"
"    duration=CONFIG[\"duration\"],\n"
"    fps=CONFIG[\"fps\"],\n"
"    baseline_rate=CONFIG[\"baseline_rate\"],\n"
"    peak_rate=CONFIG[\"peak_rate\"],\n"
"    decay_time=CONFIG[\"decay_time\"],\n"
"    calcium_noise=CONFIG[\"calcium_noise\"],\n"
"    n_discrete_features=CONFIG[\"n_discrete_features\"],\n"
"    event_active_fraction=CONFIG[\"event_active_fraction\"],\n"
"    event_avg_duration=CONFIG[\"event_avg_duration\"],\n"
"    seed=CONFIG[\"seed\"],\n"
"    verbose=True,\n"
")\n"
"ground_truth = exp3.ground_truth\n"
"\n"
"# Add a derived feature: speed with measurement noise (correlated by construction)\n"
"speed_data = exp3.dynamic_features['speed'].data\n"
"rng = np.random.RandomState(CONFIG['seed'])\n"
"speed_noisy = speed_data + rng.normal(0, 0.3 * np.std(speed_data), len(speed_data))\n"
"exp3.add_feature('speed_noisy', speed_noisy, ts_type='linear')\n"
"from scipy.stats import pearsonr\n"
"r, _ = pearsonr(speed_data, speed_noisy)\n"
"print(f'  Added speed_noisy (speed + 30% Gaussian noise, r={r:.3f})')"
))

# --- 3.1 Feature-feature analysis (before neuron analysis) ---

cells.append(md_cell(
"### Feature-feature correlations\n"
"\n"
"Before analyzing neurons, check which behavioral variables are themselves\n"
"correlated. [`compute_feat_feat_significance`](https://driada.readthedocs.io/en/latest/api/intense/pipelines.html#driada.intense.pipelines.compute_feat_feat_significance)\n"
"tests all feature pairs with FFT-based circular shuffles. For example,\n"
"the noisy speed is derived from speed by adding measurement noise, so they\n"
"are correlated by construction. Any significant correlations here will inform\n"
"the disentanglement step later."
))

cells.append(code_cell(
"feat_bunch = [\n"
"    feat_name for feat_name in exp3.dynamic_features.keys()\n"
"    if feat_name not in ['x', 'y', 'head_direction']\n"
"]\n"
"print(f'Features to test: {feat_bunch}')\n"
"\n"
"sim_mat, sig_mat, pval_mat, feature_names_ff, info_ff = compute_feat_feat_significance(\n"
"    exp3,\n"
"    feat_bunch=feat_bunch,\n"
"    n_shuffles_stage1=100,\n"
"    n_shuffles_stage2=1000,\n"
"    pval_thr=0.01,\n"
"    verbose=True,\n"
")"
))

cells.append(code_cell(
"display_names = []\n"
"for name in feature_names_ff:\n"
"    if isinstance(name, (list, tuple)):\n"
"        display_names.append(', '.join(str(n) for n in name))\n"
"    else:\n"
"        display_names.append(str(name))\n"
"\n"
"n_ff = len(feature_names_ff)\n"
"\n"
"print('Significant pairs:')\n"
"n_sig = 0\n"
"for i in range(n_ff):\n"
"    for j in range(i + 1, n_ff):\n"
"        if sig_mat[i, j]:\n"
"            n_sig += 1\n"
"            print(\n"
"                f'  {display_names[i]:20s} <-> {display_names[j]:20s}  '\n"
"                f'MI={sim_mat[i, j]:.4f}  p={pval_mat[i, j]:.2e}'\n"
"            )\n"
"if n_sig == 0:\n"
"    print('  (none)')\n"
"\n"
"fig, ax = plt.subplots(figsize=(8, 7))\n"
"plot_mat = sim_mat.copy().astype(float)\n"
"np.fill_diagonal(plot_mat, np.nan)\n"
"\n"
"im = ax.imshow(plot_mat, cmap='Blues', aspect='equal')\n"
"cbar = plt.colorbar(im, ax=ax, shrink=0.8)\n"
"cbar.set_label('Mutual information (bits)')\n"
"\n"
"for i in range(n_ff):\n"
"    for j in range(n_ff):\n"
"        if i != j and sig_mat[i, j]:\n"
"            ax.text(j, i, '*', ha='center', va='center',\n"
"                    fontsize=14, fontweight='bold', color='red')\n"
"\n"
"for i in range(n_ff):\n"
"    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,\n"
"                               fill=True, facecolor='0.85', edgecolor='none'))\n"
"\n"
"ax.set_xticks(range(n_ff))\n"
"ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)\n"
"ax.set_yticks(range(n_ff))\n"
"ax.set_yticklabels(display_names, fontsize=8)\n"
"ax.set_title('Feature-feature MI (* = significant)')\n"
"\n"
"plt.tight_layout()\n"
"plt.show()"
))

# --- 3.2 Running INTENSE ---

cells.append(md_cell(
"### Running INTENSE with disentanglement\n"
"\n"
"Some features are correlated (e.g. speed and noisy speed above). Before\n"
"interpreting neuron selectivity, we need to account for these redundancies.\n"
"Disentanglement uses conditional MI to separate genuine from inherited\n"
"selectivity.\n"
"\n"
"With the feature correlation structure in mind, run the full pipeline\n"
"with `with_disentanglement=True` to resolve redundant multi-feature\n"
"detections."
))

cells.append(code_cell(
"print('Running INTENSE analysis...')\n"
"start_time = time.time()\n"
"\n"
"stats3, significance3, info3, results3, disent_results3 = driada.compute_cell_feat_significance(\n"
"    exp3,\n"
"    feat_bunch=feat_bunch,\n"
"    mode='two_stage',\n"
"    n_shuffles_stage1=CONFIG['n_shuffles_stage1'],\n"
"    n_shuffles_stage2=CONFIG['n_shuffles_stage2'],\n"
"    find_optimal_delays=True,\n"
"    ds=5,\n"
"    pval_thr=CONFIG['pval_thr'],\n"
"    multicomp_correction=CONFIG['multicomp_correction'],\n"
"    with_disentanglement=True,\n"
"    verbose=True,\n"
")\n"
"\n"
"analysis_time = time.time() - start_time\n"
"significant_neurons3 = exp3.get_significant_neurons()\n"
"\n"
"total_pairs = sum(len(features) for features in significant_neurons3.values())\n"
"print(f\"\\n  Completed in {analysis_time:.1f} seconds\")\n"
"print(f\"  Significant neurons: {len(significant_neurons3)}/{exp3.n_cells}\")\n"
"print(f\"  Total significant pairs: {total_pairs}\")"
))

# --- 3.3 Ground truth validation ---

cells.append(md_cell(
"### Ground truth validation\n"
"\n"
"Compare detections to known selectivity. `validate_against_ground_truth`\n"
"computes sensitivity, precision, and F1 per neuron type."
))

cells.append(code_cell(
"metrics = results3.validate_against_ground_truth(ground_truth, verbose=True)"
))

# --- 3.4 Disentanglement ---

cells.append(md_cell(
"### Disentanglement: before vs after\n"
"\n"
"The feature-feature analysis above showed which features correlate.\n"
"Disentanglement leverages that: for neurons detected with multiple features,\n"
"it tests `I(neuron; F1 | F2) > 0` to decide whether each detection is\n"
"truly independent or redundant."
))

cells.append(code_cell(
"if disent_results3 is not None:\n"
"    summary = disent_results3.get(\"summary\", {})\n"
"    per_neuron_disent = disent_results3.get(\"per_neuron_disent\", {})\n"
"\n"
"    if \"overall_stats\" in summary:\n"
"        stats_d = summary[\"overall_stats\"]\n"
"        print(f\"\\n  Neuron-feature pairs analyzed: {stats_d.get('total_neuron_pairs', 0)}\")\n"
"        print(f\"  Redundancy rate: {stats_d.get('redundancy_rate', 0):.1f}%\")\n"
"        print(f\"  True mixed selectivity rate: {stats_d.get('true_mixed_selectivity_rate', 0):.1f}%\")\n"
"\n"
"    # Build corrected significant_neurons using final_sels\n"
"    corrected = {}\n"
"    n_removed = 0\n"
"    for neuron_id, features in significant_neurons3.items():\n"
"        if neuron_id in per_neuron_disent:\n"
"            final = per_neuron_disent[neuron_id].get(\"final_sels\", features)\n"
"            n_removed += len(features) - len(final)\n"
"            if final:\n"
"                corrected[neuron_id] = final\n"
"        else:\n"
"            corrected[neuron_id] = features\n"
"\n"
"    # Compute corrected metrics against ground truth\n"
"    expected_pairs = set(ground_truth[\"expected_pairs\"])\n"
"    tp, fp, fn = 0, 0, 0\n"
"    for neuron_id, features in corrected.items():\n"
"        for feat_name in features:\n"
"            if (neuron_id, feat_name) in expected_pairs:\n"
"                tp += 1\n"
"            else:\n"
"                fp += 1\n"
"    fn = len(expected_pairs) - tp\n"
"\n"
"    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0\n"
"    precision = tp / (tp + fp) if (tp + fp) > 0 else 0\n"
"    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0\n"
"\n"
"    # Show before/after\n"
"    print(f\"\\n  Pairs removed by disentanglement: {n_removed}\")\n"
"    print(f\"\\n  {'Metric':<15} {'Before':<12} {'After'}\")\n"
"    print(f\"  {'-'*40}\")\n"
"    print(f\"  {'Sensitivity':<15} {metrics['sensitivity']:>10.1%}   {sensitivity:>10.1%}\")\n"
"    print(f\"  {'Precision':<15} {metrics['precision']:>10.1%}   {precision:>10.1%}\")\n"
"    print(f\"  {'F1 Score':<15} {metrics['f1']:>10.1%}   {f1:>10.1%}\")\n"
"    print(f\"  {'False Pos':<15} {metrics['false_positives']:>10}   {fp:>10}\")\n"
"else:\n"
"    print(\"  Disentanglement not performed.\")"
))

# --- 3.5 Optimal delays ---

cells.append(md_cell(
"### Optimal delays\n"
"\n"
"Neurons often respond to stimuli with a lag. Optimal delay estimation shifts\n"
"each feature to maximize MI, capturing delayed responses that fixed-lag\n"
"analysis would miss.\n"
"\n"
"Temporal offset maximizing MI between neural activity and behavior.\n"
"Positive delays mean neural activity lags behind behavior (expected\n"
"for calcium imaging due to indicator dynamics)."
))

cells.append(code_cell(
"optimal_delays = info3.get('optimal_delays')\n"
"if optimal_delays is not None:\n"
"    fps = CONFIG['fps']\n"
"    neuron_types = ground_truth.get('neuron_types', {})\n"
"    type_delays = {}\n"
"\n"
"    for neuron_id, features in significant_neurons3.items():\n"
"        ntype = neuron_types.get(neuron_id, 'unknown')\n"
"        for feat_name in features:\n"
"            if feat_name in feat_bunch:\n"
"                fi = feat_bunch.index(feat_name)\n"
"                d = optimal_delays[neuron_id, fi]\n"
"                type_delays.setdefault(ntype, []).append(d / fps)\n"
"\n"
"    # Mean delay per neuron type\n"
"    print('Mean optimal delay per neuron type:')\n"
"    for ntype in sorted(type_delays):\n"
"        vals = type_delays[ntype]\n"
"        print(f'  {ntype:20s}  {np.mean(vals):+.2f}s  (n={len(vals)})')\n"
"else:\n"
"    print('No delay optimization performed.')"
))

cells.append(code_cell(
"fig = plt.figure(figsize=(14, 10))\n"
"\n"
"# 1. Selectivity heatmap (main plot)\n"
"ax1 = fig.add_subplot(2, 2, (1, 2))\n"
"\n"
"feature_names = feat_bunch\n"
"n_neurons = exp3.n_cells\n"
"n_features = len(feature_names)\n"
"\n"
"# Create MI matrix using 'me'\n"
"mi_matrix = np.zeros((n_neurons, n_features))\n"
"for neuron_id, features in significant_neurons3.items():\n"
"    for feat_name in features:\n"
"        if feat_name in feature_names:\n"
"            feat_idx = feature_names.index(feat_name)\n"
"            pair_stats = exp3.get_neuron_feature_pair_stats(neuron_id, feat_name)\n"
"            mi_matrix[neuron_id, feat_idx] = pair_stats.get(\"me\", 0)\n"
"\n"
"im = ax1.imshow(mi_matrix, aspect=\"auto\", cmap=\"viridis\")\n"
"ax1.set_xlabel(\"Features\")\n"
"ax1.set_ylabel(\"Neurons\")\n"
"ax1.set_title(\"INTENSE selectivity heatmap (MI values)\")\n"
"ax1.set_xticks(range(n_features))\n"
"ax1.set_xticklabels(feature_names, rotation=45, ha=\"right\")\n"
"\n"
"# Add colorbar\n"
"cbar = plt.colorbar(im, ax=ax1, shrink=0.8)\n"
"cbar.set_label(\"Mutual information (bits)\")\n"
"\n"
"# Add neuron type annotations\n"
"type_colors = {\n"
"    \"hd_cells\": \"red\",\n"
"    \"place_cells\": \"blue\",\n"
"    \"speed_cells\": \"green\",\n"
"    \"event_cells\": \"orange\",\n"
"    \"mixed_cells\": \"purple\",\n"
"    \"nonselective\": \"gray\",\n"
"}\n"
"for neuron_id, neuron_type in ground_truth[\"neuron_types\"].items():\n"
"    color = type_colors.get(neuron_type, \"gray\")\n"
"    ax1.scatter(-0.7, neuron_id, c=color, s=20, marker=\"s\")\n"
"\n"
"# 2. Detection rates by type\n"
"ax2 = fig.add_subplot(2, 2, 3)\n"
"types = list(metrics[\"type_stats\"].keys())\n"
"sensitivities = [metrics[\"type_stats\"][t][\"sensitivity\"] * 100 for t in types]\n"
"colors = [type_colors.get(t, \"gray\") for t in types]\n"
"\n"
"bars = ax2.bar(range(len(types)), sensitivities, color=colors)\n"
"ax2.set_xticks(range(len(types)))\n"
"ax2.set_xticklabels([t.replace(\"_\", \"\\n\") for t in types], fontsize=8)\n"
"ax2.set_ylabel(\"Detection rate (%)\")\n"
"ax2.set_title(\"Detection rate by neuron type\")\n"
"ax2.set_ylim(0, 105)\n"
"ax2.axhline(y=100, color=\"k\", linestyle=\"--\", alpha=0.3)\n"
"\n"
"# Add percentage labels\n"
"for bar, pct in zip(bars, sensitivities):\n"
"    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,\n"
"            f\"{pct:.0f}%\", ha=\"center\", va=\"bottom\", fontsize=9)\n"
"\n"
"# 3. Summary statistics (before and after disentanglement)\n"
"ax3 = fig.add_subplot(2, 2, 4)\n"
"ax3.axis(\"off\")\n"
"\n"
"summary_text = (\n"
"    f\"VALIDATION SUMMARY\\n\"\n"
"    f\"{'=' * 30}\\n\\n\"\n"
"    f\"{'Metric':<12} {'Raw':>8}\\n\"\n"
"    f\"{'-' * 30}\\n\"\n"
"    f\"{'Sensitivity':<12} {metrics['sensitivity']:>7.1%}\\n\"\n"
"    f\"{'Precision':<12} {metrics['precision']:>7.1%}\\n\"\n"
"    f\"{'F1 Score':<12} {metrics['f1']:>7.1%}\\n\\n\"\n"
"    f\"Detection counts:\\n\"\n"
"    f\"  True Positives:  {metrics['true_positives']}\\n\"\n"
"    f\"  False Positives: {metrics['false_positives']}\\n\"\n"
"    f\"  False Negatives: {metrics['false_negatives']}\\n\\n\"\n"
"    f\"Population:\\n\"\n"
"    f\"  Neurons: {exp3.n_cells}, Features: {len(exp3.dynamic_features)}\\n\"\n"
"    f\"  Expected pairs: {len(ground_truth['expected_pairs'])}\\n\"\n"
")\n"
"ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,\n"
"        fontfamily=\"monospace\", fontsize=9, verticalalignment=\"top\")\n"
"\n"
"plt.tight_layout()\n"
"plt.show()"
))

# --- 3.7 Save and load results ---

cells.append(md_cell(
"Once analysis is complete, save the results for later visualization or\n"
"cross-session comparison."
))

cells.append(md_cell(
"### Save and load results\n"
"\n"
"Persist INTENSE results to disk with [`save_results`](https://driada.readthedocs.io/en/latest/api/intense/base.html#driada.intense.io.save_results) and reload them\n"
"with [`load_results`](https://driada.readthedocs.io/en/latest/api/intense/base.html#driada.intense.io.load_results) for later analysis."
))

cells.append(code_cell(
"# Save/load round-trip\n"
"with tempfile.TemporaryDirectory() as tmpdir:\n"
"    results_path = os.path.join(tmpdir, \"intense_results.npz\")\n"
"    save_results(results3, results_path)\n"
"    file_mb = os.path.getsize(results_path) / 1024 / 1024\n"
"    print(f\"  Saved results: {results_path} ({file_mb:.1f} MB)\")\n"
"\n"
"    loaded = load_results(results_path)\n"
"    print(f\"  Reloaded: {len(loaded.stats)} neurons\")\n"
"    print(f\"  Stats keys match: {set(str(k) for k in results3.stats.keys()) == set(loaded.stats.keys())}\")"
))

cells.append(md_cell(
"## Further reading\n"
"\n"
"Standalone examples (run directly, no external data needed):\n"
"- [full_intense_pipeline](https://github.com/iabs-neuro/driada/tree/main/examples/full_intense_pipeline) -- Complete pipeline across all feature types\n"
"- [mixed_selectivity](https://github.com/iabs-neuro/driada/tree/main/examples/mixed_selectivity) -- Disentanglement and interaction information\n"
"- [signal_association](https://github.com/iabs-neuro/driada/tree/main/examples/signal_association) -- MI estimators, TDMI, conditional MI\n"
"- [behavior_relations](https://github.com/iabs-neuro/driada/tree/main/examples/behavior_relations) -- Feature-feature significance testing\n"
"\n"
"[All examples](https://github.com/iabs-neuro/driada/tree/main/examples)"
))

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------

nb = nbformat.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0",
    },
})
nb.cells = cells

output_dir = os.path.join(os.path.dirname(__file__), "..", "notebooks")
output_path = os.path.join(output_dir, "02_selectivity_detection_intense.ipynb")
os.makedirs(output_dir, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to {os.path.abspath(output_path)}")
print(f"  {len(cells)} cells total")
n_md = sum(1 for c in cells if c.cell_type == "markdown")
n_code = sum(1 for c in cells if c.cell_type == "code")
print(f"  {n_md} markdown cells, {n_code} code cells")
