#!/usr/bin/env python
"""
Generate Notebook 00: DRIADA Overview
======================================

Assembles a Colab-ready overview notebook with two parts:
  1. Data structures (Experiment, TimeSeries, feature types) -- adapted from NB01 Section 1
  2. Mini demos of INTENSE, dimensionality reduction, and networks using a synthetic population
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
"# DRIADA overview\n"
"\n"
"[**DRIADA**](https://driada.readthedocs.io) (Dimensionality Reduction for\n"
"Integrated Activity Data) is a Python framework for neural data analysis.\n"
"It bridges two perspectives that are usually treated separately: what\n"
"*individual* neurons encode, and how the *population as a whole* represents\n"
"information.  The typical analysis workflow looks like this:\n"
"\n"
"| Step | Notebook | What it does |\n"
"|---|---|---|\n"
"| **Overview** | **00 -- this notebook** | Core data structures, quick tour of INTENSE, dimensionality reduction, networks |\n"
"| Neuron analysis | [01 -- Neuron analysis](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/01_data_loading_and_neurons.ipynb) | Spike reconstruction, kinetics optimization, quality metrics, surrogates |\n"
"| Single-neuron selectivity | [02 -- INTENSE](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/02_selectivity_detection_intense.ipynb) | Detect which neurons encode which behavioral variables |\n"
"| Population geometry | [03 -- Dimensionality reduction](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/03_population_geometry_dr.ipynb) | Extract low-dimensional manifolds from population activity |\n"
"| Network analysis | [04 -- Networks](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/04_network_analysis.ipynb) | Build and analyze cell-cell interaction graphs |\n"
"| Putting it together | [05 -- Advanced](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/05_advanced_capabilities.ipynb) | Combine INTENSE + dimensionality reduction, leave-one-out importance, RSA, RNN analysis |\n"
"\n"
"**What you will learn:**\n"
"\n"
"1. **Loading data into an Experiment** -- wrap numpy arrays into a DRIADA [`Experiment`](https://driada.readthedocs.io/en/latest/api/experiment/core.html#driada.experiment.exp_base.Experiment).\n"
"2. **Feature types and TimeSeries** -- understand how DRIADA represents and auto-detects behavioral variables.\n"
"3. **Quick tour: selectivity, dimensionality reduction, networks** -- run INTENSE, project onto Isomap, and build a functional connectivity graph."
))

cells.append(code_cell(
"# TODO: revert to '!pip install -q driada' after v1.0.0 PyPI release\n"
"!pip install -q git+https://github.com/iabs-neuro/driada.git@main\n"
"%matplotlib inline\n"
"\n"
"import os\n"
"import tempfile\n"
"\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"\n"
"from driada.experiment import (\n"
"    load_exp_from_aligned_data,\n"
"    save_exp_to_pickle,\n"
"    load_exp_from_pickle,\n"
")\n"
"from driada.experiment.synthetic import (\n"
"    generate_pseudo_calcium_multisignal,\n"
"    generate_tuned_selectivity_exp,\n"
")"
))

# ===== PART 1: DATA STRUCTURES (from NB01 Section 1) =======================

cells.append(md_cell(
"## 1. Loading your data into DRIADA\n"
"\n"
"You have numpy arrays from your recording pipeline (Suite2P, CaImAn,\n"
"DeepLabCut, etc.).  [`load_exp_from_aligned_data`](https://driada.readthedocs.io/en/latest/api/experiment/loading.html#driada.experiment.exp_build.load_exp_from_aligned_data)\n"
"wraps them into an **Experiment** object that\n"
"keeps neural activity and behavioral features aligned and annotated.\n"
"\n"
"The data dict must contain one neural-data key -- any of `'calcium'`,\n"
"`'activations'`, `'neural_data'`, `'activity'`, or `'rates'` -- holding a\n"
"`(n_neurons, n_frames)` array.  Everything else you pass becomes a\n"
"**dynamic feature** (one value per timepoint).\n"
"\n"
"Below we use DRIADA's\n"
"[`generate_pseudo_calcium_multisignal`](https://driada.readthedocs.io/en/latest/api/experiment/synthetic.html#driada.experiment.synthetic.core.generate_pseudo_calcium_multisignal)\n"
"to create realistic synthetic calcium traces with GCaMP-like dynamics\n"
"(transient events, exponential decay, baseline noise).  In your own work,\n"
"replace this with your actual recording data."
))

cells.append(code_cell(
"# In practice: calcium = np.load('your_recording.npz')['calcium']\n"
"\n"
"n_neurons = 20  # small for demo; real datasets: 100-1000+\n"
"fps = 30.0  # frames per second of your recording\n"
"duration = 200.0  # seconds\n"
"\n"
"calcium = generate_pseudo_calcium_multisignal(\n"
"    n=n_neurons,\n"
"    duration=duration,\n"
"    sampling_rate=fps,\n"
"    event_rate=0.15,  # events per second; typical calcium: 0.05-0.5\n"
"    amplitude_range=(0.5, 2.0),\n"
"    decay_time=1.5,\n"
"    rise_time=0.15,\n"
"    noise_std=0.05,\n"
"    kernel='double_exponential',\n"
"    seed=0,\n"
")\n"
"n_frames = calcium.shape[1]\n"
"\n"
"# Behavioral variables (one value per timepoint)\n"
"np.random.seed(0)\n"
"x_pos = np.cumsum(np.random.randn(n_frames) * 0.5)            # continuous\n"
"y_pos = np.cumsum(np.random.randn(n_frames) * 0.5)            # continuous\n"
"speed = np.abs(np.random.randn(n_frames)) * 5.0               # continuous\n"
"head_direction = np.random.uniform(0, 2 * np.pi, n_frames)    # circular (radians)\n"
"trial_type = np.random.choice([0, 1, 2], size=n_frames)       # discrete labels\n"
"\n"
'print(f"calcium:        shape={calcium.shape}, dtype={calcium.dtype}")\n'
'print(f"x_pos:          shape={x_pos.shape}, dtype={x_pos.dtype}")\n'
'print(f"y_pos:          shape={y_pos.shape}, dtype={y_pos.dtype}")\n'
'print(f"speed:          shape={speed.shape}, dtype={speed.dtype}")\n'
'print(f"head_direction: shape={head_direction.shape}, dtype={head_direction.dtype}")\n'
'print(f"trial_type:     shape={trial_type.shape}, dtype={trial_type.dtype}")'
))

cells.append(code_cell(
"# Quick look at the calcium traces\n"
"fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)\n"
"\n"
"time_sec = np.arange(n_frames) / fps\n"
"n_show = min(5, n_neurons)\n"
"\n"
"# Top: overlaid traces (first few neurons)\n"
"ax = axes[0]\n"
"for i in range(n_show):\n"
"    ax.plot(time_sec, calcium[i], linewidth=0.8, label=f'neuron {i}')\n"
"ax.set_ylabel('dF/F0')\n"
"ax.set_title(f'Synthetic calcium traces ({n_neurons} neurons, {duration:.0f}s @ {fps:.0f} Hz)')\n"
"ax.legend(loc='upper right', fontsize=8)\n"
"ax.grid(True, alpha=0.3)\n"
"\n"
"# Bottom: offset traces for clearer event structure\n"
"ax = axes[1]\n"
"offsets = np.arange(n_show) * 3\n"
"for i in range(n_show):\n"
"    ax.plot(time_sec, calcium[i] + offsets[i], 'k', linewidth=0.6)\n"
"ax.set_xlabel('Time (s)')\n"
"ax.set_ylabel('dF/F0 + offset')\n"
"ax.set_title('Offset view (same neurons)')\n"
"ax.set_yticks(offsets)\n"
"ax.set_yticklabels([f'n{i}' for i in range(n_show)])\n"
"ax.grid(True, alpha=0.3)\n"
"\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(md_cell(
"### Feature types and aggregation\n"
"\n"
"DRIADA runs a multi-stage\n"
"[auto-detection pipeline](https://driada.readthedocs.io/en/latest/api/information/utilities.html#module-driada.information.time_series_types)\n"
"on every feature to determine its type.  The pipeline considers uniqueness\n"
"ratio, integer fraction, gap statistics, distribution tests, and -- for\n"
"circular candidates -- variable name, value range, wraparound jumps, and\n"
"Von Mises fit.  The result is a `primary_type` (continuous / discrete /\n"
"ambiguous) plus a `subtype`:\n"
"\n"
"| Primary type | Subtypes |\n"
"|---|---|\n"
"| continuous | `linear`, `circular` |\n"
"| discrete | `binary`, `categorical`, `count`, `timeline` |\n"
"\n"
"You can override the detection with a\n"
"[`feature_types`](https://driada.readthedocs.io/en/latest/api/experiment/loading.html#driada.experiment.exp_build.load_exp_from_aligned_data)\n"
"dict mapping feature names to type strings.  Valid strings: `continuous`,\n"
"`linear`, `circular`, `phase`, `angle`, `discrete`, `binary`, `categorical`,\n"
"`count`, `timeline`.  When `feature_types` is provided, any auto-detected\n"
"circular feature **not** listed in it is overridden to `linear` (whitelist\n"
"behaviour).\n"
"\n"
"`aggregate_features` groups related 1D features into a single\n"
"[`MultiTimeSeries`](https://driada.readthedocs.io/en/latest/api/information/core.html#driada.information.info_base.MultiTimeSeries)\n"
"(e.g. x_pos + y_pos -> position_2d).\n"
"\n"
"`create_circular_2d=True` (the default) auto-creates a `(cos, sin)` encoding\n"
"for every circular feature.  This is important because MI estimators (GCMI,\n"
"KSG) assume the real line -- a raw angle wraps at 0 / 2*pi, breaking the\n"
"distance metric.  The `(cos, sin)` encoding maps the circle onto R^2 where\n"
"Euclidean distance is meaningful."
))

cells.append(code_cell(
'# Build the data dict\n'
'data = {\n'
'    # --- neural activity (required) --------------------------------\n'
'    "calcium": calcium,               # (n_neurons, n_frames)\n'
'    # "spikes": my_spikes_array,      # optional, same shape as calcium\n'
'    # --- dynamic features: behavioral variables (one per timepoint) -\n'
'    "x_pos": x_pos,                   # continuous\n'
'    "y_pos": y_pos,                   # continuous\n'
'    "speed": speed,                   # continuous\n'
'    "head_direction": head_direction,  # circular (radians)\n'
'    "trial_type": trial_type,         # discrete labels\n'
'}\n'
'\n'
'# Override auto-detected feature types (optional)\n'
'feature_types = {\n'
'    "head_direction": "circular",   # auto-detection may miss this\n'
'    "trial_type": "categorical",    # refine from generic discrete\n'
'}\n'
'\n'
'# Aggregate multi-component features (optional)\n'
'aggregate_features = {\n'
'    ("x_pos", "y_pos"): "position_2d",\n'
'}\n'
'\n'
'# Build the Experiment\n'
'exp = load_exp_from_aligned_data(\n'
'    data_source="MyLab",\n'
'    exp_params={"name": "demo_recording"},\n'
'    data=data,\n'
'    feature_types=feature_types,\n'
'    aggregate_features=aggregate_features,\n'
'    static_features={"fps": 30.0},\n'
'    create_circular_2d=True,  # auto-create (cos, sin) for circular features\n'
'    verbose=True,\n'
')'
))

cells.append(md_cell(
"### Inspecting the Experiment\n"
"\n"
"Note the auto-generated features in the list below:\n"
"- **position_2d** -- from `aggregate_features` (x_pos + y_pos)\n"
"- **head_direction_2d** -- from `create_circular_2d` (cos + sin encoding)"
))

cells.append(code_cell(
'print(f"Neurons:     {exp.n_cells}")\n'
'print(f"Timepoints:  {exp.n_frames}")\n'
'print(f"FPS:         {exp.static_features.get(\'fps\', \'unknown\')}")\n'
'print(f"Calcium:     {exp.calcium.data.shape}")\n'
'\n'
'print("\\nDynamic features (time-varying behavioral variables):")\n'
'for name, ts in sorted(exp.dynamic_features.items()):\n'
'    ti = getattr(ts, "type_info", None)\n'
'    if ti and hasattr(ti, "primary_type"):\n'
'        dtype_str = f"{ti.primary_type}/{ti.subtype}"\n'
'        if ti.is_circular:\n'
'            dtype_str += " (circular)"\n'
'    else:\n'
'        dtype_str = "discrete" if ts.discrete else "continuous"\n'
'    shape = ts.data.shape\n'
'    print(f"  {name:25s}  shape={str(shape):15s}  type={dtype_str}")'
))

cells.append(md_cell(
"### TimeSeries and MultiTimeSeries\n"
"\n"
"Each dynamic feature is stored as one of two classes:\n"
"\n"
"| Class | Description |\n"
"|---|---|\n"
"| [**`TimeSeries`**](https://driada.readthedocs.io/en/latest/api/information/core.html#driada.information.info_base.TimeSeries) | A single 1D variable (e.g. `speed`) |\n"
"| [**`MultiTimeSeries`**](https://driada.readthedocs.io/en/latest/api/information/core.html#driada.information.info_base.MultiTimeSeries) | Multiple aligned 1D variables stacked into a 2D array (e.g. `position_2d = [x, y]`) |\n"
"\n"
"Key attributes on both:\n"
"- `.data` -- raw numpy array (1D or 2D)\n"
"- `.discrete` -- True if discrete, False if continuous\n"
"- `.type_info` -- rich type metadata (subtype, circularity)\n"
"- `.copula_normal_data` -- GCMI-ready transform (continuous only)\n"
"- `.int_data` -- integer-coded values (discrete only)\n"
"\n"
"MultiTimeSeries additionally has `.ts_list` (list of component TimeSeries)\n"
"and `.n_dim` (number of components)."
))

cells.append(code_cell(
'# Features are accessible as attributes: exp.speed, exp.position_2d, etc.\n'
'# This is equivalent to exp.dynamic_features["speed"].\n'
'speed_ts = exp.speed\n'
'print(f"speed.data.shape:   {speed_ts.data.shape}")\n'
'print(f"speed.discrete:     {speed_ts.discrete}")\n'
'print(f"speed.type_info:    {speed_ts.type_info.primary_type}"\n'
'      f"/{speed_ts.type_info.subtype}")\n'
'print(f"speed has copula:   {speed_ts.copula_normal_data is not None}")\n'
'\n'
'# Access a 2D feature (MultiTimeSeries)\n'
'pos_mts = exp.position_2d\n'
'print(f"\\nposition_2d.data.shape: {pos_mts.data.shape}")\n'
'print(f"position_2d.n_dim:      {pos_mts.n_dim}  (x and y)")\n'
'# Individual components are full TimeSeries objects:\n'
'print(f"position_2d.ts_list[0]: {pos_mts.ts_list[0].name}"\n'
'      f"  shape={pos_mts.ts_list[0].data.shape}")\n'
'\n'
'# Discrete feature\n'
'trial_ts = exp.trial_type\n'
'print(f"\\ntrial_type.discrete:  {trial_ts.discrete}")\n'
'print(f"trial_type.int_data:  {trial_ts.int_data[:8]}...")\n'
'print(f"trial_type has copula: {trial_ts.copula_normal_data is not None}")'
))

cells.append(md_cell(
"### Batch spike reconstruction\n"
"\n"
"[`reconstruct_all_neurons()`](https://driada.readthedocs.io/en/latest/api/experiment/core.html#driada.experiment.exp_base.Experiment.reconstruct_all_neurons)\n"
"applies the same reconstruction method across\n"
"the whole population.  Key parameters include `method` (`'wavelet'` or\n"
"`'threshold'`), `n_iter` (number of iterative detection passes), and\n"
"`show_progress` (display a progress bar).  After reconstruction, per-neuron\n"
"quality metrics (wavelet SNR, R-squared, event counts) are available."
))

cells.append(code_cell(
"exp.reconstruct_all_neurons(method='threshold', n_iter=3, show_progress=True)\n"
'print(f"[OK] Reconstructed spikes for {exp.n_cells} neurons")\n'
"\n"
"# Collect per-neuron quality metrics\n"
"snr_list = []\n"
"r2_list = []\n"
"event_counts = []\n"
"\n"
"for n in exp.neurons:\n"
"    snr_list.append(n.get_wavelet_snr())\n"
"    r2_list.append(n.get_reconstruction_r2())\n"
"    event_counts.append(n.get_event_count())\n"
"\n"
"snr_arr = np.array(snr_list)\n"
"r2_arr = np.array(r2_list)\n"
"evt_arr = np.array(event_counts)\n"
"\n"
'print(f"\\nPopulation quality summary ({exp.n_cells} neurons):")\n'
'print(f"  Wavelet SNR:  {np.mean(snr_arr):.2f} +/- {np.std(snr_arr):.2f}"\n'
'      f"  (range {np.min(snr_arr):.2f} - {np.max(snr_arr):.2f})")\n'
'print(f"  Recon R2:     {np.mean(r2_arr):.4f} +/- {np.std(r2_arr):.4f}"\n'
'      f"  (range {np.min(r2_arr):.4f} - {np.max(r2_arr):.4f})")\n'
'print(f"  Event count:  {np.mean(evt_arr):.1f} +/- {np.std(evt_arr):.1f}"\n'
'      f"  (range {np.min(evt_arr)} - {np.max(evt_arr)})")'
))

cells.append(md_cell(
"### Neural data access\n"
"\n"
"Neural activity is stored in two complementary ways:\n"
"\n"
"| View | Description |\n"
"|---|---|\n"
"| `exp.calcium` | `MultiTimeSeries` (n_neurons, n_frames) -- convenient for population-level analysis (dimensionality reduction, RSA, decoding) |\n"
"| `exp.neurons` | List of `Neuron` objects -- for single-cell analysis (reconstruction, kinetics, quality) |"
))

cells.append(code_cell(
'# Population-level: full calcium matrix as MultiTimeSeries\n'
'print(f"exp.calcium:        {type(exp.calcium).__name__}"\n'
'      f"  shape={exp.calcium.data.shape}")\n'
'has_spikes = exp.spikes is not None and exp.spikes.data.any()\n'
'print(f"exp.spikes:         {\'available\' if has_spikes else \'not provided\'}")\n'
'\n'
'# Single-neuron level: list of Neuron objects\n'
'neuron = exp.neurons[0]\n'
'print(f"\\nexp.neurons:        {len(exp.neurons)} Neuron objects")\n'
'print(f"neuron.cell_id:     {neuron.cell_id}")\n'
'print(f"neuron.ca:          {type(neuron.ca).__name__}"\n'
'      f"  shape={neuron.ca.data.shape}")\n'
'print(f"neuron.sp:          "\n'
'      f"{\'shape=\' + str(neuron.sp.data.shape) if neuron.sp else \'None (no spikes provided)\'}")\n'
'print(f"neuron.fps:         {neuron.fps}")\n'
'# See Section 2 for spike reconstruction, event detection,\n'
'# kinetics optimization, and other Neuron methods.'
))

cells.append(md_cell(
"### Save and reload\n"
"\n"
"The entire Experiment (neural data + features + metadata) can be serialized\n"
"with [`save_exp_to_pickle`](https://driada.readthedocs.io/en/latest/api/experiment/loading.html#driada.experiment.exp_build.save_exp_to_pickle)\n"
"and restored with [`load_exp_from_pickle`](https://driada.readthedocs.io/en/latest/api/experiment/loading.html#driada.experiment.exp_build.load_exp_from_pickle)\n"
"for fast roundtrip storage."
))

cells.append(code_cell(
'pkl_path = os.path.join(tempfile.gettempdir(), "demo_experiment.pkl")\n'
'save_exp_to_pickle(exp, pkl_path, verbose=False)\n'
'file_size_mb = os.path.getsize(pkl_path) / 1024 / 1024\n'
'print(f"Saved:  {pkl_path} ({file_size_mb:.1f} MB)")\n'
'\n'
'exp_loaded = load_exp_from_pickle(pkl_path, verbose=False)\n'
'print(f"Loaded: {exp_loaded.n_cells} neurons, {exp_loaded.n_frames} frames")\n'
'\n'
'# Verify roundtrip\n'
'assert exp_loaded.n_cells == exp.n_cells\n'
'assert exp_loaded.n_frames == exp.n_frames\n'
'assert np.allclose(exp_loaded.calcium.data, exp.calcium.data)\n'
'print("Roundtrip verified -- data matches.")\n'
'\n'
'# Clean up\n'
'os.remove(pkl_path)\n'
'print(f"Cleaned up {pkl_path}")'
))

# ===== PART 2: MINI DEMOS ==================================================

cells.append(md_cell(
"Now that you know how DRIADA stores data, let's see what you can do with it.\n"
"The following demos use a synthetic population with known ground truth, so we\n"
"can verify that each analysis recovers the correct answer."
))

# ----- 2.1 Synthetic experiment with ground truth ---------------------------

cells.append(md_cell(
"## 2. Quick tour with a synthetic population\n"
"\n"
"### 2.1 Synthetic experiment with ground truth\n"
"\n"
"[`generate_tuned_selectivity_exp`](https://driada.readthedocs.io/en/latest/api/experiment/synthetic.html#driada.experiment.synthetic.generators.generate_tuned_selectivity_exp)\n"
"creates a synthetic population where each neuron group has known selectivity\n"
"to specific features.  This lets us verify that downstream analyses recover\n"
"the ground truth."
))

cells.append(code_cell(
"population = [\n"
"    {'name': 'hd_cells',    'count': 10, 'features': ['head_direction']},\n"
"    {'name': 'speed_cells', 'count': 10, 'features': ['speed']},\n"
"    {'name': 'event_cells', 'count': 10, 'features': ['event_0']},\n"
"    {'name': 'mixed',       'count': 5,  'features': ['head_direction', 'speed']},\n"
"    {'name': 'background',  'count': 15, 'features': []},\n"
"]\n"
"\n"
"exp_demo = generate_tuned_selectivity_exp(\n"
"    population=population,\n"
"    duration=600,\n"
"    fps=20,\n"
"    n_discrete_features=1,\n"
"    seed=42,\n"
"    verbose=True,\n"
")\n"
"\n"
'print(f"\\nNeurons: {exp_demo.n_cells}, Frames: {exp_demo.n_frames}")\n'
'print(f"Features: {sorted(exp_demo.dynamic_features.keys())}")\n'
'print(f"Ground-truth selective pairs: {len(exp_demo.ground_truth[\'expected_pairs\'])}")'
))

# ----- 2.2 INTENSE mini-demo -----------------------------------------------

cells.append(md_cell(
"### 2.2 INTENSE -- single-neuron selectivity\n"
"\n"
"[INTENSE](https://driada.readthedocs.io/en/latest/api/intense/pipelines.html)\n"
"tests every neuron-feature pair for significant mutual information using a\n"
"two-stage permutation test.  The function returns an\n"
"[`IntenseResults`](https://driada.readthedocs.io/en/latest/api/intense/base.html#driada.intense.io.IntenseResults)\n"
"object -- the primary container for all INTENSE outputs (statistics,\n"
"significance, metadata).  See\n"
"[Notebook 02](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/02_selectivity_detection_intense.ipynb)\n"
"for the full walkthrough."
))

cells.append(code_cell(
"from driada.intense import compute_cell_feat_significance\n"
"\n"
"stats, significance, info, results = compute_cell_feat_significance(\n"
"    exp_demo, verbose=True,\n"
")\n"
"\n"
"# results is an IntenseResults object -- the primary container for outputs.\n"
"# Validate detections against synthetic ground truth:\n"
"metrics = results.validate_against_ground_truth(exp_demo.ground_truth)"
))

cells.append(code_cell(
"# results.stats is a nested dict: results.stats[neuron_id][feat_name] -> {'me': ...}\n"
"# Build MI matrix from IntenseResults for visualization\n"
"neuron_ids = sorted(results.stats.keys())\n"
"feat_names = sorted(next(iter(results.stats.values())).keys())\n"
"mi_matrix = np.array([[results.stats[nid][fn].get('me', 0.0)\n"
"                        for fn in feat_names] for nid in neuron_ids])\n"
"\n"
"fig, ax = plt.subplots(figsize=(8, 6))\n"
"im = ax.imshow(mi_matrix.T, aspect='auto', cmap='viridis')\n"
"ax.set_xlabel('Neuron index')\n"
"ax.set_yticks(range(len(feat_names)))\n"
"ax.set_yticklabels(feat_names, fontsize=9)\n"
"ax.set_title('Mutual information (neuron x feature)')\n"
"plt.colorbar(im, ax=ax, label='MI (bits)')\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(code_cell(
"# Convenience method: get neurons with at least 1 significant feature\n"
"sig_dict = exp_demo.get_significant_neurons(min_nspec=1)\n"
"\n"
'print(f"Neurons with >= 1 significant feature: {len(sig_dict)} / {exp_demo.n_cells}")\n'
'for nid, feats in sorted(sig_dict.items())[:10]:\n'
'    print(f"  neuron {nid:3d}: {feats}")\n'
'if len(sig_dict) > 10:\n'
'    print(f"  ... ({len(sig_dict) - 10} more)")'
))

# ----- 2.3 Dimensionality reduction mini-demo ----------------------------------------------------

cells.append(md_cell(
"### 2.3 Dimensionality reduction -- population geometry\n"
"\n"
"INTENSE identifies *which* neurons respond to *which* features. But how do\n"
"these neurons collectively represent information? Dimensionality reduction\n"
"reveals the population-level geometry.\n"
"\n"
"Project population activity onto a 2D Isomap embedding to see how behavioral\n"
"variables are encoded in the neural manifold.  `n_neighbors` controls the\n"
"locality of the manifold approximation (20-50 is typical); `ds` downsamples\n"
"the time axis for faster computation.  See\n"
"[Notebook 03](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/03_population_geometry_dr.ipynb)\n"
"for the full walkthrough."
))

cells.append(code_cell(
"embedding = exp_demo.create_embedding(\n"
"    'isomap', n_components=2, n_neighbors=30, ds=3,  # neighbors: local vs global\n"
")\n"
"# ds=3 downsamples the time axis by 3x for speed\n"
"\n"
"fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
"\n"
"# head_direction must be downsampled to match embedding length\n"
"ds = 3\n"
"hd_ds = exp_demo.head_direction.data[::ds][:len(embedding)]\n"
"speed_ds = exp_demo.speed.data[::ds][:len(embedding)]\n"
"\n"
"ax = axes[0]\n"
"sc = ax.scatter(embedding[:, 0], embedding[:, 1],\n"
"                c=hd_ds, cmap='hsv',\n"
"                s=1, alpha=0.5)\n"
"ax.set_title('Isomap colored by head direction')\n"
"plt.colorbar(sc, ax=ax, label='head direction (rad)')\n"
"\n"
"ax = axes[1]\n"
"sc = ax.scatter(embedding[:, 0], embedding[:, 1],\n"
"                c=speed_ds, cmap='plasma',\n"
"                s=1, alpha=0.5)\n"
"ax.set_title('Isomap colored by speed')\n"
"plt.colorbar(sc, ax=ax, label='speed')\n"
"\n"
"for ax in axes:\n"
"    ax.set_xlabel('Isomap 1')\n"
"    ax.set_ylabel('Isomap 2')\n"
"\n"
"plt.tight_layout()\n"
"plt.show()"
))

# ----- 2.4 Network mini-demo -----------------------------------------------

cells.append(md_cell(
"### 2.4 Network -- functional connectivity\n"
"\n"
"Individual selectivity and population geometry capture different facets of\n"
"the data. Network analysis adds a third: pairwise functional relationships\n"
"between neurons.\n"
"\n"
"Test all neuron pairs for shared mutual information, build a binary\n"
"connectivity graph, and inspect its topology.  See\n"
"[Notebook 04](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/04_network_analysis.ipynb)\n"
"for the full walkthrough.\n"
"\n"
"[Network API reference](https://driada.readthedocs.io/en/latest/api/network/core.html)"
))

cells.append(code_cell(
"from driada.intense import compute_cell_cell_significance\n"
"from driada.network import Network\n"
"import scipy.sparse as sp\n"
"import networkx as nx\n"
"\n"
"cell_sim, cell_sig, cell_pvals, cell_ids, cell_info = compute_cell_cell_significance(\n"
"    exp_demo, verbose=True,\n"
")\n"
"\n"
"net = Network(adj=sp.csr_matrix(cell_sig), preprocessing='giant_cc')\n"
"degrees = [d for _, d in net.graph.degree()]\n"
"clustering = nx.average_clustering(net.graph)\n"
'print(f"\\nNetwork: {net.n} nodes, {net.graph.number_of_edges()} edges")\n'
'print(f"Mean degree: {np.mean(degrees):.1f}")\n'
'print(f"Clustering coefficient: {clustering:.3f}")'
))

cells.append(md_cell(
"The adjacency matrix below shows significant neuron-neuron correlations.\n"
"Notice the block-diagonal structure: neurons sharing the same ground-truth\n"
"selectivity (head direction, speed, events) form dense clusters because\n"
"they co-vary with the same behavioral signal.  Mixed-selectivity neurons\n"
"(indices 30-34) connect to multiple blocks.  Background neurons (35-49)\n"
"have sparse or no connections."
))

cells.append(code_cell(
"fig, ax = plt.subplots(figsize=(6, 6))\n"
"ax.imshow(cell_sig, cmap='Greys', interpolation='nearest')\n"
"ax.set_xlabel('Neuron')\n"
"ax.set_ylabel('Neuron')\n"
"ax.set_title('Functional connectivity (significant pairs)')\n"
"\n"
"# Annotate population group boundaries\n"
"boundaries = [0, 10, 20, 30, 35, 50]\n"
"labels = ['HD', 'Speed', 'Event', 'Mixed', 'Bkg']\n"
"for b in boundaries[1:-1]:\n"
"    ax.axhline(b - 0.5, color='red', linewidth=0.5, alpha=0.5)\n"
"    ax.axvline(b - 0.5, color='red', linewidth=0.5, alpha=0.5)\n"
"for i, (start, label) in enumerate(zip(boundaries[:-1], labels)):\n"
"    mid = (start + boundaries[i+1]) / 2\n"
"    ax.text(mid, -2.5, label, ha='center', fontsize=8, fontweight='bold',\n"
"            clip_on=False)\n"
"ax.set_xlim(-0.5, 49.5)\n"
"ax.set_ylim(49.5, -0.5)\n"
"\n"
"plt.tight_layout()\n"
"plt.show()"
))

# ----- Closing --------------------------------------------------------------

cells.append(md_cell(
"## Next steps\n"
"\n"
"This notebook gave you a quick tour of DRIADA's core capabilities.\n"
"Dive deeper with the detailed tutorials:\n"
"\n"
"1. [**Neuron analysis**](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/01_data_loading_and_neurons.ipynb) -- spike reconstruction, kinetics optimization, quality metrics, surrogates.\n"
"2. [**INTENSE selectivity detection**](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/02_selectivity_detection_intense.ipynb) -- two-stage permutation test, tuning curves, ground-truth validation.\n"
"3. [**Population geometry & dimensionality reduction**](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/03_population_geometry_dr.ipynb) -- PCA, UMAP, Isomap, Laplacian Eigenmaps, sequential DR, alignment metrics.\n"
"4. [**Network analysis**](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/04_network_analysis.ipynb) -- degree distributions, community detection, spectral analysis, null models.\n"
"5. [**Advanced capabilities**](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/05_advanced_capabilities.ipynb) -- INTENSE + DR pipeline, leave-one-out importance, RSA, RNN analysis.\n"
"\n"
"**Standalone examples** (run directly, no external data needed):\n"
"- [intense_basic_usage](https://github.com/iabs-neuro/driada/tree/main/examples/intense_basic_usage) -- Minimal INTENSE workflow\n"
"- [compare_dr_methods](https://github.com/iabs-neuro/driada/tree/main/examples/compare_dr_methods) -- Dimensionality reduction method comparison with quality metrics\n"
"- [network_analysis](https://github.com/iabs-neuro/driada/tree/main/examples/network_analysis) -- Network construction and structural analysis\n"
"\n"
"[All examples](https://github.com/iabs-neuro/driada/tree/main/examples)"
))

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------

nb = nbformat.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3,
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10.19",
    },
    "colab": {
        "provenance": [],
        "toc_visible": True,
    },
})
nb.cells = cells

# Write notebook
output_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "00_driada_overview.ipynb",
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c.cell_type == 'code')} code)")
