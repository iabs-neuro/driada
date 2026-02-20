#!/usr/bin/env python
"""
Generate Notebook 01: Data Loading & Neuron Analysis
=====================================================

Assembles content from 4 validated examples into a single Colab-ready
Jupyter notebook using nbformat.

Source examples:
  1. examples/data_loading/load_data_example.py
  2. examples/neuron_basic_usage/neuron_basic_usage.py
  3. examples/spike_reconstruction/threshold_vs_wavelet_optimization.py
  4. examples/spike_reconstruction/spike_reconstruction_comparison.py
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
"# Data loading & neuron analysis\n"
"\n"
"This notebook covers the essential first steps of working with **DRIADA** --\n"
"loading calcium imaging data, inspecting core objects, reconstructing spikes,\n"
"and assessing recording quality.\n"
"\n"
"**What you will learn:**\n"
"\n"
"1. **Loading your data** -- wrap numpy arrays (from Suite2P, CaImAn, DeepLabCut, etc.) into a DRIADA `Experiment`.\n"
"2. **Single neuron analysis** -- create a `Neuron`, reconstruct spikes, optimize kinetics, compute quality metrics, and generate surrogates.\n"
"3. **Threshold vs wavelet reconstruction** -- compare two spike detection methods across four optimization modes.\n"
"4. **Method agreement** -- quantify event-region overlap between threshold and wavelet at varying tolerance."
))

cells.append(code_cell(
"!pip install -q driada\n"
"%matplotlib inline\n"
"\n"
"import os\n"
"import time\n"
"import tempfile\n"
"import warnings\n"
"\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"from matplotlib.patches import Patch\n"
"\n"
"from driada.experiment import (\n"
"    load_exp_from_aligned_data,\n"
"    save_exp_to_pickle,\n"
"    load_exp_from_pickle,\n"
"    generate_synthetic_exp,\n"
")\n"
"from driada.experiment.neuron import Neuron\n"
"from driada.experiment.synthetic import generate_pseudo_calcium_signal"
))

# ===== SECTION 1: LOADING YOUR DATA ========================================

cells.append(md_cell(
"## 1. Loading your data into DRIADA\n"
"\n"
"You have numpy arrays from your recording pipeline (Suite2P, CaImAn,\n"
"DeepLabCut, etc.).  DRIADA wraps them into an **Experiment** object that\n"
"keeps neural activity and behavioral features aligned and annotated.\n"
"\n"
"The only required key is `'calcium'` -- a `(n_neurons, n_frames)` array of\n"
"fluorescence traces.  Everything else you pass becomes a **dynamic feature**\n"
"(one value per timepoint)."
))

cells.append(code_cell(
"# In practice: raw = np.load('your_recording.npz')\n"
"# Here we generate synthetic arrays that mimic a real recording.\n"
"\n"
"np.random.seed(0)\n"
"n_neurons, n_frames = 50, 10000\n"
"fps = 30.0\n"
"\n"
"calcium = np.random.randn(n_neurons, n_frames) * 0.1          # (50, 10000)\n"
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

cells.append(md_cell(
"### Feature types and aggregation\n"
"\n"
"DRIADA auto-detects whether each feature is **continuous** or **discrete**.\n"
"You can override the detection with a `feature_types` dict.  Valid type\n"
"strings include: `continuous`, `circular`, `categorical`, `binary`, `count`.\n"
"\n"
"`aggregate_features` groups related 1D features into a single\n"
"`MultiTimeSeries` (e.g. x_pos + y_pos -> position_2d).\n"
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
'    "calcium": calcium,               # (50, 10000)\n'
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
'    # create_circular_2d=True is the default: for every circular\n'
'    # feature (here head_direction), DRIADA auto-creates a _2d\n'
'    # version as (cos, sin). This is important because MI estimators\n'
'    # (GCMI, KSG) work on the real line -- a raw angle wraps around\n'
'    # at 0/2pi, breaking the distance metric. The (cos, sin) encoding\n'
'    # maps the circle onto R^2 where Euclidean distance is meaningful.\n'
'    create_circular_2d=True,\n'
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
'# Note the auto-generated features in the list below:\n'
'#   - position_2d:        from aggregate_features (x_pos + y_pos)\n'
'#   - head_direction_2d:  from create_circular_2d (cos + sin encoding)\n'
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
"| **TimeSeries** | A single 1D variable (e.g. `speed`) |\n"
"| **MultiTimeSeries** | Multiple aligned 1D variables stacked into a 2D array (e.g. `position_2d = [x, y]`) |\n"
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
"`reconstruct_all_neurons()` applies the same reconstruction method across\n"
"the whole population.  After reconstruction, per-neuron quality metrics\n"
"(wavelet SNR, R-squared, event counts) are available."
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
"| `exp.calcium` | `MultiTimeSeries` (n_neurons, n_frames) -- convenient for population-level analysis (DR, RSA, decoding) |\n"
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
"with pickle for fast roundtrip storage."
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

# ===== SECTION 2: SINGLE NEURON ANALYSIS ==================================

cells.append(md_cell(
"## 2. Single neuron analysis\n"
"\n"
"Deep dive into individual neuron quality: generate a synthetic calcium\n"
"signal, create a `Neuron` object, reconstruct spikes, optimize kinetics,\n"
"compute quality metrics, and generate surrogates for null-hypothesis testing."
))

cells.append(code_cell(
"# Set random seed for reproducibility\n"
"np.random.seed(42)\n"
"\n"
"# =============================================================================\n"
"# Step 1: Generate Synthetic Calcium Signal\n"
"# =============================================================================\n"
'print("1. Generating synthetic calcium signal...")\n'
"\n"
"signal = generate_pseudo_calcium_signal(\n"
"    duration=200.0,              # Signal duration in seconds\n"
"    sampling_rate=20.0,          # Sampling rate (Hz)\n"
"    event_rate=0.15,             # Average event rate (Hz)\n"
"    amplitude_range=(1.0, 2.0),  # Event amplitude range (dF/F0)\n"
"    decay_time=1.5,              # Calcium decay time constant (seconds)\n"
"    rise_time=0.15,              # Calcium rise time constant (seconds)\n"
"    noise_std=0.05,              # Additive Gaussian noise level\n"
"    kernel='double_exponential'  # Realistic calcium kernel\n"
")\n"
"\n"
'print(f"   [OK] Generated signal: {len(signal)} frames ({len(signal)/20:.1f} seconds)")\n'
"\n"
"# =============================================================================\n"
"# Step 2: Create Neuron Object\n"
"# =============================================================================\n"
'print("\\n2. Creating Neuron object...")\n'
"\n"
"neuron = Neuron(\n"
"    cell_id='example_neuron',\n"
"    ca=signal,              # Calcium signal\n"
"    sp=None,                # No ground-truth spikes (will be reconstructed)\n"
"    fps=20.0                # Sampling rate\n"
")\n"
"\n"
'print(f"   [OK] Neuron created: {neuron.cell_id}")\n'
'print(f"   [OK] Signal length: {neuron.n_frames} frames")\n'
'print(f"   [OK] Sampling rate: {neuron.fps} Hz")'
))

cells.append(md_cell(
"### Spike reconstruction\n"
"\n"
"The **wavelet** method detects calcium transient events via CWT (continuous\n"
"wavelet transform) ridge analysis."
))

cells.append(code_cell(
"# =============================================================================\n"
"# Step 3: Reconstruct Spikes with Wavelet Method\n"
"# =============================================================================\n"
'print("3. Reconstructing spikes using wavelet method...")\n'
"\n"
"spikes = neuron.reconstruct_spikes(\n"
"    method='wavelet',\n"
"    create_event_regions=True  # Create event regions for quality metrics\n"
")\n"
"\n"
"n_events = int(np.sum(neuron.asp.data > 0))\n"
'print(f"   [OK] Detected {n_events} calcium events")\n'
'print(f"   [OK] Spike train stored in neuron.sp")\n'
'print(f"   [OK] Amplitude spikes stored in neuron.asp")'
))

cells.append(md_cell(
"### Kinetics optimization\n"
"\n"
"Fit rise and decay time constants to detected events using the **direct\n"
"measurement** method."
))

cells.append(code_cell(
"# =============================================================================\n"
"# Step 4: Optimize Calcium Kinetics\n"
"# =============================================================================\n"
'print("4. Optimizing calcium kinetics...")\n'
"\n"
"kinetics = neuron.get_kinetics(\n"
"    method='direct',           # Direct measurement from detected events\n"
"    use_cached=False          # Force recomputation\n"
")\n"
"\n"
'print(f"   [OK] Optimized rise time (t_rise): {kinetics[\'t_rise\']:.3f} seconds")\n'
'print(f"   [OK] Optimized decay time (t_off): {kinetics[\'t_off\']:.3f} seconds")\n'
'print(f"   [OK] Events used: {kinetics[\'n_events_detected\']}")'
))

cells.append(md_cell(
"### Quality metrics\n"
"\n"
"- **Wavelet SNR** -- ratio of event amplitude to baseline noise\n"
"- **R-squared** -- reconstruction quality (1.0 = perfect, >0.7 = good)\n"
"- **Event-only R-squared** -- quality restricted to event regions\n"
"- **NRMSE** -- normalized root mean squared error (lower is better)\n"
"- **NMAE** -- normalized mean absolute error (lower is better)"
))

cells.append(code_cell(
"# =============================================================================\n"
"# Step 5: Calculate Wavelet SNR\n"
"# =============================================================================\n"
'print("5. Computing wavelet SNR...")\n'
"\n"
"wavelet_snr = neuron.get_wavelet_snr()\n"
"\n"
'print(f"   [OK] Wavelet SNR: {wavelet_snr:.2f}")\n'
'print(f"       (Ratio of event amplitude to baseline noise)")\n'
"\n"
"# =============================================================================\n"
"# Step 6: Calculate Reconstruction Quality Metrics\n"
"# =============================================================================\n"
'print("\\n6. Computing reconstruction quality metrics...")\n'
"\n"
"# R2 (coefficient of determination)\n"
"r2 = neuron.get_reconstruction_r2()\n"
'print(f"   [OK] Reconstruction R2: {r2:.4f}")\n'
'print(f"       (1.0 = perfect, >0.7 = good quality)")\n'
"\n"
"# Event-only R2 (focuses on event regions)\n"
"r2_events = neuron.get_reconstruction_r2(event_only=True)\n"
'print(f"   [OK] Event-only R2: {r2_events:.4f}")\n'
'print(f"       (Quality in event regions only)")\n'
"\n"
"# Normalized RMSE\n"
"nrmse = neuron.get_nrmse()\n"
'print(f"   [OK] Normalized RMSE: {nrmse:.4f}")\n'
'print(f"       (Lower is better)")\n'
"\n"
"# Normalized MAE\n"
"nmae = neuron.get_nmae()\n"
'print(f"   [OK] Normalized MAE: {nmae:.4f}")\n'
'print(f"       (Lower is better)")'
))

cells.append(md_cell(
"### Surrogate generation\n"
"\n"
"Four surrogate methods for null-hypothesis testing:\n"
"\n"
"| Method | Type | Preserves |\n"
"|---|---|---|\n"
"| **roll_based** | Calcium | Autocorrelation structure, amplitude distribution |\n"
"| **waveform_based** | Calcium | Individual waveform shapes, event count |\n"
"| **chunks_based** | Calcium | Local structure within chunks |\n"
"| **isi_based** | Spikes | Inter-spike interval distribution |"
))

cells.append(code_cell(
"# =============================================================================\n"
"# Step 7: Surrogate Generation Methods\n"
"# =============================================================================\n"
'print("7. Surrogate generation methods...")\n'
'print("   Three calcium surrogate types and one spike surrogate type.")\n'
"\n"
"# --- Calcium surrogates ---\n"
"\n"
"# 7a. Roll-based: circular shift preserving all autocorrelations\n"
"shuffled_roll = neuron.get_shuffled_calcium(method='roll_based', seed=42)\n"
'print(f"\\n   [Roll-based] Circular shift surrogate:")\n'
'print(f"       Mean: {np.mean(shuffled_roll):.4f}  (original: {np.mean(neuron.ca.data):.4f})")\n'
'print(f"       Std:  {np.std(shuffled_roll):.4f}  (original: {np.std(neuron.ca.data):.4f})")\n'
'print(f"       Preserves: autocorrelation structure, amplitude distribution")\n'
"\n"
"# 7b. Waveform-based: shuffle detected spike times, reconstruct calcium\n"
"shuffled_wf = neuron.get_shuffled_calcium(method='waveform_based', seed=42)\n"
'print(f"\\n   [Waveform-based] Spike-shuffle + reconstruct surrogate:")\n'
'print(f"       Mean: {np.mean(shuffled_wf):.4f}  (original: {np.mean(neuron.ca.data):.4f})")\n'
'print(f"       Std:  {np.std(shuffled_wf):.4f}  (original: {np.std(neuron.ca.data):.4f})")\n'
'print(f"       Preserves: individual waveform shapes, event count")\n'
"\n"
"# 7c. Chunks-based: divide signal into chunks and reorder\n"
"shuffled_chunks = neuron.get_shuffled_calcium(method='chunks_based', seed=42)\n"
'print(f"\\n   [Chunks-based] Chunk reordering surrogate:")\n'
'print(f"       Mean: {np.mean(shuffled_chunks):.4f}  (original: {np.mean(neuron.ca.data):.4f})")\n'
'print(f"       Std:  {np.std(shuffled_chunks):.4f}  (original: {np.std(neuron.ca.data):.4f})")\n'
'print(f"       Preserves: local structure within chunks")\n'
"\n"
"# --- Spike surrogates ---\n"
"\n"
"# 7d. ISI-based: shuffle inter-spike intervals, preserving ISI distribution\n"
"shuffled_sp = neuron.get_shuffled_spikes(method='isi_based', seed=42)\n"
"original_spike_count = int(np.sum(neuron.sp.data > 0))\n"
"shuffled_spike_count = int(np.sum(shuffled_sp > 0))\n"
'print(f"\\n   [ISI-based] Spike train surrogate:")\n'
'print(f"       Spike count: {shuffled_spike_count}  (original: {original_spike_count})")\n'
'print(f"       Preserves: inter-spike interval distribution")'
))

# ===== SECTION 3: THRESHOLD VS WAVELET ====================================

cells.append(md_cell(
"## 3. Threshold vs wavelet reconstruction\n"
"\n"
"Two detection methods, four optimization modes:\n"
"\n"
"| Mode | Description |\n"
"|---|---|\n"
"| Default kinetics | Single pass, preset rise/decay times |\n"
"| Optimized kinetics | Single pass + fit kinetics to your signal |\n"
"| Iterative n=2 + optimized | Detect -> subtract -> detect + optimize |\n"
"| Iterative n=3 + optimized | More passes for weaker events |\n"
"\n"
"**Threshold** is faster; **wavelet** is more sensitive, especially for low\n"
"SNR or overlapping events."
))

# Helper functions cell -- contains docstrings, so we use string concat
# to avoid triple-quote conflicts
_sec3_helpers = '''\
def create_synthetic_neuron(duration=60.0, fps=30.0, event_rate=0.3, seed=42):
    """Generate synthetic calcium signal with known ground truth.

    Uses kinetics different from defaults to demonstrate optimization benefit.
    Default kinetics: t_rise=0.25s, t_off=2.0s
    True kinetics: t_rise=0.10s, t_off=0.8s (faster indicator)
    """
    np.random.seed(seed)

    # Kinetics faster than defaults - optimization should help
    t_rise_true = 0.10   # Faster than default 0.25s
    t_off_true = 0.8     # Faster than default 2.0s

    signal = generate_pseudo_calcium_signal(
        duration=duration,
        sampling_rate=fps,
        event_rate=event_rate,
        amplitude_range=(0.3, 1.2),
        decay_time=t_off_true,
        rise_time=t_rise_true,
        noise_std=0.04,               # Moderate noise
        kernel='double_exponential'
    )

    return signal, {'t_rise_true': t_rise_true, 't_off_true': t_off_true, 'event_rate': event_rate}


def reconstruct_with_mode(neuron, fps, method, mode_name, iterative=False, n_iter=1,
                          optimize=False, adaptive_thresholds=False):
    """Reconstruct with specified method and mode.

    Returns dict with reconstruction results.
    """
    start = time.time()

    # Suppress the default kinetics warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        if method == 'threshold':
            neuron.reconstruct_spikes(
                method='threshold',
                n_mad=4.0,                    # Balanced threshold for noisy data
                min_duration_frames=2,        # Allow shorter events
                create_event_regions=True,
                iterative=iterative,
                n_iter=n_iter,
                adaptive_thresholds=adaptive_thresholds
            )
        else:  # wavelet
            neuron.reconstruct_spikes(
                method='wavelet',
                create_event_regions=True,
                iterative=iterative,
                n_iter=n_iter,
                adaptive_thresholds=adaptive_thresholds
            )

    # Optionally optimize kinetics
    optimized = False
    if optimize:
        result = neuron.optimize_kinetics(
            method='direct',
            fps=fps,
            update_reconstruction=True,
            detection_method=method,
            # Pass through to reconstruct_spikes
            n_mad=4.0,
            iterative=iterative,
            n_iter=n_iter,
            adaptive_thresholds=adaptive_thresholds
        )
        optimized = result.get('optimized', False)

    time_total = time.time() - start

    # Get kinetics (optimization may fall back to defaults if events
    # are poorly characterized from single-pass with wrong kinetics)
    t_rise = neuron.t_rise if neuron.t_rise else neuron.default_t_rise
    t_off = neuron.t_off if neuron.t_off else neuron.default_t_off

    # Count events
    if method == 'threshold':
        n_events = len(neuron.threshold_events) if neuron.threshold_events else 0
    else:
        n_events = len(neuron.wvt_ridges) if neuron.wvt_ridges else 0

    # Quality metrics from Neuron API
    r2 = neuron.get_reconstruction_r2()
    corr = np.corrcoef(neuron.ca.data, neuron._reconstructed.data)[0, 1]

    return {
        'mode': mode_name,
        'reconstruction': neuron._reconstructed.data,
        't_rise': t_rise / fps,
        't_off': t_off / fps,
        'n_events': n_events,
        'optimized': optimized,
        'r2': r2,
        'correlation': corr,
        'time': time_total,
    }


def reconstruct_all_modes(signal, fps, method):
    """Run all 4 reconstruction modes for a given method.

    Modes:
    1. Default kinetics (single pass)
    2. Optimized kinetics (single pass)
    3. Iterative n_iter=2 + optimized kinetics
    4. Iterative n_iter=3 + optimized kinetics
    """
    results = []

    # Mode 1: Default kinetics (single pass, no optimization)
    print(f"   Mode 1: Default kinetics...")
    neuron = Neuron(cell_id=f'{method}_default', ca=signal.copy(), sp=None, fps=fps)
    results.append(reconstruct_with_mode(
        neuron, fps, method,
        mode_name='Default kinetics',
        iterative=False, n_iter=1, optimize=False
    ))

    # Mode 2: Optimized kinetics (single pass + optimization)
    print(f"   Mode 2: Optimized kinetics...")
    neuron = Neuron(cell_id=f'{method}_optimized', ca=signal.copy(), sp=None, fps=fps)
    results.append(reconstruct_with_mode(
        neuron, fps, method,
        mode_name='Optimized kinetics',
        iterative=False, n_iter=1, optimize=True
    ))

    # Mode 3: Iterative n_iter=2 + optimized kinetics
    print(f"   Mode 3: Iterative (n_iter=2) + optimized...")
    neuron = Neuron(cell_id=f'{method}_iter2', ca=signal.copy(), sp=None, fps=fps)
    results.append(reconstruct_with_mode(
        neuron, fps, method,
        mode_name='Iterative n=2 + opt',
        iterative=True, n_iter=2, optimize=True, adaptive_thresholds=True
    ))

    # Mode 4: Iterative n_iter=3 + optimized kinetics
    print(f"   Mode 4: Iterative (n_iter=3) + optimized...")
    neuron = Neuron(cell_id=f'{method}_iter3', ca=signal.copy(), sp=None, fps=fps)
    results.append(reconstruct_with_mode(
        neuron, fps, method,
        mode_name='Iterative n=3 + opt',
        iterative=True, n_iter=3, optimize=True, adaptive_thresholds=True
    ))

    return results'''

cells.append(code_cell(_sec3_helpers))

cells.append(code_cell(
"# Create synthetic neuron with non-default kinetics (t_rise=0.10s, t_off=0.8s)\n"
'print("1. Generating synthetic calcium signal...")\n'
"fps = 30.0\n"
"signal, ground_truth = create_synthetic_neuron(duration=300.0, fps=fps, seed=42)  # 5 minutes\n"
'print(f"   Signal: {len(signal)} frames ({len(signal)/fps:.1f} seconds)")\n'
'print(f"   Ground truth: t_rise={ground_truth[\'t_rise_true\']:.3f}s, "\n'
'      f"t_off={ground_truth[\'t_off_true\']:.3f}s")'
))

cells.append(code_cell(
"# Threshold reconstruction - all 4 modes\n"
'print("2. Threshold-based reconstruction (FAST)...")\n'
"threshold_results = reconstruct_all_modes(signal, fps, method='threshold')"
))

cells.append(code_cell(
"# Wavelet reconstruction - all 4 modes\n"
'print("3. Wavelet-based reconstruction (SENSITIVE)...")\n'
"wavelet_results = reconstruct_all_modes(signal, fps, method='wavelet')"
))

cells.append(code_cell(
"# Print summary table\n"
"time_axis = np.arange(len(signal)) / fps\n"
'print("=" * 120)\n'
'print("RECONSTRUCTION QUALITY SUMMARY")\n'
'print("=" * 120)\n'
'print(f"{\'Method\':<12} {\'Mode\':<22} {\'Events\':<8} {\'t_rise(s)\':<11} {\'t_off(s)\':<11} "\n'
'      f"{\'R^2\':<8} {\'Corr\':<8} {\'Opt\':<6} {\'Time(s)\':<10}")\n'
'print("-" * 120)\n'
"\n"
"for res in threshold_results:\n"
'    opt = "yes" if res[\'optimized\'] else "-"\n'
'    print(f"{\'Threshold\':<12} {res[\'mode\']:<22} {res[\'n_events\']:<8} "\n'
'          f"{res[\'t_rise\']:<11.3f} {res[\'t_off\']:<11.3f} "\n'
'          f"{res[\'r2\']:<8.4f} {res[\'correlation\']:<8.4f} "\n'
'          f"{opt:<6} {res[\'time\']:<10.4f}")\n'
"\n"
'print("-" * 120)\n'
"\n"
"for res in wavelet_results:\n"
'    opt = "yes" if res[\'optimized\'] else "-"\n'
'    print(f"{\'Wavelet\':<12} {res[\'mode\']:<22} {res[\'n_events\']:<8} "\n'
'          f"{res[\'t_rise\']:<11.3f} {res[\'t_off\']:<11.3f} "\n'
'          f"{res[\'r2\']:<8.4f} {res[\'correlation\']:<8.4f} "\n'
'          f"{opt:<6} {res[\'time\']:<10.4f}")\n'
"\n"
"# Calculate speedup\n"
"threshold_total_time = sum(r['time'] for r in threshold_results)\n"
"wavelet_total_time = sum(r['time'] for r in wavelet_results)\n"
"speedup = wavelet_total_time / threshold_total_time\n"
"\n"
'print("\\n" + "=" * 80)\n'
'print(f"PERFORMANCE: Threshold is {speedup:.1f}x faster than Wavelet")\n'
'print(f"  Threshold total: {threshold_total_time:.3f}s")\n'
'print(f"  Wavelet total:   {wavelet_total_time:.3f}s")\n'
'print("=" * 80)'
))

cells.append(code_cell(
"# Visualization: reconstruction traces (2 columns, 5 rows)\n"
"fig = plt.figure(figsize=(16, 12))\n"
"gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)\n"
"\n"
"# Column 1: Threshold method\n"
"# Original signal\n"
"ax0 = fig.add_subplot(gs[0, 0])\n"
"ax0.plot(time_axis, signal, 'k-', linewidth=1, label='Calcium signal')\n"
"ax0.set_ylabel('dF/F0')\n"
"ax0.set_title('THRESHOLD METHOD (fast)', fontweight='bold', fontsize=12)\n"
"ax0.grid(True, alpha=0.3)\n"
"ax0.legend(loc='upper right')\n"
"\n"
"# Reconstruction modes for threshold\n"
"for i, res in enumerate(threshold_results):\n"
"    ax = fig.add_subplot(gs[i+1, 0])\n"
"    ax.plot(time_axis, signal, 'k-', linewidth=0.8, alpha=0.5, label='Original')\n"
"    ax.plot(time_axis, res['reconstruction'], 'b-', linewidth=1.2,\n"
"            label=f\"Reconstruction (R^2={res['r2']:.3f})\")\n"
"\n"
"    ax.set_title(f\"{res['mode']} | t_rise={res['t_rise']:.3f}s, t_off={res['t_off']:.3f}s\",\n"
"                fontsize=10)\n"
"    ax.set_ylabel('dF/F0')\n"
"    if i == 3:\n"
"        ax.set_xlabel('Time (s)')\n"
"    ax.grid(True, alpha=0.3)\n"
"    ax.legend(loc='upper right', fontsize=8)\n"
"\n"
"# Column 2: Wavelet method\n"
"# Original signal\n"
"ax0 = fig.add_subplot(gs[0, 1])\n"
"ax0.plot(time_axis, signal, 'k-', linewidth=1, label='Calcium signal')\n"
"ax0.set_ylabel('dF/F0')\n"
"ax0.set_title('WAVELET METHOD (sensitive)', fontweight='bold', fontsize=12)\n"
"ax0.grid(True, alpha=0.3)\n"
"ax0.legend(loc='upper right')\n"
"\n"
"# Reconstruction modes for wavelet\n"
"for i, res in enumerate(wavelet_results):\n"
"    ax = fig.add_subplot(gs[i+1, 1])\n"
"    ax.plot(time_axis, signal, 'k-', linewidth=0.8, alpha=0.5, label='Original')\n"
"    ax.plot(time_axis, res['reconstruction'], 'r-', linewidth=1.2,\n"
"            label=f\"Reconstruction (R^2={res['r2']:.3f})\")\n"
"\n"
"    ax.set_title(f\"{res['mode']} | t_rise={res['t_rise']:.3f}s, t_off={res['t_off']:.3f}s\",\n"
"                fontsize=10)\n"
"    ax.set_ylabel('dF/F0')\n"
"    if i == 3:\n"
"        ax.set_xlabel('Time (s)')\n"
"    ax.grid(True, alpha=0.3)\n"
"    ax.legend(loc='upper right', fontsize=8)\n"
"\n"
"plt.suptitle(\n"
"    f'Spike reconstruction comparison | Ground truth: t_rise={ground_truth[\"t_rise_true\"]:.3f}s, '\n"
"    f't_off={ground_truth[\"t_off_true\"]:.3f}s | Speedup: {speedup:.1f}x',\n"
"    fontsize=14, fontweight='bold', y=0.995\n"
")\n"
"plt.show()"
))

cells.append(code_cell(
"# Convergence metrics (2x2 subplots)\n"
"fig2, axes = plt.subplots(2, 2, figsize=(14, 9))\n"
"\n"
"modes = range(len(threshold_results))\n"
"mode_labels = ['Default', 'Optimized', 'Iter n=2', 'Iter n=3']\n"
"\n"
"# R^2 comparison\n"
"ax = axes[0, 0]\n"
"ax.plot(modes, [r['r2'] for r in threshold_results], 'bo-', label='Threshold', linewidth=2, markersize=8)\n"
"ax.plot(modes, [r['r2'] for r in wavelet_results], 'ro-', label='Wavelet', linewidth=2, markersize=8)\n"
"ax.set_ylabel('R^2')\n"
"ax.set_title('Reconstruction quality (R^2)', fontweight='bold')\n"
"ax.set_xticks(list(modes))\n"
"ax.set_xticklabels(mode_labels, rotation=15)\n"
"ax.grid(True, alpha=0.3)\n"
"ax.legend()\n"
"\n"
"# Correlation comparison\n"
"ax = axes[0, 1]\n"
"ax.plot(modes, [r['correlation'] for r in threshold_results], 'bo-', label='Threshold', linewidth=2, markersize=8)\n"
"ax.plot(modes, [r['correlation'] for r in wavelet_results], 'ro-', label='Wavelet', linewidth=2, markersize=8)\n"
"ax.set_ylabel('Correlation')\n"
"ax.set_title('Correlation coefficient', fontweight='bold')\n"
"ax.set_xticks(list(modes))\n"
"ax.set_xticklabels(mode_labels, rotation=15)\n"
"ax.grid(True, alpha=0.3)\n"
"ax.legend()\n"
"\n"
"# Event count comparison\n"
"ax = axes[1, 0]\n"
"ax.plot(modes, [r['n_events'] for r in threshold_results], 'bo-', label='Threshold', linewidth=2, markersize=8)\n"
"ax.plot(modes, [r['n_events'] for r in wavelet_results], 'ro-', label='Wavelet', linewidth=2, markersize=8)\n"
"ax.set_ylabel('Events detected')\n"
"ax.set_title('Number of events detected', fontweight='bold')\n"
"ax.set_xticks(list(modes))\n"
"ax.set_xticklabels(mode_labels, rotation=15)\n"
"ax.grid(True, alpha=0.3)\n"
"ax.legend()\n"
"\n"
"# Kinetics comparison (t_rise and t_off with ground truth)\n"
"ax = axes[1, 1]\n"
"ax.plot(modes, [r['t_rise'] for r in threshold_results], 'b^-', label='Thr t_rise', linewidth=2, markersize=8)\n"
"ax.plot(modes, [r['t_rise'] for r in wavelet_results], 'r^-', label='Wvt t_rise', linewidth=2, markersize=8)\n"
"ax.plot(modes, [r['t_off'] for r in threshold_results], 'bs-', label='Thr t_off', linewidth=2, markersize=8)\n"
"ax.plot(modes, [r['t_off'] for r in wavelet_results], 'rs-', label='Wvt t_off', linewidth=2, markersize=8)\n"
"ax.axhline(ground_truth['t_rise_true'], color='g', linestyle='--', linewidth=1.5, alpha=0.7)\n"
"ax.axhline(ground_truth['t_off_true'], color='g', linestyle='--', linewidth=1.5, alpha=0.7, label='Ground truth')\n"
"ax.set_ylabel('Time (s)')\n"
"ax.set_title('Kinetics estimation', fontweight='bold')\n"
"ax.set_xticks(list(modes))\n"
"ax.set_xticklabels(mode_labels, rotation=15)\n"
"ax.grid(True, alpha=0.3)\n"
"ax.legend(fontsize=8)\n"
"\n"
"plt.suptitle(f'Reconstruction metrics by mode | Speedup: {speedup:.1f}x',\n"
"             fontsize=14, fontweight='bold', y=0.995)\n"
"plt.tight_layout()\n"
"plt.show()"
))

# ===== SECTION 4: METHOD AGREEMENT ========================================

cells.append(md_cell(
"## 4. Method agreement analysis\n"
"\n"
"Given the same data, how well do threshold and wavelet agree?  Both methods\n"
"detect calcium transient regions (event start to end) but use different\n"
"signal processing: wavelet uses CWT ridge detection while threshold uses\n"
"MAD-based signal crossing.  Event-region overlap with varying tolerance\n"
"reveals timing differences between detection mechanisms."
))

cells.append(code_cell(
'# Generate 5-neuron experiment, run both methods (iterative n=3)\n'
'print("Generating synthetic calcium imaging data...")\n'
'exp4 = generate_synthetic_exp(\n'
'    n_dfeats=2, n_cfeats=1, nneurons=5, duration=120, fps=20, seed=42  # 2 minutes\n'
')\n'
'\n'
'calcium4 = exp4.calcium\n'
'fps4 = exp4.fps\n'
'n_neurons4 = calcium4.scdata.shape[0]\n'
'time4 = np.arange(calcium4.scdata.shape[1]) / fps4\n'
'\n'
'# Both methods use Neuron-level iterative reconstruction (n_iter=3)\n'
'# to catch overlapping events via residual analysis.\n'
'wavelet_events = []\n'
'threshold_events = []\n'
'\n'
'for neuron in exp4.neurons:\n'
'    # Wavelet: CWT ridge detection on residuals\n'
'    print(f"  Neuron {neuron.cell_id}: wavelet...", end="")\n'
'    neuron.reconstruct_spikes(\n'
'        method="wavelet", iterative=True, n_iter=3, fps=fps4\n'
'    )\n'
'    wavelet_events.append(list(neuron.wvt_ridges))\n'
'\n'
'    # Threshold: MAD-based event detection on residuals\n'
'    print(" threshold...", end="")\n'
'    neuron.reconstruct_spikes(\n'
'        method="threshold", iterative=True, n_iter=3, n_mad=4.0,\n'
'        adaptive_thresholds=True, fps=fps4,\n'
'    )\n'
'    threshold_events.append(list(neuron.threshold_events))\n'
'    print(\n'
'        f" done ({len(wavelet_events[-1])} / {len(threshold_events[-1])} events)"\n'
'    )'
))

cells.append(code_cell(
'# Event counts per neuron\n'
'print("=" * 50)\n'
'print("EVENT COUNTS (iterative, n_iter=3)")\n'
'print("=" * 50)\n'
'print(f"{\'Neuron\':<10} {\'Wavelet\':<14} {\'Threshold\':<14}")\n'
'print("-" * 50)\n'
'for i in range(n_neurons4):\n'
'    n_w = len(wavelet_events[i])\n'
'    n_t = len(threshold_events[i])\n'
'    print(f"{i:<10} {n_w:<14} {n_t:<14}")\n'
'total_w = sum(len(wavelet_events[i]) for i in range(n_neurons4))\n'
'total_t = sum(len(threshold_events[i]) for i in range(n_neurons4))\n'
'print("-" * 50)\n'
'print(f"{\'Total\':<10} {total_w:<14} {total_t:<14}")'
))

cells.append(code_cell(
"# Event region visualization for one neuron\n"
"neuron_idx = 2\n"
"\n"
"fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)\n"
"\n"
"# Plot calcium signal (scaled data)\n"
"ax = axes[0]\n"
'ax.plot(time4, calcium4.scdata[neuron_idx, :], "k-", linewidth=1)\n'
'ax.set_ylabel("Calcium\\n(scaled)")\n'
'ax.set_title(f"Neuron {neuron_idx}: spike reconstruction comparison")\n'
"ax.grid(True, alpha=0.3)\n"
"\n"
"# Plot wavelet-detected event regions\n"
"ax = axes[1]\n"
"for ev in wavelet_events[neuron_idx]:\n"
'    ax.axvspan(ev.start / fps4, ev.end / fps4, alpha=0.5, color="blue")\n'
'ax.set_ylabel("Wavelet\\nEvents")\n'
"ax.set_ylim(-0.1, 1.1)\n"
"ax.grid(True, alpha=0.3)\n"
"ax.legend(\n"
'    handles=[Patch(facecolor="blue", alpha=0.5, label="Event region")],\n'
'    loc="upper right",\n'
")\n"
"\n"
"# Plot threshold-detected event regions\n"
"ax = axes[2]\n"
"for ev in threshold_events[neuron_idx]:\n"
'    ax.axvspan(ev.start / fps4, ev.end / fps4, alpha=0.5, color="red")\n'
'ax.set_ylabel("Threshold\\nEvents")\n'
"ax.set_ylim(-0.1, 1.1)\n"
'ax.set_xlabel("Time (s)")\n'
"ax.grid(True, alpha=0.3)\n"
"ax.legend(\n"
'    handles=[Patch(facecolor="red", alpha=0.5, label="Event region")],\n'
'    loc="upper right",\n'
")\n"
"\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(code_cell(
"# Agreement vs tolerance curve\n"
"# Both methods detect event regions via iterative residual analysis.\n"
"# Sweeping tolerance reveals how well the detected regions align.\n"
"tolerance_sec = np.arange(0, 1.05, 0.05)\n"
"tolerance_frames_arr = (tolerance_sec * fps4).astype(int)\n"
"\n"
"# Extract start/end arrays\n"
"w_starts = [[int(e.start) for e in wavelet_events[i]] for i in range(n_neurons4)]\n"
"w_ends = [[int(e.end) for e in wavelet_events[i]] for i in range(n_neurons4)]\n"
"t_starts = [[int(e.start) for e in threshold_events[i]] for i in range(n_neurons4)]\n"
"t_ends = [[int(e.end) for e in threshold_events[i]] for i in range(n_neurons4)]\n"
"\n"
"agreements = []\n"
"for tol in tolerance_frames_arr:\n"
"    matched = 0\n"
"    for i in range(n_neurons4):\n"
"        for ws, we in zip(w_starts[i], w_ends[i]):\n"
"            # Check if any threshold event overlaps this wavelet event\n"
"            for ts, te in zip(t_starts[i], t_ends[i]):\n"
"                if ts <= (we + tol) and te >= (ws - tol):\n"
"                    matched += 1\n"
"                    break\n"
"    agreements.append(matched / total_w if total_w > 0 else 0)\n"
"\n"
'print("=" * 50)\n'
'print("AGREEMENT VS TOLERANCE")\n'
'print("=" * 50)\n'
'print(f"{\'Tolerance (s)\':<16} {\'Matched\':<12} {\'Agreement\':<12}")\n'
'print("-" * 50)\n'
"for tol_s, agr in zip(tolerance_sec, agreements):\n"
"    if tol_s % 0.25 < 0.01 or abs(tol_s % 0.25 - 0.25) < 0.01:\n"
'        print(f"{tol_s:<16.2f} {int(agr * total_w):<12} {agr:<12.1%}")\n'
"\n"
"# Plot tolerance curve\n"
"fig2, ax2 = plt.subplots(figsize=(8, 5))\n"
"ax2.plot(\n"
'    tolerance_sec, [a * 100 for a in agreements], "ko-", linewidth=2, markersize=4\n'
")\n"
'ax2.set_xlabel("Tolerance (s)")\n'
'ax2.set_ylabel("Agreement (%)")\n'
'ax2.set_title("Event-level agreement: wavelet vs threshold")\n'
"ax2.set_ylim(0, 105)\n"
"ax2.grid(True, alpha=0.3)\n"
"plt.tight_layout()\n"
"plt.show()"
))

# ---------------------------------------------------------------------------
# Assemble notebook
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
    "01_data_loading_and_neurons.ipynb",
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c.cell_type == 'code')} code)")
