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
"# Neuron analysis\n"
"\n"
"[**DRIADA**](https://driada.readthedocs.io) (Dimensionality Reduction for\n"
"Integrated Activity Data) is a Python framework for neural data analysis.\n"
"It bridges two perspectives that are usually treated separately: what\n"
"*individual* neurons encode, and how the *population as a whole* represents\n"
"information.  The typical analysis workflow looks like this:\n"
"\n"
"| Step | Notebook | What it does |\n"
"|---|---|---|\n"
"| **Overview** | [00 -- DRIADA overview](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/00_driada_overview.ipynb) | Core data structures, quick tour of INTENSE, DR, networks |\n"
"| **Neuron analysis** | **01 -- this notebook** | Spike reconstruction, kinetics optimization, quality metrics, surrogates |\n"
"| **Single-neuron selectivity** | [02 -- INTENSE](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/02_selectivity_detection_intense.ipynb) | Detect which neurons encode which behavioral variables |\n"
"| **Population geometry** | [03 -- Dimensionality reduction](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/03_population_geometry_dr.ipynb) | Extract low-dimensional manifolds from population activity |\n"
"| **Network analysis** | [04 -- Networks](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/04_network_analysis.ipynb) | Build and analyze cell-cell interaction graphs |\n"
"| **Putting it together** | [05 -- Advanced](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/05_advanced_capabilities.ipynb) | Combine INTENSE + DR, leave-one-out importance, RSA, RNN analysis |\n"
"\n"
"This notebook focuses on individual neuron analysis: spike reconstruction,\n"
"kinetics optimization, and quality assessment.\n"
"\n"
"**What you will learn:**\n"
"\n"
"1. **Single neuron analysis** -- create a [`Neuron`](https://driada.readthedocs.io/en/latest/api/experiment/core.html#driada.experiment.neuron.Neuron), reconstruct spikes, optimize kinetics, compute quality metrics, and generate surrogates.\n"
"2. **Threshold vs wavelet reconstruction** -- compare two spike detection methods across four optimization modes.\n"
"3. **Method agreement** -- quantify event-region overlap between threshold and wavelet at varying tolerance."
))

cells.append(code_cell(
"# TODO: revert to '!pip install -q driada' after v1.0.0 PyPI release\n"
"!pip install -q git+https://github.com/iabs-neuro/driada.git@main\n"
"%matplotlib inline\n"
"\n"
"import warnings\n"
"\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"from matplotlib.patches import Patch\n"
"\n"
"from driada.experiment import generate_synthetic_exp\n"
"from driada.experiment.neuron import Neuron\n"
"from driada.experiment.synthetic import generate_pseudo_calcium_signal"
))

# ===== SECTION 1: QUICK SETUP ================================================

cells.append(md_cell(
"## 1. Setup\n"
"\n"
"This notebook covers **individual neuron analysis**: spike reconstruction,\n"
"kinetics optimization, and quality assessment.  For an introduction to\n"
"`Experiment` objects, feature types, and `TimeSeries`, see\n"
"[Notebook 00 -- DRIADA overview](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/00_driada_overview.ipynb).\n"
"\n"
"We work directly with\n"
"[`Neuron`](https://driada.readthedocs.io/en/latest/api/experiment/core.html#driada.experiment.neuron.Neuron)\n"
"objects and synthetic calcium traces throughout."
))

# ===== SECTION 2: SINGLE NEURON ANALYSIS ==================================

cells.append(md_cell(
"## 2. Single neuron analysis\n"
"\n"
"Deep dive into individual neuron quality: generate a synthetic calcium\n"
"signal with [`generate_pseudo_calcium_signal`](https://driada.readthedocs.io/en/latest/api/experiment/synthetic.html#driada.experiment.synthetic.core.generate_pseudo_calcium_signal),\n"
"create a [`Neuron`](https://driada.readthedocs.io/en/latest/api/experiment/core.html#driada.experiment.neuron.Neuron) object, reconstruct spikes, optimize kinetics,\n"
"compute quality metrics, and generate surrogates for null-hypothesis testing."
))

cells.append(code_cell(
"np.random.seed(42)\n"
"\n"
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
"wavelet transform) ridge analysis.  CWT ridge detection identifies\n"
"scale-persistent features in the wavelet scalogram -- ridges that persist\n"
"across multiple scales correspond to true transient events rather than noise."
))

cells.append(code_cell(
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
"### Wavelet scalogram\n"
"\n"
"The CWT scalogram shows how the signal's energy is distributed across\n"
"scales (frequencies) and time.  Calcium transient events appear as\n"
"bright cones extending from low to high scales.  Grey shading marks\n"
"the detected event regions."
))

cells.append(code_cell(
"from driada.experiment.wavelet_event_detection import (\n"
"    get_adaptive_wavelet_scales,\n"
")\n"
"from scipy.ndimage import gaussian_filter1d\n"
"\n"
"# Compute CWT scalogram using ssqueezepy (installed with driada)\n"
"from ssqueezepy import cwt, Wavelet as SqzWavelet\n"
"\n"
"fps_neuron = neuron.fps\n"
"scales = get_adaptive_wavelet_scales(fps_neuron)\n"
"sig_smooth = gaussian_filter1d(neuron.ca.data, sigma=int(0.4 * fps_neuron))\n"
"wavelet_obj = SqzWavelet(('gmw', {'beta': 2, 'gamma': 3}))\n"
"Wx, _ = cwt(sig_smooth, wavelet_obj, scales=scales)\n"
"\n"
"time_sec = np.arange(len(sig_smooth)) / fps_neuron\n"
"\n"
"fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,\n"
"                                gridspec_kw={'height_ratios': [1, 1]})\n"
"\n"
"# Top: scalogram\n"
"ax1.imshow(np.abs(Wx), aspect='auto', cmap='turbo',\n"
"           extent=[0, time_sec[-1], scales[-1], scales[0]])\n"
"ax1.set_ylabel('Scale')\n"
"ax1.set_title('CWT scalogram (Generalized Morse Wavelet)')\n"
"\n"
"# Bottom: signal + event regions\n"
"ax2.plot(time_sec, neuron.ca.data, 'b', linewidth=0.8)\n"
"if neuron.wvt_ridges:\n"
"    for ridge in neuron.wvt_ridges:\n"
"        t0 = ridge.start / fps_neuron\n"
"        t1 = ridge.end / fps_neuron\n"
"        ax2.axvspan(t0, t1, alpha=0.3, color='grey')\n"
"ax2.set_xlabel('Time (s)')\n"
"ax2.set_ylabel('dF/F0')\n"
"ax2.set_title(f'Calcium signal with {n_events} detected events')\n"
"\n"
"plt.tight_layout()\n"
"plt.show()"
))

cells.append(md_cell(
"### Kinetics optimization\n"
"\n"
"Fit rise and decay time constants to detected events using the **direct\n"
"measurement** method.  The `direct` method measures t_rise from the\n"
"derivative of the onset-to-peak waveform and t_off by fitting an\n"
"exponential to the peak-to-baseline decay, avoiding iterative optimization."
))

cells.append(code_cell(
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
'print("5. Computing wavelet SNR...")\n'
"\n"
"wavelet_snr = neuron.get_wavelet_snr()\n"
"\n"
'print(f"   [OK] Wavelet SNR: {wavelet_snr:.2f}")\n'
'print(f"       (Ratio of event amplitude to baseline noise)")\n'
"\n"
'print("\\n6. Computing reconstruction quality metrics...")\n'
"\n"
"r2 = neuron.get_reconstruction_r2()\n"
'print(f"   [OK] Reconstruction R2:  {r2:.4f}")\n'
"\n"
"r2_events = neuron.get_reconstruction_r2(event_only=True)\n"
'print(f"   [OK] Event-only R2:      {r2_events:.4f}")\n'
"\n"
"nrmse = neuron.get_nrmse()\n"
'print(f"   [OK] Normalized RMSE:    {nrmse:.4f}")\n'
"\n"
"nmae = neuron.get_nmae()\n"
'print(f"   [OK] Normalized MAE:     {nmae:.4f}")'
))

cells.append(md_cell(
"### Event and noise metrics\n"
"\n"
"Event-specific metrics assess quality only within detected transient\n"
"regions, which matters more than full-signal metrics when events are\n"
"sparse.  Noise estimates help set detection thresholds."
))

cells.append(code_cell(
"# Event-level metrics\n"
"event_count = neuron.get_event_count()\n"
"event_rmse = neuron.get_event_rmse()\n"
"event_mae = neuron.get_event_mae()\n"
"event_snr = neuron.get_event_snr()\n"
"\n"
'print(f"Event count:      {event_count}")\n'
'print(f"Event RMSE:       {event_rmse:.4f}")\n'
'print(f"Event MAE:        {event_mae:.4f}")\n'
'print(f"Event SNR:        {event_snr:.2f}")\n'
"\n"
"# Noise characterization\n"
"mad = neuron.get_mad()\n"
"baseline_std = neuron.get_baseline_noise_std()\n"
"\n"
'print(f"\\nMAD (robust noise):    {mad:.4f}")\n'
'print(f"Baseline noise std:    {baseline_std:.4f}")'
))

cells.append(md_cell(
"### Accessing the reconstruction\n"
"\n"
"The `reconstructed` property returns the cached model fit (calcium\n"
"kernel convolved with detected spikes).  Use `get_reconstructed()` to\n"
"recompute with custom kinetics."
))

cells.append(code_cell(
"# Cached reconstruction (uses optimized kinetics)\n"
"recon = neuron.reconstructed  # TimeSeries object\n"
'print(f"Reconstructed shape: {recon.data.shape}")\n'
'print(f"Reconstructed range: [{recon.data.min():.3f}, {recon.data.max():.3f}]")\n'
"\n"
"# Recompute with custom time constants (in frames)\n"
"recon_custom = neuron.get_reconstructed(t_rise_frames=3, t_off_frames=30)\n"
'print(f"Custom recon range:  [{recon_custom.data.min():.3f}, {recon_custom.data.max():.3f}]")'
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
'print("7. Surrogate generation methods...")\n'
"\n"
"shuffled_roll = neuron.get_shuffled_calcium(method='roll_based', seed=42)\n"
'print(f"\\n   [Roll-based] Circular shift surrogate:")\n'
'print(f"       Mean: {np.mean(shuffled_roll):.4f}  (original: {np.mean(neuron.ca.data):.4f})")\n'
'print(f"       Std:  {np.std(shuffled_roll):.4f}  (original: {np.std(neuron.ca.data):.4f})")\n'
"\n"
"shuffled_wf = neuron.get_shuffled_calcium(method='waveform_based', seed=42)\n"
'print(f"\\n   [Waveform-based] Spike-shuffle + reconstruct surrogate:")\n'
'print(f"       Mean: {np.mean(shuffled_wf):.4f}  (original: {np.mean(neuron.ca.data):.4f})")\n'
'print(f"       Std:  {np.std(shuffled_wf):.4f}  (original: {np.std(neuron.ca.data):.4f})")\n'
"\n"
"shuffled_chunks = neuron.get_shuffled_calcium(method='chunks_based', seed=42)\n"
'print(f"\\n   [Chunks-based] Chunk reordering surrogate:")\n'
'print(f"       Mean: {np.mean(shuffled_chunks):.4f}  (original: {np.mean(neuron.ca.data):.4f})")\n'
'print(f"       Std:  {np.std(shuffled_chunks):.4f}  (original: {np.std(neuron.ca.data):.4f})")\n'
"\n"
"shuffled_sp = neuron.get_shuffled_spikes(method='isi_based', seed=42)\n"
"original_spike_count = int(np.sum(neuron.sp.data > 0))\n"
"shuffled_spike_count = int(np.sum(shuffled_sp > 0))\n"
'print(f"\\n   [ISI-based] Spike train surrogate:")\n'
'print(f"       Spike count: {shuffled_spike_count}  (original: {original_spike_count})")'
))

# ===== SECTION 3: THRESHOLD VS WAVELET ====================================

cells.append(md_cell(
"## 3. Threshold vs wavelet reconstruction\n"
"\n"
"Above we analyzed one neuron with the wavelet method. DRIADA also supports\n"
"threshold-based detection. How do the two methods compare on the same signal?\n"
"\n"
"DRIADA supports two spike detection methods:\n"
"\n"
"| Method | Description |\n"
"|---|---|\n"
"| **Threshold** | MAD-based signal crossing -- fast, good for high SNR |\n"
"| **Wavelet** | CWT ridge detection -- more sensitive, better for low SNR or overlapping events |\n"
"\n"
"Both support iterative detection (detect-subtract-detect) to recover\n"
"weaker events hidden under larger transients."
))

cells.append(code_cell(
"# Generate a synthetic signal with non-default kinetics\n"
"fps_cmp = 30.0\n"
"signal_cmp = generate_pseudo_calcium_signal(\n"
"    duration=120.0, sampling_rate=fps_cmp, event_rate=0.3,  # higher rate for denser events\n"
"    amplitude_range=(0.3, 1.2), decay_time=0.8, rise_time=0.10,\n"
"    noise_std=0.04, kernel='double_exponential',\n"
")\n"
"time_cmp = np.arange(len(signal_cmp)) / fps_cmp\n"
"\n"
"# Wavelet: iterative detection with kinetics optimization\n"
"n_wvt = Neuron(cell_id='wavelet', ca=signal_cmp.copy(), sp=None, fps=fps_cmp)\n"
"with warnings.catch_warnings():\n"
"    warnings.simplefilter('ignore', UserWarning)\n"
"    n_wvt.reconstruct_spikes(method='wavelet', iterative=True, n_iter=3,\n"
"                              create_event_regions=True)\n"
"    n_wvt.optimize_kinetics(method='direct', fps=fps_cmp,\n"
"                             update_reconstruction=True, detection_method='wavelet')\n"
"\n"
"# Threshold: iterative detection with kinetics optimization\n"
"n_thr = Neuron(cell_id='threshold', ca=signal_cmp.copy(), sp=None, fps=fps_cmp)\n"
"with warnings.catch_warnings():\n"
"    warnings.simplefilter('ignore', UserWarning)\n"
"    n_thr.reconstruct_spikes(method='threshold', iterative=True, n_iter=3,\n"
"                              n_mad=4.0, create_event_regions=True,  # n_mad: noise multiplier for threshold\n"
"                              adaptive_thresholds=True)\n"
"    n_thr.optimize_kinetics(method='direct', fps=fps_cmp,\n"
"                             update_reconstruction=True, detection_method='threshold',\n"
"                             n_mad=4.0, iterative=True, n_iter=3,\n"
"                             adaptive_thresholds=True)\n"
"\n"
"wvt_events = len(n_wvt.wvt_ridges) if n_wvt.wvt_ridges else 0\n"
"thr_events = len(n_thr.threshold_events) if n_thr.threshold_events else 0\n"
"\n"
'print(f"Wavelet:   {wvt_events} events, R2={n_wvt.get_reconstruction_r2():.4f}")\n'
'print(f"Threshold: {thr_events} events, R2={n_thr.get_reconstruction_r2():.4f}")'
))

cells.append(code_cell(
"fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)\n"
"\n"
"ax = axes[0]\n"
"ax.plot(time_cmp, signal_cmp, 'k', linewidth=0.8)\n"
"ax.set_ylabel('dF/F0')\n"
"ax.set_title('Original calcium signal')\n"
"ax.grid(True, alpha=0.3)\n"
"\n"
"ax = axes[1]\n"
"ax.plot(time_cmp, signal_cmp, 'k', linewidth=0.5, alpha=0.4)\n"
"ax.plot(time_cmp, n_wvt._reconstructed.data, 'b', linewidth=1.2,\n"
"        label=f'Wavelet (R2={n_wvt.get_reconstruction_r2():.3f})')\n"
"if n_wvt.wvt_ridges:\n"
"    for r in n_wvt.wvt_ridges:\n"
"        ax.axvspan(r.start / fps_cmp, r.end / fps_cmp, alpha=0.15, color='blue')\n"
"ax.set_ylabel('dF/F0')\n"
"ax.legend(loc='upper right', fontsize=9)\n"
"ax.grid(True, alpha=0.3)\n"
"\n"
"ax = axes[2]\n"
"ax.plot(time_cmp, signal_cmp, 'k', linewidth=0.5, alpha=0.4)\n"
"ax.plot(time_cmp, n_thr._reconstructed.data, 'r', linewidth=1.2,\n"
"        label=f'Threshold (R2={n_thr.get_reconstruction_r2():.3f})')\n"
"if n_thr.threshold_events:\n"
"    for ev in n_thr.threshold_events:\n"
"        ax.axvspan(ev.start / fps_cmp, ev.end / fps_cmp, alpha=0.15, color='red')\n"
"ax.set_ylabel('dF/F0')\n"
"ax.set_xlabel('Time (s)')\n"
"ax.legend(loc='upper right', fontsize=9)\n"
"ax.grid(True, alpha=0.3)\n"
"\n"
"plt.tight_layout()\n"
"plt.show()"
))

# ===== SECTION 4: METHOD AGREEMENT ========================================

cells.append(md_cell(
"## 4. Method agreement\n"
"\n"
"Visual comparison shows broad agreement but also differences in event\n"
"boundaries. Let's quantify this overlap systematically.\n"
"\n"
"Given the same data, how well do threshold and wavelet agree?  We use\n"
"[`generate_synthetic_exp`](https://driada.readthedocs.io/en/latest/api/experiment/synthetic.html#driada.experiment.generate_synthetic_exp)\n"
"to create a small population, then run both methods and compare\n"
"event regions."
))

cells.append(code_cell(
'print("Generating synthetic experiment...")\n'
'exp4 = generate_synthetic_exp(\n'
'    n_dfeats=2, n_cfeats=1, nneurons=5, duration=120, fps=20, seed=42\n'
')\n'
'\n'
'fps4 = exp4.fps\n'
'n_neurons4 = exp4.calcium.scdata.shape[0]\n'
'time4 = np.arange(exp4.calcium.scdata.shape[1]) / fps4\n'
'\n'
'wavelet_events = []\n'
'threshold_events = []\n'
'\n'
'for n4 in exp4.neurons:\n'
'    n4.reconstruct_spikes(method="wavelet", iterative=True, n_iter=3, fps=fps4)\n'
'    wavelet_events.append(list(n4.wvt_ridges))\n'
'    n4.reconstruct_spikes(method="threshold", iterative=True, n_iter=3,\n'
'                           n_mad=4.0, adaptive_thresholds=True, fps=fps4)  # lower n_mad catches more events\n'
'    threshold_events.append(list(n4.threshold_events))\n'
'\n'
'for i in range(n_neurons4):\n'
'    print(f"  Neuron {i}: wavelet={len(wavelet_events[i])}, "\n'
'          f"threshold={len(threshold_events[i])}")'
))

cells.append(code_cell(
"# Event region comparison for one neuron\n"
"neuron_idx = 2\n"
"\n"
"fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)\n"
"\n"
"ax = axes[0]\n"
'ax.plot(time4, exp4.calcium.scdata[neuron_idx, :], "k-", linewidth=1)\n'
'ax.set_ylabel("Calcium")\n'
'ax.set_title(f"Neuron {neuron_idx}: event region comparison")\n'
"ax.grid(True, alpha=0.3)\n"
"\n"
"ax = axes[1]\n"
"for ev in wavelet_events[neuron_idx]:\n"
'    ax.axvspan(ev.start / fps4, ev.end / fps4, alpha=0.5, color="blue")\n'
'ax.set_ylabel("Wavelet")\n'
"ax.set_ylim(-0.1, 1.1)\n"
"ax.grid(True, alpha=0.3)\n"
"ax.legend(\n"
'    handles=[Patch(facecolor="blue", alpha=0.5, label="Event region")],\n'
'    loc="upper right",\n'
")\n"
"\n"
"ax = axes[2]\n"
"for ev in threshold_events[neuron_idx]:\n"
'    ax.axvspan(ev.start / fps4, ev.end / fps4, alpha=0.5, color="red")\n'
'ax.set_ylabel("Threshold")\n'
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
    "01_data_loading_and_neurons.ipynb",
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c.cell_type == 'code')} code)")
