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
"[**DRIADA**](https://driada.readthedocs.io) (Dimensionality Reduction for\n"
"Integrated Activity Data) is a Python framework for neural data analysis.\n"
"It bridges two perspectives that are usually treated separately: what\n"
"*individual* neurons encode, and how the *population as a whole* represents\n"
"information.  The typical analysis workflow looks like this:\n"
"\n"
"| Step | Notebook | What it does |\n"
"|---|---|---|\n"
"| **Overview** | [00 -- DRIADA overview](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/00_driada_overview.ipynb) | Core data structures, quick tour of INTENSE, DR, networks |\n"
"| **Load & inspect** | **01 -- this notebook** | Wrap your recording into an `Experiment`, reconstruct spikes, assess quality |\n"
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
"from driada.experiment.synthetic import (\n"
"    generate_pseudo_calcium_signal,\n"
"    generate_pseudo_calcium_multisignal,\n"
")"
))

# ===== SECTION 1: QUICK SETUP ================================================

cells.append(md_cell(
"## 1. Setup\n"
"\n"
"This notebook covers spike reconstruction and neuron quality analysis.\n"
"For an introduction to `Experiment` objects, feature types, and\n"
"TimeSeries, see\n"
"[Notebook 00 -- DRIADA overview](https://colab.research.google.com/github/iabs-neuro/driada/blob/main/notebooks/00_driada_overview.ipynb).\n"
"\n"
"We start by creating a synthetic Experiment with pseudo-calcium traces."
))

cells.append(code_cell(
"n_neurons = 20\n"
"fps = 30.0\n"
"duration = 200.0  # seconds\n"
"\n"
"calcium = generate_pseudo_calcium_multisignal(\n"
"    n=n_neurons,\n"
"    duration=duration,\n"
"    sampling_rate=fps,\n"
"    event_rate=0.15,\n"
"    amplitude_range=(0.5, 2.0),\n"
"    decay_time=1.5,\n"
"    rise_time=0.15,\n"
"    noise_std=0.05,\n"
"    kernel='double_exponential',\n"
"    seed=0,\n"
")\n"
"n_frames = calcium.shape[1]\n"
"\n"
"np.random.seed(0)\n"
"speed = np.abs(np.random.randn(n_frames)) * 5.0\n"
"head_direction = np.random.uniform(0, 2 * np.pi, n_frames)\n"
"trial_type = np.random.choice([0, 1, 2], size=n_frames)\n"
"\n"
"data = {\n"
'    "calcium": calcium,\n'
'    "speed": speed,\n'
'    "head_direction": head_direction,\n'
'    "trial_type": trial_type,\n'
"}\n"
"\n"
"exp = load_exp_from_aligned_data(\n"
'    data_source="MyLab",\n'
'    exp_params={"name": "demo_recording"},\n'
"    data=data,\n"
'    feature_types={"head_direction": "circular"},\n'
'    static_features={"fps": fps},\n'
"    verbose=True,\n"
")\n"
"\n"
'print(f"Experiment: {exp.n_cells} neurons, {exp.n_frames} frames")'
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
"Given the same data, how well do threshold and wavelet agree?  We use\n"
"[`generate_synthetic_exp`](https://driada.readthedocs.io/en/latest/api/experiment/synthetic.html#driada.experiment.generate_synthetic_exp)\n"
"to create a small population, then run both methods.  Both\n"
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
'wavelet_events = []\n'
'threshold_events = []\n'
'\n'
'for neuron in exp4.neurons:\n'
'    print(f"  Neuron {neuron.cell_id}: wavelet...", end="")\n'
'    neuron.reconstruct_spikes(\n'
'        method="wavelet", iterative=True, n_iter=3, fps=fps4\n'
'    )\n'
'    wavelet_events.append(list(neuron.wvt_ridges))\n'
'\n'
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
