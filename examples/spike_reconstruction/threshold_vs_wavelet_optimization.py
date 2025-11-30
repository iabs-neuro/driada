"""
Threshold vs Wavelet Reconstruction Comparison
===============================================

This example demonstrates different reconstruction modes:
1. Default kinetics - fast, uses preset rise/decay times
2. Optimized kinetics - fits kinetics to your signal for better accuracy
3. Iterative detection (n_iter=2) - detect -> subtract -> detect more
4. Iterative detection (n_iter=3) - more passes for weaker events

Key Workflow:
-------------
1. reconstruct_spikes() - Detect events
2. optimize_kinetics(update_reconstruction=True) - Optional: improve kinetics

Key Takeaways:
- Threshold: much faster, good for high SNR data
- Wavelet: More sensitive, better for low SNR or overlapping events
- Both methods support iterative detection (detect -> subtract -> repeat)
- Kinetics optimization improves reconstruction quality
"""

import numpy as np
import matplotlib.pyplot as plt
from driada.experiment.neuron import Neuron
from driada.utils.neural import generate_pseudo_calcium_signal
import time


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
        amplitude_range=(0.3, 1.2),   # Lower peaks (realistic)
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
    import warnings

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
    if optimize:
        neuron.optimize_kinetics(
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

    time_total = time.time() - start

    # Get kinetics
    t_rise = neuron.t_rise if neuron.t_rise else neuron.default_t_rise
    t_off = neuron.t_off if neuron.t_off else neuron.default_t_off

    # Get reconstruction
    recon = Neuron.get_restored_calcium(
        neuron.asp.data if neuron.asp else neuron.sp.data,
        t_rise,
        t_off
    )

    # Count events
    if method == 'threshold':
        n_events = len(neuron.threshold_events) if neuron.threshold_events else 0
    else:
        n_events = len(neuron.wvt_ridges) if neuron.wvt_ridges else 0

    return {
        'mode': mode_name,
        'reconstruction': recon[:neuron.n_frames],
        't_rise': t_rise / fps,
        't_off': t_off / fps,
        'n_events': n_events,
        'time': time_total
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

    return results


def calculate_metrics(true_signal, reconstruction):
    """Calculate multiple quality metrics."""
    residuals = true_signal - reconstruction

    # R² (coefficient of determination)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((true_signal - np.mean(true_signal)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(residuals ** 2))

    # Normalized RMSE
    signal_range = np.max(true_signal) - np.min(true_signal)
    nrmse = rmse / signal_range if signal_range > 0 else 0

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(residuals))

    # Correlation coefficient
    correlation = np.corrcoef(true_signal, reconstruction)[0, 1]

    # Explained variance
    var_true = np.var(true_signal)
    var_residual = np.var(residuals)
    explained_var = 1 - (var_residual / var_true) if var_true > 0 else 0

    return {
        'r2': r2,
        'rmse': rmse,
        'nrmse': nrmse,
        'mae': mae,
        'correlation': correlation,
        'explained_variance': explained_var
    }


def main():
    print("=" * 80)
    print("THRESHOLD VS WAVELET RECONSTRUCTION WITH ITERATIVE OPTIMIZATION")
    print("=" * 80)

    # Generate synthetic data
    print("\n1. Generating synthetic calcium signal...")
    fps = 30.0
    signal, ground_truth = create_synthetic_neuron(duration=300.0, fps=fps, seed=42)  # 5 minutes
    print(f"   Signal: {len(signal)} frames ({len(signal)/fps:.1f} seconds)")
    print(f"   Ground truth: t_rise={ground_truth['t_rise_true']:.3f}s, "
          f"t_off={ground_truth['t_off_true']:.3f}s")

    # Threshold reconstruction - all 4 modes
    print("\n2. Threshold-based reconstruction (FAST)...")
    threshold_results = reconstruct_all_modes(signal, fps, method='threshold')

    # Wavelet reconstruction - all 4 modes
    print("\n3. Wavelet-based reconstruction (SENSITIVE)...")
    wavelet_results = reconstruct_all_modes(signal, fps, method='wavelet')

    # Compute quality metrics
    print("\n4. Computing reconstruction quality metrics...")
    time_axis = np.arange(len(signal)) / fps

    for results, method in [(threshold_results, 'Threshold'), (wavelet_results, 'Wavelet')]:
        for res in results:
            metrics = calculate_metrics(signal, res['reconstruction'])
            res.update(metrics)

    # Print summary table
    print("\n" + "=" * 120)
    print("RECONSTRUCTION QUALITY SUMMARY")
    print("=" * 120)
    print(f"{'Method':<12} {'Mode':<22} {'Events':<8} {'t_rise(s)':<11} {'t_off(s)':<11} "
          f"{'R^2':<8} {'NRMSE':<8} {'Corr':<8} {'Time(s)':<10}")
    print("-" * 120)

    for res in threshold_results:
        print(f"{'Threshold':<12} {res['mode']:<22} {res['n_events']:<8} "
              f"{res['t_rise']:<11.3f} {res['t_off']:<11.3f} "
              f"{res['r2']:<8.4f} {res['nrmse']:<8.4f} {res['correlation']:<8.4f} "
              f"{res['time']:<10.4f}")

    print("-" * 120)

    for res in wavelet_results:
        print(f"{'Wavelet':<12} {res['mode']:<22} {res['n_events']:<8} "
              f"{res['t_rise']:<11.3f} {res['t_off']:<11.3f} "
              f"{res['r2']:<8.4f} {res['nrmse']:<8.4f} {res['correlation']:<8.4f} "
              f"{res['time']:<10.4f}")

    # Calculate speedup
    threshold_total_time = sum(r['time'] for r in threshold_results)
    wavelet_total_time = sum(r['time'] for r in wavelet_results)
    speedup = wavelet_total_time / threshold_total_time

    print("\n" + "=" * 80)
    print(f"PERFORMANCE: Threshold is {speedup:.1f}x faster than Wavelet")
    print(f"  Threshold total: {threshold_total_time:.3f}s")
    print(f"  Wavelet total:   {wavelet_total_time:.3f}s")
    print("=" * 80)

    # Visualize results
    print("\n5. Creating visualization...")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)

    # Column 1: Threshold method
    # Original signal
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(time_axis, signal, 'k-', linewidth=1, label='Calcium signal')
    ax0.set_ylabel('dF/F0')
    ax0.set_title('THRESHOLD METHOD (Fast)', fontweight='bold', fontsize=12)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='upper right')

    # Reconstruction modes for threshold
    for i, res in enumerate(threshold_results):
        ax = fig.add_subplot(gs[i+1, 0])
        ax.plot(time_axis, signal, 'k-', linewidth=0.8, alpha=0.5, label='Original')
        ax.plot(time_axis, res['reconstruction'], 'b-', linewidth=1.2,
                label=f"Reconstruction (R^2={res['r2']:.3f})")

        ax.set_title(f"{res['mode']} | t_rise={res['t_rise']:.3f}s, t_off={res['t_off']:.3f}s",
                    fontsize=10)
        ax.set_ylabel('dF/F0')
        if i == 3:
            ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    # Column 2: Wavelet method
    # Original signal
    ax0 = fig.add_subplot(gs[0, 1])
    ax0.plot(time_axis, signal, 'k-', linewidth=1, label='Calcium signal')
    ax0.set_ylabel('dF/F0')
    ax0.set_title('WAVELET METHOD (Sensitive)', fontweight='bold', fontsize=12)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='upper right')

    # Reconstruction modes for wavelet
    for i, res in enumerate(wavelet_results):
        ax = fig.add_subplot(gs[i+1, 1])
        ax.plot(time_axis, signal, 'k-', linewidth=0.8, alpha=0.5, label='Original')
        ax.plot(time_axis, res['reconstruction'], 'r-', linewidth=1.2,
                label=f"Reconstruction (R^2={res['r2']:.3f})")

        ax.set_title(f"{res['mode']} | t_rise={res['t_rise']:.3f}s, t_off={res['t_off']:.3f}s",
                    fontsize=10)
        ax.set_ylabel('dF/F0')
        if i == 3:
            ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(
        f'Spike Reconstruction Comparison | Ground truth: t_rise={ground_truth["t_rise_true"]:.3f}s, '
        f't_off={ground_truth["t_off_true"]:.3f}s | Speedup: {speedup:.1f}x',
        fontsize=14, fontweight='bold', y=0.995
    )

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'threshold_vs_wavelet_optimization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Figure saved: {output_file}")

    # Create summary plot with more metrics
    fig2, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Prepare mode labels
    modes = range(len(threshold_results))
    mode_labels = ['Default', 'Optimized', 'Iter n=2', 'Iter n=3']

    # R^2 comparison
    ax = axes[0, 0]
    threshold_r2 = [r['r2'] for r in threshold_results]
    wavelet_r2 = [r['r2'] for r in wavelet_results]
    ax.plot(modes, threshold_r2, 'bo-', label='Threshold', linewidth=2, markersize=8)
    ax.plot(modes, wavelet_r2, 'ro-', label='Wavelet', linewidth=2, markersize=8)
    ax.set_xlabel('Mode')
    ax.set_ylabel('R^2')
    ax.set_title('Reconstruction Quality (R^2)', fontweight='bold')
    ax.set_xticks(list(modes))
    ax.set_xticklabels(mode_labels, rotation=15)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Correlation comparison
    ax = axes[0, 1]
    threshold_corr = [r['correlation'] for r in threshold_results]
    wavelet_corr = [r['correlation'] for r in wavelet_results]
    ax.plot(modes, threshold_corr, 'bo-', label='Threshold', linewidth=2, markersize=8)
    ax.plot(modes, wavelet_corr, 'ro-', label='Wavelet', linewidth=2, markersize=8)
    ax.set_xlabel('Mode')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation Coefficient', fontweight='bold')
    ax.set_xticks(list(modes))
    ax.set_xticklabels(mode_labels, rotation=15)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # NRMSE comparison
    ax = axes[1, 0]
    threshold_nrmse = [r['nrmse'] for r in threshold_results]
    wavelet_nrmse = [r['nrmse'] for r in wavelet_results]
    ax.plot(modes, threshold_nrmse, 'bo-', label='Threshold', linewidth=2, markersize=8)
    ax.plot(modes, wavelet_nrmse, 'ro-', label='Wavelet', linewidth=2, markersize=8)
    ax.set_xlabel('Mode')
    ax.set_ylabel('NRMSE')
    ax.set_title('Normalized RMSE (lower is better)', fontweight='bold')
    ax.set_xticks(list(modes))
    ax.set_xticklabels(mode_labels, rotation=15)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Event count comparison
    ax = axes[1, 1]
    threshold_events = [r['n_events'] for r in threshold_results]
    wavelet_events = [r['n_events'] for r in wavelet_results]
    ax.plot(modes, threshold_events, 'bo-', label='Threshold', linewidth=2, markersize=8)
    ax.plot(modes, wavelet_events, 'ro-', label='Wavelet', linewidth=2, markersize=8)
    ax.set_xlabel('Mode')
    ax.set_ylabel('Events detected')
    ax.set_title('Number of Events Detected', fontweight='bold')
    ax.set_xticks(list(modes))
    ax.set_xticklabels(mode_labels, rotation=15)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # t_rise comparison
    ax = axes[2, 0]
    threshold_trise = [r['t_rise'] for r in threshold_results]
    wavelet_trise = [r['t_rise'] for r in wavelet_results]
    ax.plot(modes, threshold_trise, 'bo-', label='Threshold', linewidth=2, markersize=8)
    ax.plot(modes, wavelet_trise, 'ro-', label='Wavelet', linewidth=2, markersize=8)
    ax.axhline(ground_truth['t_rise_true'], color='g', linestyle='--', linewidth=2, label='Ground truth')
    ax.set_xlabel('Mode')
    ax.set_ylabel('t_rise (s)')
    ax.set_title('Rise Time', fontweight='bold')
    ax.set_xticks(list(modes))
    ax.set_xticklabels(mode_labels, rotation=15)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # t_off comparison
    ax = axes[2, 1]
    threshold_toff = [r['t_off'] for r in threshold_results]
    wavelet_toff = [r['t_off'] for r in wavelet_results]
    ax.plot(modes, threshold_toff, 'bo-', label='Threshold', linewidth=2, markersize=8)
    ax.plot(modes, wavelet_toff, 'ro-', label='Wavelet', linewidth=2, markersize=8)
    ax.axhline(ground_truth['t_off_true'], color='g', linestyle='--', linewidth=2, label='Ground truth')
    ax.set_xlabel('Mode')
    ax.set_ylabel('t_off (s)')
    ax.set_title('Decay Time', fontweight='bold')
    ax.set_xticks(list(modes))
    ax.set_xticklabels(mode_labels, rotation=15)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle(f'Reconstruction Metrics by Mode | Speedup: {speedup:.1f}x',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_file2 = os.path.join(script_dir, 'threshold_vs_wavelet_convergence.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"   Figure saved: {output_file2}")

    print("\n" + "=" * 120)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 120)
    print("\nKey Findings:")
    print(f"  1. Threshold method is {speedup:.1f}x faster than wavelet")
    print(f"  2. Best mode comparison (Iterative n=3 + optimized):")
    print(f"     - Threshold: R^2={threshold_results[-1]['r2']:.4f}, Events={threshold_results[-1]['n_events']}")
    print(f"     - Wavelet:   R^2={wavelet_results[-1]['r2']:.4f}, Events={wavelet_results[-1]['n_events']}")
    print(f"  3. Kinetics estimation (ground truth: t_rise={ground_truth['t_rise_true']:.3f}s, t_off={ground_truth['t_off_true']:.3f}s):")
    print(f"     - Threshold: t_rise={threshold_results[-1]['t_rise']:.3f}s, t_off={threshold_results[-1]['t_off']:.3f}s")
    print(f"     - Wavelet:   t_rise={wavelet_results[-1]['t_rise']:.3f}s, t_off={wavelet_results[-1]['t_off']:.3f}s")
    print(f"  4. Iterative detection finds more events:")
    print(f"     - Threshold: {threshold_results[0]['n_events']} (default) -> {threshold_results[-1]['n_events']} (iter n=3)")
    print(f"     - Wavelet:   {wavelet_results[0]['n_events']} (default) -> {wavelet_results[-1]['n_events']} (iter n=3)")
    print(f"  5. Use threshold for fast processing, wavelet for best sensitivity")

    # plt.show()  # Commented out for non-interactive use


if __name__ == "__main__":
    main()
