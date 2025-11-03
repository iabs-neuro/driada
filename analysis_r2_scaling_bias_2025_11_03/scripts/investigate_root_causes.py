"""Investigate root causes of reconstruction errors.

This script analyzes:
1. NNLS amplitude estimation accuracy
2. Event detection accuracy (false positives/negatives)
3. Timing precision errors
4. Kernel normalization effects
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import numpy as np
import matplotlib.pyplot as plt

from driada.experiment.neuron import Neuron
from driada.experiment.synthetic import generate_synthetic_exp


def analyze_event_detection(neuron):
    """Analyze event detection accuracy against ground truth"""

    # Get ground truth spikes
    true_spikes = neuron.sp.data if hasattr(neuron, 'sp') and neuron.sp is not None else None
    if true_spikes is None:
        return None

    # Get reconstructed spikes
    rec_spikes = neuron.asp.data

    # Find event locations
    true_events = np.where(true_spikes > 0)[0]
    rec_events = np.where(rec_spikes > 0)[0]

    # Calculate detection metrics with tolerance window
    tolerance = 2  # frames (100ms at 20Hz)

    # True positives: detected events within tolerance of true events
    tp = 0
    matched_true = set()
    matched_rec = set()

    for rec_idx in rec_events:
        for true_idx in true_events:
            if abs(rec_idx - true_idx) <= tolerance and true_idx not in matched_true:
                tp += 1
                matched_true.add(true_idx)
                matched_rec.add(rec_idx)
                break

    # False negatives: true events not matched
    fn = len(true_events) - len(matched_true)

    # False positives: detected events not matched
    fp = len(rec_events) - len(matched_rec)

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'n_true': len(true_events),
        'n_detected': len(rec_events),
        'true_positive': tp,
        'false_negative': fn,
        'false_positive': fp,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def analyze_amplitude_accuracy(neuron):
    """Analyze amplitude estimation accuracy"""

    # Get ground truth and reconstructed spikes
    true_spikes = neuron.sp.data if hasattr(neuron, 'sp') and neuron.sp is not None else None
    if true_spikes is None:
        return None

    rec_spikes = neuron.asp.data

    # Find matched events
    true_events = np.where(true_spikes > 0)[0]
    rec_events = np.where(rec_spikes > 0)[0]

    tolerance = 2
    matched_pairs = []

    for rec_idx in rec_events:
        for true_idx in true_events:
            if abs(rec_idx - true_idx) <= tolerance:
                matched_pairs.append((true_idx, rec_idx))
                break

    if len(matched_pairs) == 0:
        return None

    # Compare amplitudes for matched events
    true_amps = [true_spikes[t] for t, r in matched_pairs]
    rec_amps = [rec_spikes[r] for t, r in matched_pairs]

    true_amps = np.array(true_amps)
    rec_amps = np.array(rec_amps)

    # Calculate metrics
    amp_ratio = rec_amps / true_amps
    mean_ratio = np.mean(amp_ratio)
    std_ratio = np.std(amp_ratio)

    # Amplitude error
    amp_error = rec_amps - true_amps
    mae = np.mean(np.abs(amp_error))
    rmse = np.sqrt(np.mean(amp_error ** 2))

    return {
        'n_matched': len(matched_pairs),
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'mae': mae,
        'rmse': rmse,
        'true_amps_mean': np.mean(true_amps),
        'rec_amps_mean': np.mean(rec_amps),
    }


def analyze_timing_accuracy(neuron):
    """Analyze timing precision of detected events"""

    # Get ground truth and reconstructed spikes
    true_spikes = neuron.sp.data if hasattr(neuron, 'sp') and neuron.sp is not None else None
    if true_spikes is None:
        return None

    rec_spikes = neuron.asp.data

    # Find matched events
    true_events = np.where(true_spikes > 0)[0]
    rec_events = np.where(rec_spikes > 0)[0]

    tolerance = 5  # larger tolerance for timing analysis
    timing_errors = []

    for rec_idx in rec_events:
        best_match = None
        best_error = tolerance + 1

        for true_idx in true_events:
            error = abs(rec_idx - true_idx)
            if error <= tolerance and error < best_error:
                best_match = true_idx
                best_error = error

        if best_match is not None:
            timing_errors.append(rec_idx - best_match)

    if len(timing_errors) == 0:
        return None

    timing_errors = np.array(timing_errors)

    return {
        'n_matched': len(timing_errors),
        'mean_error': np.mean(timing_errors),
        'std_error': np.std(timing_errors),
        'mae': np.mean(np.abs(timing_errors)),
        'max_error': np.max(np.abs(timing_errors)),
    }


def main():
    # Generate synthetic experiment
    print("Generating synthetic experiment...")
    exp = generate_synthetic_exp(
        n_dfeats=1,
        n_cfeats=0,
        nneurons=10,
        duration=200,
        fps=20,
        seed=42,
        with_spikes=True
    )

    print(f"Analyzing {len(exp.neurons)} neurons\n")

    detection_results = []
    amplitude_results = []
    timing_results = []

    for i, neuron in enumerate(exp.neurons):
        # Run reconstruction
        if neuron.asp is None:
            neuron.reconstruct_spikes()

        # Analyze
        det = analyze_event_detection(neuron)
        amp = analyze_amplitude_accuracy(neuron)
        tim = analyze_timing_accuracy(neuron)

        if det:
            detection_results.append(det)
        if amp:
            amplitude_results.append(amp)
        if tim:
            timing_results.append(tim)

        print(f"Neuron {i}:")
        print(f"  Detection: {det['f1']:.3f} F1 (P={det['precision']:.3f}, R={det['recall']:.3f})")
        if amp:
            print(f"  Amplitude: {amp['mean_ratio']:.3f}±{amp['std_ratio']:.3f} ratio (MAE={amp['mae']:.4f})")
        if tim:
            print(f"  Timing: {tim['mean_error']:.2f}±{tim['std_error']:.2f} frames (MAE={tim['mae']:.2f})")
        print()

    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Detection
    print("\nEvent Detection:")
    print(f"  Mean Precision: {np.mean([r['precision'] for r in detection_results]):.3f}")
    print(f"  Mean Recall:    {np.mean([r['recall'] for r in detection_results]):.3f}")
    print(f"  Mean F1:        {np.mean([r['f1'] for r in detection_results]):.3f}")
    print(f"  False Positive Rate: {np.mean([r['false_positive'] / r['n_detected'] if r['n_detected'] > 0 else 0 for r in detection_results]):.3f}")
    print(f"  False Negative Rate: {np.mean([r['false_negative'] / r['n_true'] if r['n_true'] > 0 else 0 for r in detection_results]):.3f}")

    # Amplitude
    print("\nAmplitude Estimation:")
    print(f"  Mean Ratio: {np.mean([r['mean_ratio'] for r in amplitude_results]):.3f}±{np.mean([r['std_ratio'] for r in amplitude_results]):.3f}")
    print(f"  Mean MAE:   {np.mean([r['mae'] for r in amplitude_results]):.4f}")
    print(f"  Mean RMSE:  {np.mean([r['rmse'] for r in amplitude_results]):.4f}")

    # Timing
    print("\nTiming Precision:")
    print(f"  Mean Error: {np.mean([r['mean_error'] for r in timing_results]):.2f}±{np.mean([r['std_error'] for r in timing_results]):.2f} frames")
    print(f"  Mean MAE:   {np.mean([r['mae'] for r in timing_results]):.2f} frames ({np.mean([r['mae'] for r in timing_results]) / 20 * 1000:.1f} ms at 20Hz)")

    # Save results
    save_dir = Path(__file__).parent.parent / 'data'
    save_dir.mkdir(exist_ok=True)

    # Detection metrics
    with open(save_dir / 'detection_metrics.csv', 'w') as f:
        f.write("neuron,n_true,n_detected,tp,fp,fn,precision,recall,f1\n")
        for i, r in enumerate(detection_results):
            f.write(f"{i},{r['n_true']},{r['n_detected']},{r['true_positive']},"
                   f"{r['false_positive']},{r['false_negative']},{r['precision']:.4f},"
                   f"{r['recall']:.4f},{r['f1']:.4f}\n")

    # Amplitude metrics
    with open(save_dir / 'amplitude_metrics.csv', 'w') as f:
        f.write("neuron,n_matched,mean_ratio,std_ratio,mae,rmse,true_mean,rec_mean\n")
        for i, r in enumerate(amplitude_results):
            f.write(f"{i},{r['n_matched']},{r['mean_ratio']:.4f},{r['std_ratio']:.4f},"
                   f"{r['mae']:.4f},{r['rmse']:.4f},{r['true_amps_mean']:.4f},"
                   f"{r['rec_amps_mean']:.4f}\n")

    # Timing metrics
    with open(save_dir / 'timing_metrics.csv', 'w') as f:
        f.write("neuron,n_matched,mean_error,std_error,mae,max_error\n")
        for i, r in enumerate(timing_results):
            f.write(f"{i},{r['n_matched']},{r['mean_error']:.4f},{r['std_error']:.4f},"
                   f"{r['mae']:.4f},{r['max_error']:.4f}\n")

    print("\nResults saved to data/")
    print("  - detection_metrics.csv")
    print("  - amplitude_metrics.csv")
    print("  - timing_metrics.csv")


if __name__ == '__main__':
    main()
