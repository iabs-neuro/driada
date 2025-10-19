#!/usr/bin/env python
"""
Analyze firing patterns of place cells (manifold neurons) in mixed population experiments.

This script addresses the key question: Do place cells fire frequently enough
in their place fields to be statistically distinguishable from random neurons?

Key analyses:
1. Firing frequency above baseline in place fields
2. Fraction of time active in place fields
3. Number of distinct place field visits
4. Signal-to-noise ratio comparison
5. Distinguishability from feature neurons
6. Spatial trajectory movement patterns
"""

import sys
import os
sys.path.insert(0, '/Users/nikita/PycharmProjects/driada2/src')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, classification_report
import logging
from typing import Dict, Tuple, List, Optional

from driada.experiment.synthetic import generate_mixed_population_exp
from driada.experiment.synthetic.manifold_spatial_2d import gaussian_place_field
from driada.utils.data import check_positive


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlaceCellFiringAnalyzer:
    """
    Production-grade analyzer for place cell firing patterns and sparsity.

    This class provides comprehensive analysis of place cell firing patterns
    to determine if they are distinguishable from noise or random fluctuations.
    """

    def __init__(self,
                 exp,
                 info: Dict,
                 baseline_threshold: float = 2.0,
                 field_visit_threshold: float = 0.5,
                 min_visit_duration: int = 5,
                 verbose: bool = True):
        """
        Initialize the analyzer with experiment data.

        Parameters
        ----------
        exp : Experiment
            DRIADA experiment object containing neural data
        info : dict
            Information dictionary from experiment generation
        baseline_threshold : float, optional
            Threshold in standard deviations above baseline for activity detection
        field_visit_threshold : float, optional
            Minimum place field response for considering a "visit"
        min_visit_duration : int, optional
            Minimum duration (in time points) for a valid place field visit
        verbose : bool, optional
            Whether to print progress messages
        """
        self.exp = exp
        self.info = info
        self.baseline_threshold = baseline_threshold
        self.field_visit_threshold = field_visit_threshold
        self.min_visit_duration = min_visit_duration
        self.verbose = verbose

        # Validate inputs
        check_positive(baseline_threshold=baseline_threshold,
                      field_visit_threshold=field_visit_threshold,
                      min_visit_duration=min_visit_duration)

        # Extract basic information
        self.n_manifold = info['population_composition']['n_manifold']
        self.n_feature = info['population_composition']['n_feature_selective']
        self.spatial_data = info.get('spatial_data')
        self.manifold_info = info.get('manifold_info', {})

        # Get place field information
        self.place_field_centers = self.manifold_info.get('place_field_centers')
        self.field_sigma = self.exp.static_features.get('field_sigma', 0.1)

        # Initialize results storage
        self.manifold_metrics = {}
        self.feature_metrics = {}
        self.analysis_results = {}

        if self.verbose:
            logger.info(f"Initialized analyzer for {self.n_manifold} manifold neurons, "
                       f"{self.n_feature} feature neurons")

    def analyze_neuron_firing_pattern(self,
                                    neuron_id: int,
                                    neuron_signal: np.ndarray,
                                    place_field_center: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze firing pattern for a single neuron.

        Parameters
        ----------
        neuron_id : int
            Neuron identifier
        neuron_signal : ndarray
            Calcium signal for the neuron
        place_field_center : ndarray, optional
            Center coordinates of place field (for manifold neurons)

        Returns
        -------
        dict
            Dictionary containing firing pattern metrics
        """
        if len(neuron_signal) == 0:
            raise ValueError("Empty neuron signal provided")

        # Basic signal statistics
        baseline = np.percentile(neuron_signal, 20)  # 20th percentile as baseline
        peak = np.percentile(neuron_signal, 95)      # 95th percentile as peak
        signal_std = np.std(neuron_signal)

        # Firing above baseline
        threshold = baseline + self.baseline_threshold * signal_std
        above_baseline = neuron_signal > threshold
        n_activations = np.sum(above_baseline)
        activation_fraction = n_activations / len(neuron_signal)

        # Activity burst analysis
        burst_starts, burst_ends = self._detect_activity_bursts(above_baseline)
        n_bursts = len(burst_starts)

        if n_bursts > 0:
            burst_durations = burst_ends - burst_starts
            mean_burst_duration = np.mean(burst_durations)
            max_burst_duration = np.max(burst_durations)
        else:
            mean_burst_duration = 0
            max_burst_duration = 0

        # Signal-to-noise ratio
        signal_power = np.mean(neuron_signal**2)
        noise_estimate = np.var(neuron_signal[~above_baseline]) if np.any(~above_baseline) else np.var(neuron_signal)
        snr = signal_power / (noise_estimate + 1e-10)

        # Place field specific analysis
        place_field_metrics = {}
        if place_field_center is not None and self.spatial_data is not None:
            place_field_metrics = self._analyze_place_field_activity(
                neuron_signal, place_field_center, above_baseline
            )

        return {
            'neuron_id': neuron_id,
            'baseline': baseline,
            'peak': peak,
            'signal_std': signal_std,
            'n_activations': n_activations,
            'activation_fraction': activation_fraction,
            'n_bursts': n_bursts,
            'mean_burst_duration': mean_burst_duration,
            'max_burst_duration': max_burst_duration,
            'snr': snr,
            **place_field_metrics
        }

    def _detect_activity_bursts(self, binary_activity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect continuous bursts of activity in binary activity signal.

        Parameters
        ----------
        binary_activity : ndarray
            Binary array indicating activity periods

        Returns
        -------
        tuple
            (burst_starts, burst_ends) arrays of burst boundaries
        """
        if len(binary_activity) == 0:
            return np.array([]), np.array([])

        # Find transitions
        diff = np.diff(binary_activity.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if binary_activity[0]:
            starts = np.concatenate([[0], starts])
        if binary_activity[-1]:
            ends = np.concatenate([ends, [len(binary_activity)]])

        # Filter by minimum duration
        valid_bursts = (ends - starts) >= self.min_visit_duration

        return starts[valid_bursts], ends[valid_bursts]

    def _analyze_place_field_activity(self,
                                    neuron_signal: np.ndarray,
                                    place_field_center: np.ndarray,
                                    above_baseline: np.ndarray) -> Dict:
        """
        Analyze activity specifically within place field regions.

        Parameters
        ----------
        neuron_signal : ndarray
            Calcium signal for the neuron
        place_field_center : ndarray
            Place field center coordinates
        above_baseline : ndarray
            Binary array indicating above-baseline activity

        Returns
        -------
        dict
            Place field specific metrics
        """
        # Calculate place field response
        positions = self.spatial_data  # Shape: (2, n_timepoints)
        place_field_response = gaussian_place_field(
            positions, place_field_center, self.field_sigma
        )

        # Define "in field" periods
        in_field = place_field_response > self.field_visit_threshold

        # Place field visits
        visit_starts, visit_ends = self._detect_activity_bursts(in_field)
        n_field_visits = len(visit_starts)

        # Activity during field visits
        active_during_visits = 0
        total_visit_time = 0

        if n_field_visits > 0:
            for start, end in zip(visit_starts, visit_ends):
                visit_duration = end - start
                total_visit_time += visit_duration

                # Check if neuron was active during this visit
                visit_activity = above_baseline[start:end]
                if np.any(visit_activity):
                    active_during_visits += 1

        # In-field vs out-of-field firing
        in_field_activity = activation_fraction_in_field = 0
        out_field_activity = activation_fraction_out_field = 0

        if np.any(in_field):
            in_field_activity = np.mean(neuron_signal[in_field])
            activation_fraction_in_field = np.mean(above_baseline[in_field])

        if np.any(~in_field):
            out_field_activity = np.mean(neuron_signal[~in_field])
            activation_fraction_out_field = np.mean(above_baseline[~in_field])

        # Field selectivity ratio
        field_selectivity = (in_field_activity / (out_field_activity + 1e-10))

        return {
            'n_field_visits': n_field_visits,
            'active_during_visits': active_during_visits,
            'visit_success_rate': active_during_visits / max(n_field_visits, 1),
            'total_visit_time': total_visit_time,
            'time_in_field_fraction': np.sum(in_field) / len(in_field),
            'in_field_activity': in_field_activity,
            'out_field_activity': out_field_activity,
            'field_selectivity': field_selectivity,
            'activation_fraction_in_field': activation_fraction_in_field,
            'activation_fraction_out_field': activation_fraction_out_field
        }

    def analyze_all_neurons(self) -> None:
        """
        Analyze firing patterns for all neurons in the population.
        """
        if self.verbose:
            logger.info("Analyzing firing patterns for all neurons...")

        calcium_data = self.exp.calcium.scdata  # Use scaled calcium data

        # Analyze manifold neurons
        for i in range(self.n_manifold):
            neuron_signal = calcium_data[i, :]
            place_field_center = self.place_field_centers[i] if self.place_field_centers is not None else None

            metrics = self.analyze_neuron_firing_pattern(i, neuron_signal, place_field_center)
            self.manifold_metrics[i] = metrics

        # Analyze feature neurons
        for i in range(self.n_manifold, self.n_manifold + self.n_feature):
            neuron_signal = calcium_data[i, :]
            metrics = self.analyze_neuron_firing_pattern(i, neuron_signal)
            self.feature_metrics[i] = metrics

        if self.verbose:
            logger.info(f"Analysis complete: {len(self.manifold_metrics)} manifold neurons, "
                       f"{len(self.feature_metrics)} feature neurons")

    def analyze_spatial_trajectory(self) -> Dict:
        """
        Analyze spatial movement patterns and their relationship to place cell firing.

        Returns
        -------
        dict
            Spatial trajectory analysis results
        """
        if self.spatial_data is None:
            logger.warning("No spatial data available for trajectory analysis")
            return {}

        positions = self.spatial_data  # Shape: (2, n_timepoints)

        # Movement speed analysis
        dx = np.diff(positions[0, :])
        dy = np.diff(positions[1, :])
        speeds = np.sqrt(dx**2 + dy**2) * self.exp.fps  # Convert to units/second

        # Spatial coverage
        x_range = np.max(positions[0, :]) - np.min(positions[0, :])
        y_range = np.max(positions[1, :]) - np.min(positions[1, :])

        # Path tortuosity (path length / direct distance)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        direct_distance = np.sqrt((positions[0, -1] - positions[0, 0])**2 +
                                 (positions[1, -1] - positions[1, 0])**2)
        tortuosity = path_length / (direct_distance + 1e-10)

        # Time spent in different regions (grid analysis)
        n_grid = 10
        x_bins = np.linspace(np.min(positions[0, :]), np.max(positions[0, :]), n_grid + 1)
        y_bins = np.linspace(np.min(positions[1, :]), np.max(positions[1, :]), n_grid + 1)

        occupancy, _, _ = np.histogram2d(positions[0, :], positions[1, :], bins=[x_bins, y_bins])
        occupancy_fraction = occupancy / np.sum(occupancy)

        return {
            'mean_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'speed_std': np.std(speeds),
            'x_range': x_range,
            'y_range': y_range,
            'spatial_coverage': x_range * y_range,
            'path_length': path_length,
            'tortuosity': tortuosity,
            'occupancy_entropy': -np.sum(occupancy_fraction * np.log(occupancy_fraction + 1e-10))
        }

    def compute_distinguishability_metrics(self) -> Dict:
        """
        Compute metrics for distinguishing place cells from feature neurons.

        Returns
        -------
        dict
            Distinguishability analysis results
        """
        if not self.manifold_metrics or not self.feature_metrics:
            raise ValueError("Must run analyze_all_neurons() first")

        # Extract key metrics for comparison
        manifold_values = {}
        feature_values = {}

        key_metrics = ['activation_fraction', 'snr', 'n_bursts', 'field_selectivity']

        for metric in key_metrics:
            manifold_vals = []
            feature_vals = []

            for neuron_metrics in self.manifold_metrics.values():
                if metric in neuron_metrics:
                    manifold_vals.append(neuron_metrics[metric])

            for neuron_metrics in self.feature_metrics.values():
                if metric in neuron_metrics:
                    feature_vals.append(neuron_metrics[metric])

            if manifold_vals and feature_vals:
                manifold_values[metric] = np.array(manifold_vals)
                feature_values[metric] = np.array(feature_vals)

        # Statistical tests
        statistical_results = {}
        for metric in manifold_values.keys():
            man_vals = manifold_values[metric]
            feat_vals = feature_values[metric]

            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(man_vals, feat_vals, alternative='two-sided')

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(man_vals, ddof=1) + np.var(feat_vals, ddof=1)) / 2)
            cohens_d = (np.mean(man_vals) - np.mean(feat_vals)) / (pooled_std + 1e-10)

            statistical_results[metric] = {
                'manifold_mean': np.mean(man_vals),
                'feature_mean': np.mean(feat_vals),
                'manifold_std': np.std(man_vals),
                'feature_std': np.std(feat_vals),
                'mannwhitney_statistic': statistic,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
            }

        # Classification analysis (distinguishability)
        classification_results = {}

        for metric in manifold_values.keys():
            man_vals = manifold_values[metric]
            feat_vals = feature_values[metric]

            if len(man_vals) > 0 and len(feat_vals) > 0:
                # Create labels and data
                labels = np.concatenate([np.ones(len(man_vals)), np.zeros(len(feat_vals))])
                values = np.concatenate([man_vals, feat_vals])

                # ROC AUC
                try:
                    auc = roc_auc_score(labels, values)
                except ValueError:
                    auc = 0.5  # If all values are identical

                classification_results[metric] = {
                    'auc': auc,
                    'separability': self._interpret_auc(auc)
                }

        return {
            'statistical_tests': statistical_results,
            'classification_performance': classification_results,
            'manifold_values': manifold_values,
            'feature_values': feature_values
        }

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_auc(self, auc: float) -> str:
        """Interpret AUC separability."""
        if auc < 0.6:
            return "poor"
        elif auc < 0.7:
            return "fair"
        elif auc < 0.8:
            return "good"
        elif auc < 0.9:
            return "very good"
        else:
            return "excellent"

    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.

        Returns
        -------
        str
            Formatted summary report
        """
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            return "No analysis results available. Run full analysis first."

        report = []
        report.append("="*80)
        report.append("PLACE CELL FIRING PATTERN ANALYSIS REPORT")
        report.append("="*80)

        # Basic statistics
        report.append(f"\nEXPERIMENT OVERVIEW:")
        report.append(f"  Total neurons: {self.n_manifold + self.n_feature}")
        report.append(f"  Manifold neurons (place cells): {self.n_manifold}")
        report.append(f"  Feature neurons: {self.n_feature}")
        report.append(f"  Duration: {self.exp.static_features.get('duration', 'unknown')}s")
        report.append(f"  Sampling rate: {self.exp.fps} Hz")

        # Spatial trajectory
        if 'spatial_analysis' in self.analysis_results:
            spatial = self.analysis_results['spatial_analysis']
            report.append(f"\nSPATIAL TRAJECTORY ANALYSIS:")
            report.append(f"  Mean speed: {spatial.get('mean_speed', 0):.4f} units/s")
            report.append(f"  Spatial coverage: {spatial.get('spatial_coverage', 0):.4f}")
            report.append(f"  Path tortuosity: {spatial.get('tortuosity', 0):.2f}")

        # Distinguishability results
        if 'distinguishability' in self.analysis_results:
            dist = self.analysis_results['distinguishability']

            report.append(f"\nDISTINGUISHABILITY ANALYSIS:")
            report.append(f"=" + "="*40)

            for metric, results in dist['statistical_tests'].items():
                report.append(f"\n{metric.upper()}:")
                report.append(f"  Manifold mean: {results['manifold_mean']:.4f} ± {results['manifold_std']:.4f}")
                report.append(f"  Feature mean:  {results['feature_mean']:.4f} ± {results['feature_std']:.4f}")
                report.append(f"  Effect size (Cohen's d): {results['cohens_d']:.3f} ({results['effect_size_interpretation']})")

                if results['p_value'] < 0.001:
                    sig_str = "*** (p < 0.001)"
                elif results['p_value'] < 0.01:
                    sig_str = "** (p < 0.01)"
                elif results['p_value'] < 0.05:
                    sig_str = "* (p < 0.05)"
                else:
                    sig_str = "(not significant)"

                report.append(f"  Statistical significance: {sig_str}")

                if metric in dist['classification_performance']:
                    auc = dist['classification_performance'][metric]['auc']
                    sep = dist['classification_performance'][metric]['separability']
                    report.append(f"  Classification AUC: {auc:.3f} ({sep} separability)")

        # Key findings
        report.append(f"\nKEY FINDINGS:")
        report.append(f"=" + "="*40)

        if 'distinguishability' in self.analysis_results:
            dist = self.analysis_results['distinguishability']

            # Check for good distinguishability
            good_separability = []
            for metric, results in dist['classification_performance'].items():
                if results['auc'] > 0.7:
                    good_separability.append(metric)

            if good_separability:
                report.append(f"✓ DISTINGUISHABLE: Place cells can be distinguished from feature neurons")
                report.append(f"  Best separating features: {', '.join(good_separability)}")
            else:
                report.append(f"⚠ LIMITED DISTINGUISHABILITY: Place cells are difficult to distinguish")

            # Check sparsity
            if 'activation_fraction' in dist['statistical_tests']:
                act_frac = dist['statistical_tests']['activation_fraction']
                manifold_sparsity = 1 - act_frac['manifold_mean']

                if manifold_sparsity > 0.8:
                    report.append(f"⚠ HIGH SPARSITY: Place cells are active only {100*(1-manifold_sparsity):.1f}% of the time")
                    report.append(f"  This may make them difficult to distinguish from noise")
                else:
                    report.append(f"✓ MODERATE SPARSITY: Place cells are active {100*(1-manifold_sparsity):.1f}% of the time")

        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        report.append(f"=" + "="*40)

        if hasattr(self, 'analysis_results') and 'distinguishability' in self.analysis_results:
            dist = self.analysis_results['distinguishability']

            # Check if we need more data
            if 'activation_fraction' in dist['classification_performance']:
                auc = dist['classification_performance']['activation_fraction']['auc']
                if auc < 0.7:
                    report.append("• Consider longer recording sessions to capture more place field visits")
                    report.append("• Increase sampling rate to better capture brief activation events")

            if 'field_selectivity' in dist['statistical_tests']:
                selectivity = dist['statistical_tests']['field_selectivity']
                if selectivity['cohens_d'] < 0.5:
                    report.append("• Consider adjusting place field parameters (size, peak rate)")
                    report.append("• Verify spatial trajectory covers place fields adequately")

        return "\n".join(report)


def run_place_cell_firing_analysis(n_neurons: int = 60,
                                  duration: float = 600,
                                  fps: float = 20.0,
                                  seed: int = 42,
                                  verbose: bool = True) -> PlaceCellFiringAnalyzer:
    """
    Run complete place cell firing pattern analysis.

    Parameters
    ----------
    n_neurons : int, optional
        Total number of neurons
    duration : float, optional
        Experiment duration in seconds
    fps : float, optional
        Sampling rate in Hz
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Whether to print progress messages

    Returns
    -------
    PlaceCellFiringAnalyzer
        Analyzer object with complete results
    """
    if verbose:
        logger.info("Generating mixed population experiment...")

    # Generate experiment (same parameters as other analyses)
    exp, info = generate_mixed_population_exp(
        n_neurons=n_neurons,
        manifold_fraction=0.6,  # 60% manifold neurons (place cells)
        manifold_type='2d_spatial',
        n_discrete_features=2,
        n_continuous_features=2,
        correlation_mode='independent',
        duration=duration,
        fps=fps,
        seed=seed,
        verbose=verbose,
        return_info=True
    )

    if verbose:
        n_manifold = info['population_composition']['n_manifold']
        n_feature = info['population_composition']['n_feature_selective']
        logger.info(f"Generated experiment: {n_manifold} place cells, {n_feature} feature neurons")

    # Initialize analyzer
    analyzer = PlaceCellFiringAnalyzer(exp, info, verbose=verbose)

    # Run analysis
    if verbose:
        logger.info("Analyzing firing patterns...")

    analyzer.analyze_all_neurons()

    if verbose:
        logger.info("Analyzing spatial trajectory...")

    spatial_results = analyzer.analyze_spatial_trajectory()
    analyzer.analysis_results['spatial_analysis'] = spatial_results

    if verbose:
        logger.info("Computing distinguishability metrics...")

    distinguishability_results = analyzer.compute_distinguishability_metrics()
    analyzer.analysis_results['distinguishability'] = distinguishability_results

    if verbose:
        logger.info("Analysis complete!")

    return analyzer


if __name__ == "__main__":
    # Run the analysis
    analyzer = run_place_cell_firing_analysis(verbose=True)

    # Generate and print summary report
    report = analyzer.generate_summary_report()
    print("\n" + report)

    # Create visualization
    if analyzer.analysis_results:
        plt.figure(figsize=(15, 10))

        # Plot 1: Activation fraction comparison
        plt.subplot(2, 3, 1)
        dist = analyzer.analysis_results['distinguishability']
        if 'activation_fraction' in dist['manifold_values']:
            manifold_act = dist['manifold_values']['activation_fraction']
            feature_act = dist['feature_values']['activation_fraction']

            plt.boxplot([manifold_act, feature_act], labels=['Place Cells', 'Feature Neurons'])
            plt.ylabel('Activation Fraction')
            plt.title('Neural Activity Levels')
            plt.grid(True, alpha=0.3)

        # Plot 2: Signal-to-noise ratio
        plt.subplot(2, 3, 2)
        if 'snr' in dist['manifold_values']:
            manifold_snr = dist['manifold_values']['snr']
            feature_snr = dist['feature_values']['snr']

            plt.boxplot([manifold_snr, feature_snr], labels=['Place Cells', 'Feature Neurons'])
            plt.ylabel('Signal-to-Noise Ratio')
            plt.title('Signal Quality')
            plt.grid(True, alpha=0.3)

        # Plot 3: Field selectivity (manifold neurons only)
        plt.subplot(2, 3, 3)
        if 'field_selectivity' in dist['manifold_values']:
            field_sel = dist['manifold_values']['field_selectivity']
            plt.hist(field_sel, bins=15, alpha=0.7, color='blue')
            plt.xlabel('Field Selectivity Ratio')
            plt.ylabel('Count')
            plt.title('Place Field Selectivity')
            plt.grid(True, alpha=0.3)

        # Plot 4: Spatial trajectory
        plt.subplot(2, 3, 4)
        if analyzer.spatial_data is not None:
            positions = analyzer.spatial_data
            plt.plot(positions[0, :], positions[1, :], 'k-', alpha=0.5, linewidth=0.5)
            plt.scatter(positions[0, ::100], positions[1, ::100],
                       c=np.arange(0, len(positions[0, :]), 100),
                       cmap='viridis', s=10)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Spatial Trajectory')
            plt.axis('equal')

        # Plot 5: Classification performance
        plt.subplot(2, 3, 5)
        metrics = list(dist['classification_performance'].keys())
        aucs = [dist['classification_performance'][m]['auc'] for m in metrics]

        bars = plt.bar(range(len(metrics)), aucs)
        plt.xticks(range(len(metrics)), metrics, rotation=45)
        plt.ylabel('AUC')
        plt.title('Classification Performance')
        plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Good threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Color bars based on performance
        for i, (bar, auc) in enumerate(zip(bars, aucs)):
            if auc > 0.7:
                bar.set_color('green')
            elif auc > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # Plot 6: Summary statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')

        # Create summary text
        summary_lines = []
        if 'activation_fraction' in dist['statistical_tests']:
            act_stats = dist['statistical_tests']['activation_fraction']
            sparsity = 1 - act_stats['manifold_mean']
            summary_lines.append(f"Place Cell Sparsity: {100*sparsity:.1f}%")

        if 'field_selectivity' in dist['statistical_tests']:
            sel_stats = dist['statistical_tests']['field_selectivity']
            summary_lines.append(f"Field Selectivity: {sel_stats['manifold_mean']:.2f}x")

        good_aucs = sum(1 for auc in aucs if auc > 0.7)
        summary_lines.append(f"Distinguishable Features: {good_aucs}/{len(aucs)}")

        # Add spatial info
        if 'spatial_analysis' in analyzer.analysis_results:
            spatial = analyzer.analysis_results['spatial_analysis']
            summary_lines.append(f"Mean Speed: {spatial.get('mean_speed', 0):.3f} u/s")

        plt.text(0.1, 0.5, '\n'.join(summary_lines), fontsize=12,
                verticalalignment='center', fontfamily='monospace')
        plt.title('Summary Statistics')

        plt.suptitle('Place Cell Firing Pattern Analysis Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # Save results
    logger.info(f"Analysis complete. Results saved in analyzer object.")
    logger.info(f"Manifold neurons analyzed: {len(analyzer.manifold_metrics)}")
    logger.info(f"Feature neurons analyzed: {len(analyzer.feature_metrics)}")