"""
Investigation of MI distribution types for INTENSE statistical testing.

This module investigates why metric_distr_type='norm' works better than 'gamma'
for MI distributions in INTENSE, despite gamma being theoretically more appropriate.

Key questions addressed:
1. What are the statistical properties of actual MI shuffle distributions?
2. How do different distributions (norm, gamma, lognorm) fit the data?
3. Why does normal distribution give better detection performance?
4. What are the root causes of this empirical observation?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import normaltest, shapiro, anderson, kstest, gamma, norm, lognorm
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from .stats import get_mi_distr_pvalue
from ..experiment import generate_mixed_population_exp, generate_circular_manifold_exp


@dataclass
class DistributionFitResult:
    """Results from fitting a distribution to data."""

    distribution: str
    parameters: Tuple
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float
    log_likelihood: float
    fitted_distribution: object


@dataclass
class ShuffleDistributionData:
    """Container for shuffle distribution data and metadata."""

    shuffle_values: np.ndarray
    true_mi: float
    neuron_id: Union[str, int]
    feature_id: str
    is_significant: bool
    p_value_norm: float
    p_value_gamma: float
    statistical_properties: Dict


class MIDistributionInvestigator:
    """
    Investigates MI distribution fitting for INTENSE statistical testing.

    This class provides comprehensive analysis of why normal distribution
    works better than gamma for MI shuffle distributions.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the MI distribution investigator.

        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility. Default: 42.
        """
        self.random_state = random_state
        np.random.seed(random_state)

        # Distributions to test
        self.distributions = {
            "norm": norm,
            "gamma": gamma,
            "lognorm": lognorm,
            "expon": stats.expon,
            "weibull_min": stats.weibull_min,
            "beta": stats.beta,
        }

        # Results storage
        self.shuffle_data: List[ShuffleDistributionData] = []
        self.fit_results: Dict[str, Dict[str, DistributionFitResult]] = {}

    def generate_test_data(
        self, n_scenarios: int = 5, n_shuffles: int = 1000
    ) -> List[ShuffleDistributionData]:
        """
        Generate test data with known MI distributions.

        Parameters
        ----------
        n_scenarios : int, optional
            Number of different scenarios to test. Default: 5.
        n_shuffles : int, optional
            Number of shuffles per scenario. Default: 1000.

        Returns
        -------
        shuffle_data : List[ShuffleDistributionData]
            List of shuffle distribution data for analysis.
        """
        print("Generating test data for MI distribution investigation...")

        shuffle_data = []

        # Scenario 1: Circular manifold (head direction cells)
        print("  - Scenario 1: Circular manifold")
        exp_circular = generate_circular_manifold_exp(
            n_neurons=20, duration=300, fps=20, seed=self.random_state
        )

        # Extract shuffle distributions from INTENSE analysis
        circular_data = self._extract_shuffle_distributions(
            exp_circular, scenario_name="circular", n_shuffles=n_shuffles
        )
        shuffle_data.extend(circular_data)

        # Scenario 2: Mixed population with spatial and feature components
        print("  - Scenario 2: Mixed population")
        exp_mixed = generate_mixed_population_exp(
            n_neurons=50,
            manifold_type="2d_spatial",
            manifold_fraction=0.6,
            n_discrete_features=1,
            n_continuous_features=2,
            duration=300,
            fps=20,
            seed=self.random_state + 1,
        )

        mixed_data = self._extract_shuffle_distributions(
            exp_mixed, scenario_name="mixed", n_shuffles=n_shuffles
        )
        shuffle_data.extend(mixed_data)

        self.shuffle_data = shuffle_data
        print(f"Generated {len(shuffle_data)} shuffle distributions for analysis")

        return shuffle_data

    def _extract_shuffle_distributions(
        self, exp, scenario_name: str, n_shuffles: int
    ) -> List[ShuffleDistributionData]:
        """
        Extract shuffle distributions from INTENSE analysis.

        Parameters
        ----------
        exp : Experiment
            DRIADA experiment object.
        scenario_name : str
            Name of the scenario for identification.
        n_shuffles : int
            Number of shuffles to use.

        Returns
        -------
        shuffle_data : List[ShuffleDistributionData]
            Extracted shuffle distribution data.
        """
        # Run INTENSE analysis to get shuffle distributions
        from .intense_base import scan_pairs, calculate_optimal_delays

        # Get features and neurons
        available_features = list(exp.dynamic_features.keys())
        if len(available_features) > 3:
            available_features = available_features[:3]  # Limit for efficiency

        cell_ids = list(range(min(10, exp.n_cells)))  # Limit for efficiency

        shuffle_data = []

        for i, cell_id in enumerate(cell_ids):
            for j, feature_id in enumerate(available_features):
                try:
                    # Get neural and feature time series
                    neural_ts = exp.calcium[cell_id]
                    feature_ts = exp.dynamic_features[feature_id]

                    # Create TimeSeries objects
                    from ..information.info_base import TimeSeries

                    ts_neural = TimeSeries(neural_ts)
                    ts_feature = (
                        feature_ts
                        if hasattr(feature_ts, "data")
                        else TimeSeries(feature_ts)
                    )

                    # Calculate optimal delays
                    optimal_delays = calculate_optimal_delays(
                        [ts_neural],
                        [ts_feature],
                        metric="mi",
                        shift_window=20,  # Small window for efficiency
                        ds=1,
                        verbose=False,
                    )

                    # Run scan_pairs to get MI shuffle distributions
                    random_shifts, me_total = scan_pairs(
                        [ts_neural],
                        [ts_feature],
                        metric="mi",
                        nsh=n_shuffles,
                        optimal_delays=optimal_delays,
                        joint_distr=False,
                        ds=1,
                        mask=None,
                        noise_const=1e-3,
                        seed=self.random_state + i * 100 + j,
                        allow_mixed_dimensions=False,
                        enable_progressbar=False,
                    )

                    # Extract true MI and shuffle values
                    true_mi = me_total[0, 0, 0]  # First element is true MI
                    shuffle_values = me_total[0, 0, 1:]  # Rest are shuffles

                    # Calculate p-values with both distributions
                    p_val_norm = get_mi_distr_pvalue(shuffle_values, true_mi, "norm")
                    p_val_gamma = get_mi_distr_pvalue(shuffle_values, true_mi, "gamma")

                    # Calculate statistical properties
                    stats_props = self._calculate_statistical_properties(shuffle_values)

                    # Determine significance (using p < 0.05 as threshold)
                    is_significant = min(p_val_norm, p_val_gamma) < 0.05

                    # Create data container
                    data = ShuffleDistributionData(
                        shuffle_values=shuffle_values,
                        true_mi=true_mi,
                        neuron_id=f"{scenario_name}_neuron_{cell_id}",
                        feature_id=feature_id,
                        is_significant=is_significant,
                        p_value_norm=p_val_norm,
                        p_value_gamma=p_val_gamma,
                        statistical_properties=stats_props,
                    )

                    shuffle_data.append(data)

                except Exception as e:
                    print(
                        f"    Warning: Failed to extract data for neuron {cell_id}, feature {feature_id}: {e}"
                    )
                    continue

        return shuffle_data

    def _calculate_statistical_properties(self, data: np.ndarray) -> Dict:
        """
        Calculate comprehensive statistical properties of data.

        Parameters
        ----------
        data : np.ndarray
            Input data array.

        Returns
        -------
        properties : Dict
            Dictionary containing statistical properties.
        """
        # Basic moments
        mean = np.mean(data)
        std = np.std(data)
        var = np.var(data)

        # Shape statistics
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        # Quantiles
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        iqr = q75 - q25

        # Normality tests
        try:
            shapiro_stat, shapiro_p = shapiro(data)
        except:
            shapiro_stat, shapiro_p = np.nan, np.nan

        try:
            normaltest_stat, normaltest_p = normaltest(data)
        except:
            normaltest_stat, normaltest_p = np.nan, np.nan

        # Anderson-Darling test for normality
        try:
            anderson_result = anderson(data, dist="norm")
            anderson_stat = anderson_result.statistic
            anderson_critical = anderson_result.critical_values[2]  # 5% level
        except:
            anderson_stat, anderson_critical = np.nan, np.nan

        # Range and outliers
        data_range = np.max(data) - np.min(data)
        outlier_threshold = q75 + 1.5 * iqr
        n_outliers = np.sum(data > outlier_threshold)

        return {
            "mean": mean,
            "std": std,
            "var": var,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "median": q50,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "range": data_range,
            "n_outliers": n_outliers,
            "shapiro_stat": shapiro_stat,
            "shapiro_pvalue": shapiro_p,
            "normaltest_stat": normaltest_stat,
            "normaltest_pvalue": normaltest_p,
            "anderson_stat": anderson_stat,
            "anderson_critical": anderson_critical,
            "min_value": np.min(data),
            "max_value": np.max(data),
            "n_samples": len(data),
        }

    def fit_distributions(
        self, data: np.ndarray, distributions: Optional[List[str]] = None
    ) -> Dict[str, DistributionFitResult]:
        """
        Fit multiple distributions to data and compare goodness of fit.

        Parameters
        ----------
        data : np.ndarray
            Data to fit distributions to.
        distributions : List[str], optional
            List of distribution names to test. If None, uses default set.

        Returns
        -------
        results : Dict[str, DistributionFitResult]
            Dictionary mapping distribution names to fit results.
        """
        if distributions is None:
            distributions = list(self.distributions.keys())

        results = {}
        n_samples = len(data)

        for dist_name in distributions:
            try:
                dist = self.distributions[dist_name]

                # Fit distribution
                if dist_name in ["gamma", "lognorm"]:
                    # Use floc=0 for positive distributions
                    params = dist.fit(data, floc=0)
                elif dist_name == "beta":
                    # Beta distribution needs data in [0,1]
                    if np.min(data) < 0 or np.max(data) > 1:
                        # Skip beta if data is outside [0,1]
                        continue
                    params = dist.fit(data)
                else:
                    params = dist.fit(data)

                # Create fitted distribution
                fitted_dist = dist(*params)

                # Calculate log-likelihood
                log_likelihood = np.sum(dist.logpdf(data, *params))

                # Calculate AIC and BIC
                k = len(params)  # Number of parameters
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n_samples) - 2 * log_likelihood

                # Kolmogorov-Smirnov test
                ks_stat, ks_p = kstest(data, fitted_dist.cdf)

                # Store results
                results[dist_name] = DistributionFitResult(
                    distribution=dist_name,
                    parameters=params,
                    aic=aic,
                    bic=bic,
                    ks_statistic=ks_stat,
                    ks_pvalue=ks_p,
                    log_likelihood=log_likelihood,
                    fitted_distribution=fitted_dist,
                )

            except Exception as e:
                print(f"    Warning: Failed to fit {dist_name}: {e}")
                continue

        return results

    def analyze_all_distributions(self) -> Dict[str, Dict[str, DistributionFitResult]]:
        """
        Analyze distribution fitting for all collected shuffle data.

        Returns
        -------
        fit_results : Dict[str, Dict[str, DistributionFitResult]]
            Nested dictionary with fit results for each data sample.
        """
        print("Analyzing distribution fitting for all shuffle data...")

        fit_results = {}

        for i, data in enumerate(self.shuffle_data):
            data_id = f"{data.neuron_id}_{data.feature_id}"

            # Fit distributions to shuffle values
            results = self.fit_distributions(data.shuffle_values)
            fit_results[data_id] = results

            # Print progress
            if (i + 1) % 10 == 0:
                print(f"  Analyzed {i + 1}/{len(self.shuffle_data)} distributions")

        self.fit_results = fit_results
        print(f"Completed distribution analysis for {len(fit_results)} datasets")

        return fit_results

    def compare_detection_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Compare detection performance between different distributions.

        Returns
        -------
        performance : Dict[str, Dict[str, float]]
            Performance metrics for each distribution type.
        """
        print("Comparing detection performance between distributions...")

        # Calculate detection performance for each distribution
        performance = {}

        for dist_name in ["norm", "gamma", "lognorm"]:
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            for data in self.shuffle_data:
                # Get p-value for this distribution
                p_val = get_mi_distr_pvalue(
                    data.shuffle_values, data.true_mi, dist_name
                )
                predicted_significant = p_val < 0.05

                # Compare with ground truth (using minimum p-value as reference)
                actual_significant = data.is_significant

                if predicted_significant and actual_significant:
                    true_positives += 1
                elif predicted_significant and not actual_significant:
                    false_positives += 1
                elif not predicted_significant and actual_significant:
                    false_negatives += 1
                else:
                    true_negatives += 1

            # Calculate metrics
            total = len(self.shuffle_data)
            sensitivity = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            specificity = (
                true_negatives / (true_negatives + false_positives)
                if (true_negatives + false_positives) > 0
                else 0
            )
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            accuracy = (true_positives + true_negatives) / total

            performance[dist_name] = {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "accuracy": accuracy,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
            }

        return performance

    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of the investigation.

        Returns
        -------
        report : str
            Formatted summary report.
        """
        report = []
        report.append("=" * 80)
        report.append("MI DISTRIBUTION INVESTIGATION REPORT")
        report.append("=" * 80)

        # Data summary
        report.append("\nDATA SUMMARY:")
        report.append(
            f"  Total shuffle distributions analyzed: {len(self.shuffle_data)}"
        )

        significant_count = sum(1 for d in self.shuffle_data if d.is_significant)
        report.append(f"  Significant pairs: {significant_count}")
        report.append(
            f"  Non-significant pairs: {len(self.shuffle_data) - significant_count}"
        )

        # Statistical properties summary
        report.append("\nSTATISTICAL PROPERTIES SUMMARY:")

        # Average statistics across all distributions
        avg_stats = {}
        for prop in [
            "mean",
            "std",
            "skewness",
            "kurtosis",
            "shapiro_pvalue",
            "normaltest_pvalue",
        ]:
            values = [
                d.statistical_properties[prop]
                for d in self.shuffle_data
                if not np.isnan(d.statistical_properties[prop])
            ]
            if values:
                avg_stats[prop] = np.mean(values)

        report.append(f"  Average skewness: {avg_stats.get('skewness', 'N/A'):.3f}")
        report.append(f"  Average kurtosis: {avg_stats.get('kurtosis', 'N/A'):.3f}")
        report.append(
            f"  Average Shapiro p-value: {avg_stats.get('shapiro_pvalue', 'N/A'):.3f}"
        )
        report.append(
            f"  Average normaltest p-value: {avg_stats.get('normaltest_pvalue', 'N/A'):.3f}"
        )

        # Distribution fitting summary
        if self.fit_results:
            report.append("\nDISTRIBUTION FITTING SUMMARY:")

            # Calculate average AIC/BIC for each distribution
            dist_summary = {}
            for dist_name in ["norm", "gamma", "lognorm"]:
                aics = []
                bics = []
                ks_stats = []

                for data_id, results in self.fit_results.items():
                    if dist_name in results:
                        aics.append(results[dist_name].aic)
                        bics.append(results[dist_name].bic)
                        ks_stats.append(results[dist_name].ks_statistic)

                if aics:
                    dist_summary[dist_name] = {
                        "avg_aic": np.mean(aics),
                        "avg_bic": np.mean(bics),
                        "avg_ks_stat": np.mean(ks_stats),
                    }

            for dist_name, stats in dist_summary.items():
                report.append(f"  {dist_name.upper()}:")
                report.append(f"    Average AIC: {stats['avg_aic']:.2f}")
                report.append(f"    Average BIC: {stats['avg_bic']:.2f}")
                report.append(f"    Average KS statistic: {stats['avg_ks_stat']:.3f}")

        # Detection performance summary
        performance = self.compare_detection_performance()
        report.append("\nDETECTION PERFORMANCE COMPARISON:")

        for dist_name, metrics in performance.items():
            report.append(f"  {dist_name.upper()}:")
            report.append(f"    Sensitivity: {metrics['sensitivity']:.3f}")
            report.append(f"    Specificity: {metrics['specificity']:.3f}")
            report.append(f"    Precision: {metrics['precision']:.3f}")
            report.append(f"    Accuracy: {metrics['accuracy']:.3f}")

        # Recommendations
        report.append("\nRECOMMENDATIONS:")

        # Find best performing distribution
        best_dist = max(performance.keys(), key=lambda d: performance[d]["accuracy"])
        report.append(f"  Best performing distribution: {best_dist.upper()}")

        # Analyze why norm might be better
        norm_better_count = sum(
            1 for d in self.shuffle_data if d.p_value_norm < d.p_value_gamma
        )
        report.append(
            f"  Cases where norm gives lower p-value: {norm_better_count}/{len(self.shuffle_data)}"
        )

        # Statistical significance of normality
        normal_like_count = sum(
            1
            for d in self.shuffle_data
            if d.statistical_properties["shapiro_pvalue"] > 0.05
        )
        report.append(
            f"  Distributions that appear normal (Shapiro p>0.05): {normal_like_count}/{len(self.shuffle_data)}"
        )

        return "\n".join(report)

    def create_visualizations(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualizations of the investigation results.

        Parameters
        ----------
        save_path : str, optional
            Path to save visualizations. If None, displays plots.
        """
        print("Creating visualizations...")

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("MI Distribution Investigation Results", fontsize=16)

        # 1. Distribution of statistical properties
        ax1 = axes[0, 0]
        skewness_values = [
            d.statistical_properties["skewness"] for d in self.shuffle_data
        ]
        ax1.hist(skewness_values, bins=20, alpha=0.7, color="skyblue")
        ax1.set_xlabel("Skewness")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Skewness Values")
        ax1.axvline(0, color="red", linestyle="--", alpha=0.7, label="Normal (skew=0)")
        ax1.legend()

        # 2. Kurtosis distribution
        ax2 = axes[0, 1]
        kurtosis_values = [
            d.statistical_properties["kurtosis"] for d in self.shuffle_data
        ]
        ax2.hist(kurtosis_values, bins=20, alpha=0.7, color="lightgreen")
        ax2.set_xlabel("Kurtosis")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Kurtosis Values")
        ax2.axvline(0, color="red", linestyle="--", alpha=0.7, label="Normal (kurt=0)")
        ax2.legend()

        # 3. P-value comparison
        ax3 = axes[0, 2]
        p_norm = [d.p_value_norm for d in self.shuffle_data]
        p_gamma = [d.p_value_gamma for d in self.shuffle_data]
        ax3.scatter(p_norm, p_gamma, alpha=0.6)
        ax3.plot([0, 1], [0, 1], "r--", alpha=0.7)
        ax3.set_xlabel("P-value (norm)")
        ax3.set_ylabel("P-value (gamma)")
        ax3.set_title("P-value Comparison: Norm vs Gamma")
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

        # 4. Example shuffle distribution
        ax4 = axes[1, 0]
        if self.shuffle_data:
            example_data = self.shuffle_data[0]
            ax4.hist(
                example_data.shuffle_values,
                bins=30,
                alpha=0.7,
                density=True,
                color="lightcoral",
            )
            ax4.axvline(
                example_data.true_mi,
                color="red",
                linestyle="-",
                linewidth=2,
                label="True MI",
            )
            ax4.set_xlabel("MI Value")
            ax4.set_ylabel("Density")
            ax4.set_title("Example Shuffle Distribution")
            ax4.legend()

        # 5. Goodness of fit comparison
        ax5 = axes[1, 1]
        if self.fit_results:
            # Get average AIC values for each distribution
            avg_aics = {}
            for dist_name in ["norm", "gamma", "lognorm"]:
                aics = []
                for results in self.fit_results.values():
                    if dist_name in results:
                        aics.append(results[dist_name].aic)
                if aics:
                    avg_aics[dist_name] = np.mean(aics)

            if avg_aics:
                dists = list(avg_aics.keys())
                aics = list(avg_aics.values())
                ax5.bar(dists, aics, color=["skyblue", "lightgreen", "lightcoral"])
                ax5.set_ylabel("Average AIC")
                ax5.set_title("Distribution Fit Quality (lower AIC = better)")
                ax5.tick_params(axis="x", rotation=45)

        # 6. Detection performance
        ax6 = axes[1, 2]
        performance = self.compare_detection_performance()
        if performance:
            dists = list(performance.keys())
            accuracies = [performance[d]["accuracy"] for d in dists]
            ax6.bar(dists, accuracies, color=["skyblue", "lightgreen", "lightcoral"])
            ax6.set_ylabel("Accuracy")
            ax6.set_title("Detection Performance")
            ax6.set_ylim(0, 1)
            ax6.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualizations saved to: {save_path}")
        else:
            plt.show()
