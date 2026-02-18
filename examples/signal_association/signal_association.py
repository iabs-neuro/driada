"""
Signal association example -- information-theory primitives
===========================================================

Demonstrates DRIADA's information-theory functions on synthetic signals
with known relationships.

Sections:
1. Create TimeSeries from numpy arrays
2. get_mi -- pairwise MI, GCMI vs KSG on monotonic and non-monotonic data
3. get_sim -- compare MI vs Pearson vs Spearman on the same data
4. get_tdmi -- time-delayed MI profile, find optimal lag
5. conditional_mi -- I(X;Y|Z): conditioning removes shared variance
6. interaction_information -- synergy vs redundancy
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from driada.information import (
    TimeSeries,
    get_mi,
    get_sim,
    get_tdmi,
    conditional_mi,
    interaction_information,
)


def main():
    print("=" * 60)
    print("DRIADA signal association example")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 5000

    # ------------------------------------------------------------------
    # 1. Create TimeSeries from numpy arrays
    # ------------------------------------------------------------------
    print("\n[1] Creating TimeSeries objects")
    print("-" * 40)

    continuous = rng.normal(size=n)
    ts_cont = TimeSeries(continuous, ts_type="linear", name="continuous")
    print(f"  Continuous: type={ts_cont.type_info}, len={len(ts_cont.data)}")

    discrete = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])
    ts_disc = TimeSeries(discrete, ts_type="categorical", name="discrete")
    print(f"  Discrete:   type={ts_disc.type_info}, len={len(ts_disc.data)}")

    circular = rng.uniform(0, 2 * np.pi, size=n)
    ts_circ = TimeSeries(circular, ts_type="circular", name="circular")
    print(f"  Circular:   type={ts_circ.type_info}, len={len(ts_circ.data)}")

    # ------------------------------------------------------------------
    # 2. get_mi -- pairwise MI
    # ------------------------------------------------------------------
    print("\n[2] Pairwise mutual information (get_mi)")
    print("-" * 40)

    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    y_corr = x + 0.5 * noise          # correlated with x
    y_indep = rng.normal(size=n)       # independent of x

    ts_x = TimeSeries(x)
    ts_y_corr = TimeSeries(y_corr)
    ts_y_indep = TimeSeries(y_indep)

    mi_corr = get_mi(ts_x, ts_y_corr)
    mi_indep = get_mi(ts_x, ts_y_indep)
    print(f"  MI(X, Y_correlated)  = {mi_corr:.4f} bits")
    print(f"  MI(X, Y_independent) = {mi_indep:.4f} bits")
    print(f"  Correlated MI >> independent MI: {mi_corr > 5 * mi_indep}")

    # Compare estimators on monotonic vs non-monotonic relationships.
    # GCMI reduces to -0.5*log(1-rho^2) where rho is Spearman rank correlation,
    # so it only captures monotonic dependency. KSG captures any dependency.
    mi_gcmi = get_mi(ts_x, ts_y_corr, estimator="gcmi")
    mi_ksg = get_mi(ts_x, ts_y_corr, estimator="ksg")
    print(f"\n  Monotonic relationship (y = x + noise):")
    print(f"    GCMI: {mi_gcmi:.4f} bits")
    print(f"    KSG:  {mi_ksg:.4f} bits")
    print(f"    (agree because relationship is monotonic)")

    # Non-monotonic: y = x^2. Spearman rho ~ 0 due to exact symmetry, so GCMI ~ 0.
    x_sym = rng.uniform(-3, 3, size=n)
    y_quad = x_sym ** 2 + 0.3 * rng.normal(size=n)
    ts_x_sym = TimeSeries(x_sym)
    ts_y_quad = TimeSeries(y_quad)
    mi_gcmi_q = get_mi(ts_x_sym, ts_y_quad, estimator="gcmi")
    mi_ksg_q = get_mi(ts_x_sym, ts_y_quad, estimator="ksg")
    print(f"\n  Non-monotonic relationship (y = x^2 + noise):")
    print(f"    GCMI: {mi_gcmi_q:.4f} bits  (blind to symmetric dependency)")
    print(f"    KSG:  {mi_ksg_q:.4f} bits  (captures it)")
    print(f"    KSG >> GCMI: {mi_ksg_q > 3 * mi_gcmi_q}")

    # ------------------------------------------------------------------
    # 3. get_sim -- compare metrics on the same data
    # ------------------------------------------------------------------
    print("\n[3] Similarity metrics comparison (get_sim)")
    print("-" * 40)

    metrics = ["mi", "pearsonr", "spearmanr"]
    for metric in metrics:
        val = get_sim(ts_x, ts_y_corr, metric=metric)
        print(f"  {metric:12s}(X, Y_corr) = {val:.4f}")

    # ------------------------------------------------------------------
    # 4. get_tdmi -- time-delayed MI
    # ------------------------------------------------------------------
    print("\n[4] Time-delayed MI (get_tdmi)")
    print("-" * 40)

    # Create a signal with known lag=15 autocorrelation
    base = rng.normal(size=n)
    lag = 15
    lagged = np.zeros(n)
    lagged[lag:] = base[:-lag]
    signal = base + 0.3 * rng.normal(size=n) + 0.8 * lagged

    max_shift = 50
    tdmi_values = np.array(get_tdmi(signal, max_shift=max_shift))
    best_lag = np.argmax(tdmi_values) + 1  # get_tdmi starts at min_shift=1
    print(f"  True lag: {lag}")
    print(f"  TDMI peak lag: {best_lag}")
    print(f"  TDMI at peak: {tdmi_values[best_lag - 1]:.4f} bits")
    print(f"  Lag correctly detected: {abs(best_lag - lag) <= 2}")

    # ------------------------------------------------------------------
    # 5. conditional_mi -- I(X;Y|Z)
    # ------------------------------------------------------------------
    print("\n[5] Conditional MI: I(X;Y|Z)")
    print("-" * 40)

    z = rng.normal(size=n)
    x_from_z = z + 0.3 * rng.normal(size=n)
    y_from_z = z + 0.3 * rng.normal(size=n)

    ts_xz = TimeSeries(x_from_z)
    ts_yz = TimeSeries(y_from_z)
    ts_z = TimeSeries(z)

    mi_xy = get_mi(ts_xz, ts_yz)
    cmi_xy_z = conditional_mi(ts_xz, ts_yz, ts_z)

    print(f"  I(X;Y)   = {mi_xy:.4f} bits  (shared via Z)")
    print(f"  I(X;Y|Z) = {cmi_xy_z:.4f} bits  (residual after conditioning)")
    print(f"  Conditioning reduces MI: {cmi_xy_z < mi_xy * 0.5}")

    # ------------------------------------------------------------------
    # 6. interaction_information -- synergy vs redundancy
    # ------------------------------------------------------------------
    print("\n[6] Interaction information: synergy vs redundancy")
    print("-" * 40)

    # Redundancy: Y and Z provide overlapping info about X
    x_r = rng.normal(size=n)
    y_r = TimeSeries(x_r + 0.2 * rng.normal(size=n))
    z_r = TimeSeries(x_r + 0.2 * rng.normal(size=n))
    ts_xr = TimeSeries(x_r)

    ii_redund = interaction_information(ts_xr, y_r, z_r)
    print(f"  Redundancy example: II = {ii_redund:.4f} (expected < 0)")

    # Synergy: XOR-like relationship
    a = rng.choice([0, 1], size=n).astype(float)
    b = rng.choice([0, 1], size=n).astype(float)
    xor_signal = (a + b + 0.1 * rng.normal(size=n))

    ts_xor = TimeSeries(xor_signal)
    ts_a = TimeSeries(a, ts_type="binary")
    ts_b = TimeSeries(b, ts_type="binary")

    ii_synergy = interaction_information(ts_xor, ts_a, ts_b)
    print(f"  Synergy example:    II = {ii_synergy:.4f} (expected > 0)")
    print(f"  Redundancy is negative: {ii_redund < 0}")

    print("\n" + "=" * 60)
    print("Signal association example complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
