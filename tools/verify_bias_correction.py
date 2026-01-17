"""Verify mathematical equivalence of bias correction patterns."""
import numpy as np
from scipy.special import psi
import sys
sys.path.insert(0, 'C:\\Users\\User\\PycharmProjects\\driada\\src')

from driada.information.info_utils import py_fast_digamma, py_fast_digamma_arr

ln2 = np.log(2)

def pattern1_univariate(n):
    """Pattern 1: compute_mi_batch_fft (direct MI correction)."""
    psi_1 = py_fast_digamma((n - 1) / 2.0)
    psi_2 = py_fast_digamma((n - 2) / 2.0)
    bias_correction = (psi_2 - psi_1) / (2.0 * ln2)
    return bias_correction


def pattern2_univariate(n):
    """Pattern 2: mi_gg-style (correct individual H terms, then combine)."""
    dterm = (ln2 - np.log(n - 1.0)) / 2.0

    # For univariate case: Nvarx=1, Nvary=1, Nvarxy=2
    # H_X correction
    psiterm_x = py_fast_digamma((n - 1.0) / 2.0) / 2.0
    H_X_correction = (dterm + psiterm_x) / ln2

    # H_Y correction (same as H_X for univariate)
    H_Y_correction = (dterm + psiterm_x) / ln2

    # H_XY correction
    psiterm_xy1 = py_fast_digamma((n - 1.0) / 2.0) / 2.0
    psiterm_xy2 = py_fast_digamma((n - 2.0) / 2.0) / 2.0
    H_XY_correction = (2 * dterm + psiterm_xy1 + psiterm_xy2) / ln2

    # MI correction = -H_X_corr - H_Y_corr + H_XY_corr
    return -H_X_correction - H_Y_correction + H_XY_correction


def pattern3_univariate(n):
    """Pattern 3: mi_cd_fft (using scipy.special.psi)."""
    dterm = (ln2 - np.log(n - 1.0)) / 2.0
    psiterm = psi((n - 1.0) / 2.0) / 2.0

    # For univariate case: Nvarx=1, Nvary=1, Nvarxy=2
    # H_X correction
    H_X_correction = (dterm + psiterm) / ln2

    # H_Y correction (same)
    H_Y_correction = (dterm + psiterm) / ln2

    # H_XY correction
    psiterm_xy1 = psi((n - 1.0) / 2.0) / 2.0
    psiterm_xy2 = psi((n - 2.0) / 2.0) / 2.0
    H_XY_correction = (2 * dterm + psiterm_xy1 + psiterm_xy2) / ln2

    # MI correction = -H_X_corr - H_Y_corr + H_XY_corr
    return -H_X_correction - H_Y_correction + H_XY_correction


def pattern2_multivariate(n, d):
    """Pattern 2: multivariate case using py_fast_digamma."""
    dterm = (ln2 - np.log(n - 1.0)) / 2.0
    psiterms = py_fast_digamma_arr((n - np.arange(1, d + 1)) / 2.0) / 2.0
    correction = d * dterm + psiterms.sum()
    return correction  # in nats


def pattern3_multivariate(n, d):
    """Pattern 3: multivariate case using scipy.special.psi."""
    dterm = (ln2 - np.log(n - 1.0)) / 2.0
    psiterms = np.zeros(d)
    for i in range(d):
        psiterms[i] = psi((n - i - 1.0) / 2.0) / 2.0
    correction = d * dterm + psiterms.sum()
    return correction  # in nats


def test_digamma_equivalence():
    """Test if py_fast_digamma and scipy.special.psi are equivalent."""
    print("=" * 60)
    print("Testing digamma function equivalence")
    print("=" * 60)

    test_values = [5.5, 10.0, 50.0, 100.0, 500.0, 1000.0]

    print(f"{'x':>10} {'psi(x)':>15} {'py_fast_digamma(x)':>20} {'Diff':>15}")
    print("-" * 60)

    for x in test_values:
        psi_val = psi(x)
        fast_val = py_fast_digamma(x)
        diff = abs(psi_val - fast_val)
        print(f"{x:>10.1f} {psi_val:>15.10f} {fast_val:>20.10f} {diff:>15.2e}")

    print()


def test_univariate_equivalence():
    """Test if all three univariate patterns give the same result."""
    print("=" * 60)
    print("Testing univariate bias correction equivalence")
    print("=" * 60)

    test_n = [10, 50, 100, 500, 1000]

    print(f"{'n':>6} {'Pattern 1':>15} {'Pattern 2':>15} {'Pattern 3':>15} {'|P1-P3|':>12} {'|P2-P3|':>12}")
    print("-" * 80)

    for n in test_n:
        p1 = pattern1_univariate(n)
        p2 = pattern2_univariate(n)
        p3 = pattern3_univariate(n)
        diff_13 = abs(p1 - p3)
        diff_23 = abs(p2 - p3)
        print(f"{n:>6} {p1:>15.10f} {p2:>15.10f} {p3:>15.10f} {diff_13:>12.2e} {diff_23:>12.2e}")

    print()


def test_multivariate_equivalence():
    """Test if pattern 2 and 3 give the same result for multivariate case."""
    print("=" * 60)
    print("Testing multivariate bias correction equivalence")
    print("=" * 60)

    test_cases = [(10, 1), (10, 2), (10, 3), (100, 1), (100, 2), (100, 3), (1000, 2), (1000, 3)]

    print(f"{'n':>6} {'d':>3} {'Pattern 2 (fast)':>20} {'Pattern 3 (scipy)':>20} {'Difference':>15}")
    print("-" * 70)

    for n, d in test_cases:
        p2 = pattern2_multivariate(n, d)
        p3 = pattern3_multivariate(n, d)
        diff = abs(p2 - p3)
        print(f"{n:>6} {d:>3} {p2:>20.10f} {p3:>20.10f} {diff:>15.2e}")

    print()


if __name__ == "__main__":
    test_digamma_equivalence()
    test_univariate_equivalence()
    test_multivariate_equivalence()

    print("=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print("If differences are < 1e-8, the patterns are mathematically equivalent.")
    print("We should standardize on scipy.special.psi for reliability.")
    print()
