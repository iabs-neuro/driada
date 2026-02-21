"""
Multiple comparison correction for INTENSE analysis.

Provides p-value threshold calculation for family-wise error rate (FWER) and
false discovery rate (FDR) control methods.
"""


def get_multicomp_correction_thr(fwer, mode="holm", **multicomp_kwargs) -> float:
    """
    Calculate p-value threshold for multiple hypothesis correction.

    Parameters
    ----------
    fwer : float
        Family-wise error rate or false discovery rate (e.g., 0.05).
        Must be between 0 and 1.
    mode : str or None, optional
        Multiple comparison correction method. Default: 'holm'.
        - None: No correction, threshold = fwer
        - 'bonferroni': Bonferroni correction (FWER control)
        - 'holm': Holm-Bonferroni correction (FWER control, more powerful)
        - 'fdr_bh': Benjamini-Hochberg FDR correction
    **multicomp_kwargs
        Additional arguments for correction method:
        - For 'bonferroni': nhyp (int) - number of hypotheses, must be > 0
        - For 'holm': all_pvals (list) - all p-values to be tested
        - For 'fdr_bh': all_pvals (list) - all p-values to be tested

    Returns
    -------
    threshold : float
        Adjusted p-value threshold for individual hypothesis testing.
        Returns 0 if no p-values pass the correction criteria (reject all).

    Raises
    ------
    ValueError
        If fwer is not between 0 and 1.
        If required arguments are missing or invalid.
        If unknown method specified.
        If nhyp <= 0 for bonferroni method.
        If all_pvals is empty for holm or fdr_bh methods.

    Notes
    -----
    - FWER methods (bonferroni, holm) control probability of ANY false positive
    - FDR methods control expected proportion of false positives among rejections
    - Holm is uniformly more powerful than Bonferroni
    - FDR typically allows more discoveries but with controlled false positive rate
    - If no p-values satisfy the correction criteria, threshold is set to 0
      (reject all hypotheses)

    Examples
    --------
    >>> # Compare different multiple comparison correction methods
    >>> pvals = [0.001, 0.01, 0.02, 0.03, 0.04]
    >>>
    >>> # No correction - uses raw threshold
    >>> thr_none = get_multicomp_correction_thr(0.05, mode=None)
    >>> thr_none
    0.05
    >>>
    >>> # Bonferroni correction - most conservative
    >>> thr_bonf = get_multicomp_correction_thr(0.05, mode='bonferroni', nhyp=5)
    >>> thr_bonf
    0.01
    >>>
    >>> # Holm correction - less conservative than Bonferroni
    >>> thr_holm = get_multicomp_correction_thr(0.05, mode='holm', all_pvals=pvals)
    >>> round(thr_holm, 4)
    0.0125
    >>>
    >>> # FDR correction - controls false discovery rate
    >>> thr_fdr = get_multicomp_correction_thr(0.05, mode='fdr_bh', all_pvals=pvals)
    >>> thr_fdr
    0.04
    >>>
    >>> # FDR is least conservative: bonf < holm < fdr < none
    >>> thr_bonf < thr_holm < thr_fdr < thr_none
    True"""
    # Validate fwer parameter
    if not 0 <= fwer <= 1:
        raise ValueError(f"fwer must be between 0 and 1, got {fwer}")

    if mode is None:
        threshold = fwer

    elif mode == "bonferroni":
        if "nhyp" not in multicomp_kwargs:
            raise ValueError("Number of hypotheses for Bonferroni correction not provided")
        nhyp = multicomp_kwargs["nhyp"]
        if nhyp <= 0:
            raise ValueError(f"Number of hypotheses must be positive, got {nhyp}")
        threshold = fwer / nhyp

    elif mode == "holm":
        if "all_pvals" not in multicomp_kwargs:
            raise ValueError("List of p-values for Holm correction not provided")
        all_pvals = multicomp_kwargs["all_pvals"]
        if len(all_pvals) == 0:
            raise ValueError("Empty p-value list provided for Holm correction")
        all_pvals = sorted(all_pvals)
        nhyp = len(all_pvals)
        threshold = 0  # Default if no discoveries (reject all)
        for i, pval in enumerate(all_pvals):
            cthr = fwer / (nhyp - i)
            if pval > cthr:
                break
            threshold = cthr

    elif mode == "fdr_bh":
        if "all_pvals" not in multicomp_kwargs:
            raise ValueError("List of p-values for FDR correction not provided")
        all_pvals = multicomp_kwargs["all_pvals"]
        if len(all_pvals) == 0:
            raise ValueError("Empty p-value list provided for FDR correction")
        all_pvals = sorted(all_pvals)
        nhyp = len(all_pvals)
        threshold = 0.0  # Default if no discoveries (reject all)

        # Benjamini-Hochberg procedure
        for i in range(nhyp - 1, -1, -1):
            if all_pvals[i] <= fwer * (i + 1) / nhyp:
                threshold = all_pvals[i]
                break

    else:
        raise ValueError("Unknown multiple comparisons correction method")

    return threshold
