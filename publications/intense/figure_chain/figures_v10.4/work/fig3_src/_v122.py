"""Puts the DRIADA v1.2.2 snapshot first on sys.path and names the data folders used by Figure 3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # figure_chain/
import _paths_fig as _F  # noqa: E402

CODE = _F.P.DRIADA_SRC.parent
for p in (CODE / "tools", CODE / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import driada  # noqa: E402

assert Path(driada.__file__).resolve().is_relative_to(CODE), driada.__file__

RAW = _F.V.get("nof_v1")


def aligned(mouse, session):
    return RAW / f"NOF_{mouse}_{session}_aligned.npz"


def load_session(mouse, session):
    """Experiment exactly as the analysis tool builds it (same loader, seed 42)."""
    import warnings
    from selectivity_dynamics import load_experiment_from_npz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_experiment_from_npz(aligned(mouse, session), verbose=False, seed=42)
