"""Locations for the figure scripts of the paper (figure_chain/).

The figure scripts were written version on version of the manuscript (figures_v10.4/work ->
figures_v10.5/work -> figures_v10.6/work -> figures_v11/work -> figures_v11.1/work), each version importing
the one before it and reading what lies next to it. Here the code keeps the same relative layout, so the
imports between versions are unchanged, while:

* everything a script writes goes to the same relative folder under [figures] output_dir/figure_chain:
  out_dir(folder);
* what a script reads from another script of the chain comes from the same relative folder under
  [v11] figure_artifacts, by default the same place: data_dir(folder);
* the small inputs of the example panels that cannot be rebuilt from the deposit lie in
  figure_chain/saved_inputs (README.md): SAVED;
* the results of the analysis come from the configuration ([figures] cache_dir, [results] db_dir,
  [data] nof_aligned, [v11] fps_table, ...);
* figures of earlier manuscript versions, with which some scripts compare their output, are optional:
  reference(path).
"""
import sys
from pathlib import Path

CHAIN = Path(__file__).resolve().parent
if str(CHAIN.parent) not in sys.path:
    sys.path.insert(0, str(CHAIN.parent))
import intense_paths as P  # noqa: E402
import paths_v11 as V  # noqa: E402

D = V.get("figure_artifacts")
OUT = P.OUTPUT_DIR / "figure_chain"
SAVED = CHAIN / "saved_inputs"


def _rel(folder):
    return Path(folder).resolve().relative_to(CHAIN)


def data_dir(folder):
    """Folder of the inputs of a code folder of figure_chain (same relative path under figure_artifacts)."""
    return D / _rel(folder)


def out_dir(folder):
    """Folder for what the scripts of a code folder write (created)."""
    p = OUT / _rel(folder)
    p.mkdir(parents=True, exist_ok=True)
    return p


def reference(path):
    """``path`` if the file to compare with is there; otherwise a printed note and None."""
    if Path(path).exists():
        return Path(path)
    print(f"[SKIPPED] comparison with {Path(path).name}: the file is not there", flush=True)
    return None
