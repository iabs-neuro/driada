"""INTENSE with fast Pearson correlation instead of MI on all NOF sessions (MI versus
correlation comparison of Results), before disentanglement.

Correlation needs one-dimensional features, so the representation is 'raw'
(original features); the MI run uses 'by_type'. Results go to <pearson_dir>.

usage: python compute/run_pearson.py [npz ...]   (no npz -> all sessions)
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

driada = P.use_driada()
print("driada", driada.__version__, driada.__file__, flush=True)
npz = sys.argv[1:]
sys.argv = [str(P.TOOLS_DIR / "run_intense_analysis.py"), *(npz or ["--dir", str(P.NOF_ALIGNED)]),
            "--output-dir", str(P.PEARSON_DIR), "--metric", "fast_pearsonr", "--representation", "raw",
            "--no-disentanglement", "--seed", str(P.SEED), "--skip-computed"]
runpy.run_path(sys.argv[0], run_name="__main__")
