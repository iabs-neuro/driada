"""sys.path setup shared by the Figure 2 v8 scripts: DRIADA 1.2.2 of [driada] src first, then this
folder; figure_chain/ stays on sys.path for the read-only plotting helpers under
figure_chain/science/ (copies of the helpers of the DRIADA working tree)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # figure_chain/
import _paths_fig as _F  # noqa: E402

CODE = _F.P.DRIADA_SRC.parent
for p in (str(HERE), str(CODE / "src")):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import driada  # noqa: E402

if not Path(driada.__file__).resolve().is_relative_to(CODE):
    raise RuntimeError(f"driada imported from {driada.__file__}, not from [driada] src of the configuration")
