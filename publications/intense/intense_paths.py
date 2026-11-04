"""Locations and settings shared by the scripts of the INTENSE paper.

Every script imports this module instead of building paths of its own. The
values come from ``config.toml`` next to this file, or from the file named by
the environment variable ``INTENSE_PAPER_CONFIG``.

A location in the file may start with ``${INTENSE_DATA}`` (the downloaded
inputs, default ``<this folder>/data``) or ``${INTENSE_RESULTS}`` (everything
the scripts write, default ``<this folder>/results``); both roots can be moved
by setting the environment variables of the same names. Other environment
variables in a value are expanded as well, and a relative path is resolved
against the folder of the configuration file.

Importing the module also puts the DRIADA release named in the configuration
first on ``sys.path`` and checks its version, so that no script silently runs
on another version of the library.
"""

import os
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parents[1]
CONFIG_FILE = Path(os.environ.get("INTENSE_PAPER_CONFIG", PAPER_DIR / "config.toml")).resolve()
DATA_ROOT = Path(os.environ.get("INTENSE_DATA") or PAPER_DIR / "data").resolve()
RESULTS_ROOT = Path(os.environ.get("INTENSE_RESULTS") or PAPER_DIR / "results").resolve()

with open(CONFIG_FILE, "rb") as _f:
    CFG = tomllib.load(_f)


def _path(value, default=None):
    if not value:
        return default
    value = str(value).replace("${INTENSE_DATA}", str(DATA_ROOT)).replace("${INTENSE_RESULTS}", str(RESULTS_ROOT))
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    return p if p.is_absolute() else (CONFIG_FILE.parent / p).resolve()


def get(section, key):
    """Location ``[section] key`` of the configuration, resolved as above.

    Raises KeyError naming the configuration file when the entry is missing or empty.
    """
    value = CFG.get(section, {}).get(key)
    if not value:
        raise KeyError(f"[{section}] {key} is not set in {CONFIG_FILE}")
    return _path(value)


def setting(section, key, default=None):
    """A non-path value of the configuration (a number, a name), or ``default``."""
    return CFG.get(section, {}).get(key, default)


# --- analysis settings -------------------------------------------------------
REPRESENTATION = CFG["analysis"]["representation"]
SEED = int(CFG["analysis"]["seed"])
FPS = float(CFG["analysis"]["fps"])
PVAL_THR = float(CFG["analysis"]["pval_thr"])
MI_EXCESS_THR = float(CFG["analysis"]["mi_excess_thr"])
ZONE_RULE = CFG["analysis"]["zone_rule"]

# --- DRIADA ------------------------------------------------------------------
DRIADA_SRC = _path(CFG["driada"].get("src"))
TOOLS_DIR = _path(CFG["driada"].get("tools"), REPO_ROOT / "tools")
DRIADA_VERSION = CFG["driada"]["version"]

# --- data and results --------------------------------------------------------
NOF_ALIGNED = _path(CFG["data"]["nof_aligned"])
NOF_LEGACY = _path(CFG["data"].get("nof_legacy"))
ZONE_MASKS = _path(CFG["data"].get("zone_masks"))
CLASSIC_MAT = _path(CFG["data"]["classic_mat"])

INTENSE_ROOT = _path(CFG["results"]["intense_root"])
# the type-based feature representations put their results into a subfolder
INTENSE_RUN = INTENSE_ROOT / REPRESENTATION if REPRESENTATION == "by_type" else INTENSE_ROOT
INTENSE_RESULTS = INTENSE_RUN / "results"
RAW_TABLES = INTENSE_RUN / "tables"
DISENT_TABLES = INTENSE_RUN / "tables_disentangled"
DISENT_DIR = INTENSE_RUN / "disentanglement"
DB_DIR = _path(CFG["results"]["db_dir"])
DB_PRE = DB_DIR / "NOF_pre_disent.csv"
DB_POST = DB_DIR / "NOF_post_disent.csv"
NUMBERS_DIR = _path(CFG["results"]["numbers_dir"])
PEARSON_DIR = _path(CFG["results"].get("pearson_dir"))
CALIBRATION_DIR = _path(CFG["results"]["calibration_dir"])

# --- figures -----------------------------------------------------------------
CACHE_DIR = _path(CFG["figures"]["cache_dir"])
OUTPUT_DIR = _path(CFG["figures"]["output_dir"])
SOURCE_DATA_DIR = OUTPUT_DIR / "source_data"
FIG1_SCHEME = _path(CFG["figures"]["fig1_scheme"])
FIG3_MOUSE_A = _path(CFG["figures"]["fig3_mouse_a"])
FIG3_MOUSE_B = _path(CFG["figures"]["fig3_mouse_b"])

# 16 mice x 4 days, in the order of the classic .mat file
MICE = ["H01", "H02", "H03", "H06", "H07", "H08", "H09", "H14", "H23", "H26",
        "H27", "H31", "H32", "H33", "H36", "H39"]
SESSIONS = [f"{m}_{d}D" for d in range(1, 5) for m in MICE]


def aligned_file(mouse, day):
    """Synchronized session file, e.g. ``aligned_file('H01', '2D')``."""
    return NOF_ALIGNED / f"NOF_{mouse}_{day}_aligned.npz"


def legacy_file(mouse, day):
    """Earlier export of a session (one extra leading calcium row, see config.toml)."""
    return NOF_LEGACY / f"NOF_{mouse}_{day} syn data.npz"


def cache_path(name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def out_path(name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def use_driada(tools=True):
    """Put the configured DRIADA release (and its tools/) first on sys.path and check it."""
    for p in ([TOOLS_DIR] if tools else []) + ([DRIADA_SRC] if DRIADA_SRC else []):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    import driada

    where = Path(driada.__file__).resolve()
    if DRIADA_SRC and not where.is_relative_to(DRIADA_SRC):
        raise RuntimeError(f"driada imported from {where}, not from {DRIADA_SRC}")
    if driada.__version__ != DRIADA_VERSION:
        raise RuntimeError(f"driada {driada.__version__} imported, the paper needs {DRIADA_VERSION}")
    return driada


def check_representation(feature_names):
    """Fail if INTENSE results were computed with another representation than configured.

    The results file does not record the representation, but its feature names
    do: by_type replaces place, speed and directions by ``place_quad``,
    ``speed_quad`` and ``*_harm2``; raw keeps the original names.
    """
    typed = any(str(f).endswith(("_quad", "_harm2")) for f in feature_names)
    if typed != (REPRESENTATION == "by_type"):
        raise RuntimeError(f"results look like representation={'by_type' if typed else 'raw'!r}, "
                           f"config says {REPRESENTATION!r}")
