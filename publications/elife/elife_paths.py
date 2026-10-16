"""Locations shared by the scripts of the eLife paper.

Every script imports this module instead of building paths of its own, so the
data can live anywhere: set the environment variables below, or put the data
into the default places next to this file.

==========================  ====================================  ===========================
Variable                    Meaning                               Default
==========================  ====================================  ===========================
``DRIADA_ELIFE_DATA``       root of the experimental data         ``<this folder>/data``
``DRIADA_ELIFE_CACHE``      intermediate results of the figures   ``<this folder>/cache``
``DRIADA_ELIFE_OUTPUT``     figures, tables and logs              ``<this folder>/output``
``DRIADA_ELIFE_INTENSE``    output folder of the INTENSE run      ``<data root>/INTENSE``
``DRIADA_TOOLS``            ``tools`` folder of the library       ``<repository>/tools``
==========================  ====================================  ===========================

Expected layout of the data root::

    synchronized/<session>_aligned.npz        one file per session
    Mice_info.csv                             animal metadata
    INTENSE/by_type/results/                  written by compute/run_intense.py
    INTENSE/by_type/tables/
    INTENSE/by_type/tables_disentangled/
"""

import os
import sys

ELIFE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ELIFE_DIR, "..", ".."))

DATA_ROOT = os.environ.get("DRIADA_ELIFE_DATA", os.path.join(ELIFE_DIR, "data"))
CACHE_DIR = os.environ.get("DRIADA_ELIFE_CACHE", os.path.join(ELIFE_DIR, "cache"))
OUTPUT_DIR = os.environ.get("DRIADA_ELIFE_OUTPUT", os.path.join(ELIFE_DIR, "output"))
TOOLS_DIR = os.environ.get("DRIADA_TOOLS", os.path.join(REPO_ROOT, "tools"))

SYNC_DIR = os.path.join(DATA_ROOT, "synchronized")
INTENSE_ROOT = os.environ.get("DRIADA_ELIFE_INTENSE", os.path.join(DATA_ROOT, "INTENSE"))
# the type-based feature representations put their results into a subfolder
INTENSE_DIR = os.path.join(INTENSE_ROOT, "by_type", "results")
RAW_TABLES_DIR = os.path.join(INTENSE_ROOT, "by_type", "tables")
TABLES_DIR = os.path.join(INTENSE_ROOT, "by_type", "tables_disentangled")


def add_tools_to_path():
    """Make the helper packages of the library ``tools`` folder importable."""
    if TOOLS_DIR not in sys.path:
        sys.path.insert(0, TOOLS_DIR)


def out_path(name):
    """Path of an output file; the output directory is created on first use."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, name)
