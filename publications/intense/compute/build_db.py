"""Tidy NOF databases from the INTENSE run, before and after disentanglement.

Uses tools/neuron_database of the configured DRIADA release on the tables of
<intense_root>/<representation>. The post-disentanglement file equals the
NOF_database.csv of the cross-session analysis.

usage: python compute/build_db.py  ->  <db_dir>/NOF_pre_disent.csv, NOF_post_disent.csv
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import intense_paths as P  # noqa: E402

P.use_driada()
from neuron_database import EXPERIMENT_CONFIGS, load_experiment  # noqa: E402

P.DB_DIR.mkdir(parents=True, exist_ok=True)
for tables, name in [(P.RAW_TABLES, "NOF_pre_disent"), (P.DISENT_TABLES, "NOF_post_disent")]:
    cfg = replace(EXPERIMENT_CONFIGS["NOF"], tables_subdir=str(tables))
    # the experiment folder holds the synchronized sessions and the matching files
    db = load_experiment("NOF", str(P.NOF_ALIGNED.parent), config=cfg)
    db.data.drop(columns="significant").to_csv(P.DB_DIR / f"{name}.csv", index=False)
    print("[OK]", name, db.data.shape, "neuron-sessions:", db.n_neurons())
