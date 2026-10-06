"""Add the chance level to INTENSE results saved without it.

Saved results keep the shuffled values of every pair, so ``me_null`` (their
mean) and ``me_excess = me - me_null`` can be recovered without recomputing
anything. The tables are re-exported into a new folder; the source is only
read.

Usage:
    python backfill_chance_level.py <intense_output_dir> <new_output_dir>
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from neuron_database.loaders import _parse_dict_cell  # noqa: E402

TABLE_DIRS = ('tables', 'tables_disentangled')

# Shuffled values are stored in half precision, the statistics in single.
_HALF_PRECISION_RTOL = 1e-3


def _stat_key(feat_id):
    """Name of a feature in the saved statistics (tuples are saved as their text)."""
    return feat_id if isinstance(feat_id, str) else str(tuple(feat_id))


def _stage_values(data, stage):
    """Saved metric values of one stage as {(row, column): values}, or None."""
    key = f'info_me_total{stage}'
    if f'{key}_data' not in data.files:
        return None
    values = data[f'{key}_data']
    return {(int(i), int(j)): values[k] for k, (i, j) in enumerate(data[f'{key}_indices'])}


def chance_levels_from_results(npz_path):
    """Chance level of every pair of a saved INTENSE result.

    Parameters
    ----------
    npz_path : str or Path
        Result file written by ``driada.intense.io.save_results``.

    Returns
    -------
    dict
        ``{(neuron, feature): {'me_null': float, 'me_excess': float}}`` for
        every pair that has a metric value. The shuffles of stage 2 are used
        for pairs that reached it, the shuffles of stage 1 otherwise.

    Raises
    ------
    ValueError
        If the file holds no shuffled values, or the saved values of a pair
        do not match its statistics.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        stats = json.loads(str(data['_stats_json'][0]))
        params = json.loads(str(data['_params_json'][0]))['intense_params']
        stage_values = {stage: _stage_values(data, stage) for stage in (1, 2)}
    if stage_values[1] is None and stage_values[2] is None:
        raise ValueError(f'{npz_path}: no saved shuffles (info_me_total1/2)')

    levels = {}
    for i, cell_id in params['neurons'].items():
        cell_stats = stats.get(str(cell_id), {})
        for j, feat_id in params['feat_bunch'].items():
            feat = _stat_key(feat_id)
            pair = cell_stats.get(feat, {})
            me = pair.get('me')
            if me is None:
                continue
            stage = 2 if pair.get('rval') is not None else 1
            values = (stage_values[stage] or {}).get((int(i), int(j)))
            if values is None:
                raise ValueError(f'{npz_path}: no stage {stage} shuffles for neuron {cell_id}, {feat}')
            if abs(float(values[0]) - me) > _HALF_PRECISION_RTOL * max(abs(me), 1e-3):
                raise ValueError(
                    f'{npz_path}: saved shuffles of neuron {cell_id}, {feat} do not match '
                    f'its statistics ({float(values[0]):.6g} vs me={me:.6g})'
                )
            me_null = float(values[1:].mean(dtype=np.float64))
            levels[(str(cell_id), feat)] = {'me_null': me_null, 'me_excess': me - me_null}
    return levels


def _resolve_component(name, features):
    """Saved feature behind a component name of a merged entry, or None.

    A merged 'place-<zone>' entry may name the position feature without the
    suffix of its representation ('place' for 'place_quad').
    """
    if name in features:
        return name
    candidates = [feat for feat in features if feat.startswith(name + '_')]
    return candidates[0] if len(candidates) == 1 else None


def _backfill_entry(entry, cell, feature, levels, features, where):
    """Add the chance level to one table cell (a statistics dict), in place."""
    if (cell, feature) in levels:
        entry.update(levels[(cell, feature)])
        return
    parts = [_resolve_component(part, features) for part in feature.split('-', 1)]
    if len(parts) != 2 or None in parts or any((cell, part) not in levels for part in parts):
        raise ValueError(f'{where}: no chance level for neuron {cell}, {feature}')
    # The merged entry reports the component with the larger metric value.
    first, second = (levels[(cell, part)] for part in parts)
    me_first, me_second = (lev['me_null'] + lev['me_excess'] for lev in (first, second))
    entry.update(first if me_first >= me_second else second)
    components = entry.get('component_stats')
    if isinstance(components, dict):
        for component, level in zip(components.values(), (first, second)):
            component.update(level)


def backfill_stats_table(src_csv, dst_csv, levels):
    """Write a copy of a statistics table with the chance level in every cell.

    Parameters
    ----------
    src_csv, dst_csv : str or Path
        Source table and the new table.
    levels : dict
        Output of ``chance_levels_from_results`` for the same session.

    Returns
    -------
    int
        Number of cells that received a chance level.
    """
    table = pd.read_csv(src_csv, index_col=0, dtype=str)
    features = {feat for _, feat in levels}
    n_filled = 0
    for cell in table.index:
        for feature in table.columns:
            entry = _parse_dict_cell(table.at[cell, feature])
            if entry is None:
                table.at[cell, feature] = repr({})
                continue
            _backfill_entry(entry, str(cell), feature, levels, features, src_csv)
            table.at[cell, feature] = repr(entry)
            n_filled += 1
    table.index.name = None
    table.to_csv(dst_csv)
    return n_filled


def backfill_output_dir(src_dir, out_dir, verbose=False):
    """Re-export the tables of an INTENSE output folder with chance levels.

    Parameters
    ----------
    src_dir : str or Path
        Output folder of the INTENSE runner: ``results/<session>_results.npz``
        with ``tables/`` and, optionally, ``tables_disentangled/``.
    out_dir : str or Path
        New folder for the tables. Must differ from ``src_dir``; existing
        tables in it are not overwritten.
    verbose : bool
        Print one line per session.

    Returns
    -------
    int
        Number of sessions processed.
    """
    src_dir, out_dir = Path(src_dir).resolve(), Path(out_dir).resolve()
    if out_dir == src_dir:
        raise ValueError('The tables are written to a new folder, not in place')
    result_files = sorted((src_dir / 'results').glob('*_results.npz'))
    if not result_files:
        raise FileNotFoundError(f'No *_results.npz in {src_dir / "results"}')
    table_dirs = [name for name in TABLE_DIRS if (src_dir / name).is_dir()]
    for name in table_dirs:
        if (out_dir / name).exists() and any((out_dir / name).iterdir()):
            raise FileExistsError(f'{out_dir / name} already holds tables')
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    for npz_path in result_files:
        session = npz_path.name[:-len('_results.npz')]
        levels = chance_levels_from_results(npz_path)
        for name in table_dirs:
            stats_csv = src_dir / name / f'{session} INTENSE stats.csv'
            if not stats_csv.exists():
                continue
            n_filled = backfill_stats_table(stats_csv, out_dir / name / stats_csv.name, levels)
            sig_csv = src_dir / name / f'{session} INTENSE significance.csv'
            if sig_csv.exists():
                shutil.copyfile(sig_csv, out_dir / name / sig_csv.name)
            if verbose:
                print(f'  {session} {name}: {n_filled} cells')
    return len(result_files)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    n_sessions = backfill_output_dir(sys.argv[1], sys.argv[2], verbose=True)
    print(f'[OK] sessions: {n_sessions}')
