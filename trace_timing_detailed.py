"""Detailed timing trace of the INTENSE pipeline."""
import numpy as np
import time
import sys
import os

sys.path.insert(0, 'src')

# Cache file path
CACHE_PATH = "benchmark_exp_cache.pickle"

from driada.experiment.synthetic import generate_tuned_selectivity_exp
from driada.experiment import save_exp_to_pickle, load_exp_from_pickle

print('=' * 70)
print('DETAILED TIMING TRACE')
print('=' * 70)

# Population configuration - same as benchmark
population = [
    {"name": "hd_cells", "count": 50, "features": ["head_direction"]},
    {"name": "speed_cells", "count": 50, "features": ["speed"]},
    {"name": "place_cells", "count": 50, "features": ["position_2d"]},
    {"name": "event0_cells", "count": 50, "features": ["event_0"]},
    {"name": "event1_cells", "count": 50, "features": ["event_1"]},
    {"name": "fbm_cells", "count": 50, "features": ["fbm_0"]},
    {"name": "mixed_cells", "count": 50, "features": ["head_direction", "event_0"]},
    {"name": "nonselective", "count": 150, "features": []},
]

# Load or generate experiment
if os.path.exists(CACHE_PATH):
    print('[1] Loading cached experiment...')
    exp = load_exp_from_pickle(CACHE_PATH, verbose=False)
else:
    print('[1] Generating experiment (this will be cached)...')
    exp = generate_tuned_selectivity_exp(
        population=population,
        duration=15 * 60,
        fps=20,
        n_discrete_features=5,
        seed=42,
        verbose=False,
    )
    save_exp_to_pickle(exp, CACHE_PATH, verbose=False)

print(f'  Neurons: {exp.n_cells}')
print(f'  Frames: {exp.n_frames}')

# Monkey-patch functions to trace timing
from driada.intense import intense_base
from driada.intense import stats as intense_stats

_timing = {
    'calculate_optimal_delays': 0,
    'scan_pairs_router_stage1': 0,
    'get_table_of_stats_stage1': 0,
    'criterion1_loop': 0,
    'scan_pairs_router_stage2': 0,
    'get_table_of_stats_stage2': 0,
    'criterion2_loop': 0,
    'other': 0,
}

# Patch calculate_optimal_delays_parallel
_orig_calculate_optimal_delays_parallel = intense_base.calculate_optimal_delays_parallel
def _traced_calculate_optimal_delays_parallel(*args, **kwargs):
    start = time.perf_counter()
    result = _orig_calculate_optimal_delays_parallel(*args, **kwargs)
    _timing['calculate_optimal_delays'] += time.perf_counter() - start
    return result
intense_base.calculate_optimal_delays_parallel = _traced_calculate_optimal_delays_parallel

# Patch scan_pairs_router
_orig_scan_pairs_router = intense_base.scan_pairs_router
_scan_pairs_call_count = [0]
def _traced_scan_pairs_router(*args, **kwargs):
    start = time.perf_counter()
    result = _orig_scan_pairs_router(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _scan_pairs_call_count[0] += 1
    if _scan_pairs_call_count[0] == 1:
        _timing['scan_pairs_router_stage1'] += elapsed
    else:
        _timing['scan_pairs_router_stage2'] += elapsed
    return result
intense_base.scan_pairs_router = _traced_scan_pairs_router

# Patch get_table_of_stats
_orig_get_table_of_stats = intense_stats.get_table_of_stats
_get_table_call_count = [0]
def _traced_get_table_of_stats(*args, **kwargs):
    start = time.perf_counter()
    result = _orig_get_table_of_stats(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _get_table_call_count[0] += 1
    if _get_table_call_count[0] == 1:
        _timing['get_table_of_stats_stage1'] += elapsed
    else:
        _timing['get_table_of_stats_stage2'] += elapsed
    return result
intense_stats.get_table_of_stats = _traced_get_table_of_stats

# Now run the pipeline
print('\n[2] Running INTENSE pipeline with tracing...')

import driada
from driada.information import MultiTimeSeries

feat_bunch = [f for f in exp.dynamic_features.keys() if f not in ['x', 'y']]
skip_delays = [
    fname for fname, fdata in exp.dynamic_features.items()
    if isinstance(fdata, MultiTimeSeries)
]

print(f'  Features: {feat_bunch}')
print(f'  Skip delays: {skip_delays}')

total_start = time.perf_counter()

stats, significance, info, results = driada.compute_cell_feat_significance(
    exp,
    feat_bunch=feat_bunch,
    mode="two_stage",
    n_shuffles_stage1=100,
    n_shuffles_stage2=10000,
    allow_mixed_dimensions=True,
    find_optimal_delays=True,
    skip_delays=skip_delays,
    ds=5,
    pval_thr=0.05,
    multicomp_correction="holm",
    use_precomputed_stats=False,
    with_disentanglement=False,
    verbose=True,
)

total_elapsed = time.perf_counter() - total_start

# Calculate 'other' time
traced_time = sum(_timing.values())
_timing['other'] = total_elapsed - traced_time

print('\n' + '=' * 70)
print('TIMING BREAKDOWN')
print('=' * 70)

print(f'\nTotal time: {total_elapsed:.1f}s')
print(f'\nComponent breakdown:')
for name, elapsed in sorted(_timing.items(), key=lambda x: -x[1]):
    pct = elapsed / total_elapsed * 100
    print(f'  {name:35s}: {elapsed:7.2f}s ({pct:5.1f}%)')

# Detailed analysis
print('\n' + '-' * 70)
print('Analysis:')
print('-' * 70)

n_neurons = exp.n_cells
n_features = len(feat_bunch)
n_pairs = n_neurons * n_features
print(f'Total pairs: {n_pairs} ({n_neurons} neurons x {n_features} features)')

# Calculate time per operation
if _timing['calculate_optimal_delays'] > 0:
    print(f'\nDelay optimization:')
    print(f'  Total: {_timing["calculate_optimal_delays"]:.2f}s')
    n_delay_pairs = n_neurons * (n_features - len(skip_delays))
    print(f'  Pairs with delays: {n_delay_pairs}')
    print(f'  Time per pair: {_timing["calculate_optimal_delays"]/n_delay_pairs*1000:.2f}ms')

if _timing['scan_pairs_router_stage1'] > 0:
    print(f'\nStage 1 scanning:')
    print(f'  Total: {_timing["scan_pairs_router_stage1"]:.2f}s')
    print(f'  Time per pair: {_timing["scan_pairs_router_stage1"]/n_pairs*1000:.2f}ms')

if _timing['get_table_of_stats_stage1'] > 0:
    print(f'\nStage 1 stats (gamma fitting + p-values):')
    print(f'  Total: {_timing["get_table_of_stats_stage1"]:.2f}s')
    print(f'  Time per pair: {_timing["get_table_of_stats_stage1"]/n_pairs*1000:.2f}ms')

if _timing['scan_pairs_router_stage2'] > 0:
    print(f'\nStage 2 scanning:')
    print(f'  Total: {_timing["scan_pairs_router_stage2"]:.2f}s')

if _timing['get_table_of_stats_stage2'] > 0:
    print(f'\nStage 2 stats:')
    print(f'  Total: {_timing["get_table_of_stats_stage2"]:.2f}s')

print('\n' + '=' * 70)
