#!/usr/bin/env python
"""Summarize missing references by module and check if functions exist in the code."""

import os
import re
import ast
from pathlib import Path
from collections import defaultdict


def get_function_definitions(py_file):
    """Extract all function and method definitions from a Python file."""
    functions = set()
    
    try:
        with open(py_file, 'r') as f:
            tree = ast.parse(f.read(), filename=str(py_file))
        
        module_name = get_module_name(py_file)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Top-level function
                if hasattr(node, 'col_offset') and node.col_offset == 0:
                    functions.add(f"{module_name}.{node.name}")
            elif isinstance(node, ast.ClassDef):
                # Class methods
                class_name = node.name
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        functions.add(f"{module_name}.{class_name}.{item.name}")
    except:
        pass
    
    return functions


def get_module_name(py_file):
    """Convert file path to module name."""
    parts = Path(py_file).parts
    src_idx = parts.index('src')
    module_parts = parts[src_idx+1:-1] + (Path(py_file).stem,)
    return '.'.join(module_parts)


def main():
    """Analyze missing references."""
    src_dir = Path('/Users/nikita/PycharmProjects/driada2/src')
    
    # Get all function definitions
    all_functions = set()
    for py_file in src_dir.glob('**/*.py'):
        all_functions.update(get_function_definitions(py_file))
    
    # Missing references from previous analysis
    missing_refs = [
        'driada.dim_reduction.neural.get_code',
        'driada.dim_reduction.neural.reparameterization',
        'driada.experiment.exp_base._update_stats_and_significance',
        'driada.experiment.exp_base.compute_rdm',
        'driada.experiment.exp_base.get_embedding',
        'driada.experiment.exp_base.get_significant_neurons',
        'driada.experiment.exp_base.get_stats_slice',
        'driada.experiment.exp_base.store_embedding',
        'driada.experiment.wavelet_event_detection.get_cwt_ridges_fast',
        'driada.experiment.wavelet_event_detection.passing_criterion',
        'driada.information.info_base.get_kdtree',
        'driada.information.info_base.get_kdtree_query',
        'driada.information.gcmi.gcmi_cc',
        'driada.information.gcmi.mi_gg',
        'driada.information.gcmi.mi_mixture_gd',
        'driada.information.ksg.get_lnc_alpha',
        'driada.information.time_series_types._detect_circular',
        'driada.information.time_series_types._detect_primary_type',
        'driada.intense.intense_base.update',
        'driada.network.net_base.calculate_free_entropy',
        'driada.network.net_base.calculate_thermodynamic_entropy',
        'driada.network.net_base.get_ipr',
        'driada.network.net_base.get_z_values',
        'driada.network.quantum.js_divergence',
        'driada.network.quantum.renyi_divergence',
        'driada.network.randomization.random_rewiring_complete_graph',
        'driada.network.spectral.free_entropy',
        'driada.network.spectral.q_entropy',
        'driada.rsa.core.compare_rdms',
        'driada.rsa.core.compute_rdm',
        'driada.rsa.core.compute_rdm_from_timeseries_labels',
        'driada.rsa.core.compute_rdm_unified',
        'driada.rsa.integration.compute_experiment_rdm',
        'driada.utils.data.write_dict_to_hdf5',
        'driada.utils.matrix.is_positive_definite',
        'driada.utils.matrix.nearestPD',
        'driada.utils.jit.is_jit_enabled',
        'driada.utils.jit.jit_info',
        'driada.utils.neural.generate_pseudo_calcium_multisignal',
        'driada.utils.neural.generate_pseudo_calcium_signal',
        'driada.utils.gif.create_gif_from_image_series',
        'driada.utils.gif.save_image_series',
        'driada.utils.plot.create_default_figure',
        'driada.utils.plot.make_beautiful',
    ]
    
    # Check which ones exist
    print("MISSING REFERENCES ANALYSIS")
    print("=" * 80)
    print("\nChecking if referenced functions actually exist in the code...\n")
    
    by_module = defaultdict(list)
    for ref in missing_refs:
        exists = False
        # Check exact match
        if ref in all_functions:
            exists = True
        else:
            # Check if it's a method reference
            parts = ref.split('.')
            if len(parts) >= 4:  # module.submodule.Class.method
                # Try without the submodule
                alt_ref = '.'.join(parts[:2] + parts[3:])
                if alt_ref in all_functions:
                    exists = True
        
        by_module['.'.join(ref.split('.')[:3])].append((ref, exists))
    
    # Report by module
    for module, refs in sorted(by_module.items()):
        print(f"\n{module}:")
        print("-" * len(module))
        
        existing = [(r, e) for r, e in refs if e]
        missing = [(r, e) for r, e in refs if not e]
        
        if existing:
            print("\n  Functions that EXIST but are NOT documented:")
            for ref, _ in existing:
                func_name = ref.split('.')[-1]
                print(f"    ✓ {func_name}")
        
        if missing:
            print("\n  Functions that DON'T EXIST (possible errors in See Also):")
            for ref, _ in missing:
                func_name = ref.split('.')[-1]
                print(f"    ✗ {func_name}")
    
    # Summary
    total_existing = sum(1 for _, refs in by_module.items() for r, e in refs if e)
    total_missing = sum(1 for _, refs in by_module.items() for r, e in refs if not e)
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  • {total_existing} functions exist but are not documented in RST files")
    print(f"  • {total_missing} functions don't exist (incorrect See Also references)")
    print(f"  • Total: {total_existing + total_missing} issues to fix")


if __name__ == '__main__':
    main()