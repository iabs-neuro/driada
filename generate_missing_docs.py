#!/usr/bin/env python
"""Generate documentation additions for missing See Also references."""

import os
from pathlib import Path


def generate_documentation_fixes():
    """Generate the RST file additions needed to document missing functions."""
    
    fixes = {
        'api/experiment/wavelet.rst': [
            'driada.experiment.wavelet_event_detection.get_cwt_ridges_fast',
            'driada.experiment.wavelet_event_detection.passing_criterion',
        ],
        'api/information/estimators.rst': [
            'driada.information.gcmi.gcmi_cc',
            'driada.information.gcmi.mi_gg',
            'driada.information.ksg.get_lnc_alpha',
        ],
        'api/information/utilities.rst': [
            'driada.information.time_series_types._detect_circular',
            'driada.information.time_series_types._detect_primary_type',
        ],
        'api/network/quantum.rst': [
            'driada.network.quantum.js_divergence',
            'driada.network.quantum.renyi_divergence',
        ],
        'api/network/randomization.rst': [
            'driada.network.randomization.random_rewiring_complete_graph',
        ],
        'api/network/spectral.rst': [
            'driada.network.spectral.free_entropy',
            'driada.network.spectral.q_entropy',
        ],
        'api/rsa/core.rst': [
            'driada.rsa.core.compare_rdms',
            'driada.rsa.core.compute_rdm',
            'driada.rsa.core.compute_rdm_from_timeseries_labels',
            'driada.rsa.core.compute_rdm_unified',
        ],
        'api/rsa/integration.rst': [
            'driada.rsa.integration.compute_experiment_rdm',
        ],
        'api/utils/data.rst': [
            'driada.utils.data.write_dict_to_hdf5',
        ],
        'api/utils/visualization.rst': [
            'driada.utils.gif.create_gif_from_image_series',
            'driada.utils.gif.save_image_series',
            'driada.utils.plot.create_default_figure',
            'driada.utils.plot.make_beautiful',
        ],
        'api/utils/misc.rst': [
            'driada.utils.jit.is_jit_enabled',
            'driada.utils.jit.jit_info',
        ],
        'api/utils/matrix.rst': [
            'driada.utils.matrix.is_positive_definite',
            'driada.utils.matrix.nearestPD',
        ],
        'api/utils/signals.rst': [
            'driada.utils.neural.generate_pseudo_calcium_multisignal',
            'driada.utils.neural.generate_pseudo_calcium_signal',
        ],
    }
    
    print("DOCUMENTATION FIXES NEEDED")
    print("=" * 80)
    print("\nFor each RST file, add the following autofunction directives:\n")
    
    docs_dir = Path('/Users/nikita/PycharmProjects/driada2/docs')
    
    for rst_file, functions in sorted(fixes.items()):
        full_path = docs_dir / rst_file
        print(f"\n{rst_file}:")
        print("-" * len(rst_file))
        
        # Check current content
        if full_path.exists():
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Find where to add functions
            if "Main Functions" in content or "Functions" in content:
                print("Add to existing Functions section:")
            else:
                print("Create new Functions section before any Examples:")
                print("\nFunctions")
                print("---------")
        
        print()
        for func in sorted(functions):
            print(f".. autofunction:: {func}")
        
        # Special notes for private functions
        private_funcs = [f for f in functions if '._' in f]
        if private_funcs:
            print("\nNote: These are private functions (start with _):")
            for func in private_funcs:
                print(f"  - {func}")
            print("Consider whether they should be in See Also or made public.")


def generate_see_also_fixes():
    """Generate fixes for incorrect See Also references."""
    
    incorrect_refs = {
        'driada/dim_reduction/neural.py': [
            ('driada.dim_reduction.neural.get_code', 'Method of VAE class, use VAE.get_code'),
            ('driada.dim_reduction.neural.reparameterization', 'Method of VAEEncoder, use VAEEncoder.reparameterization'),
        ],
        'driada/experiment/exp_base.py': [
            ('driada.experiment.exp_base._update_stats_and_significance', 'Private method, remove from See Also'),
            ('driada.experiment.exp_base.compute_rdm', 'Method of ExperimentBase, use ExperimentBase.compute_rdm'),
            ('driada.experiment.exp_base.get_embedding', 'Method of ExperimentBase, use ExperimentBase.get_embedding'),
            ('driada.experiment.exp_base.get_significant_neurons', 'Method of ExperimentBase, use ExperimentBase.get_significant_neurons'),
            ('driada.experiment.exp_base.get_stats_slice', 'Method of ExperimentBase, use ExperimentBase.get_stats_slice'),
            ('driada.experiment.exp_base.store_embedding', 'Method of ExperimentBase, use ExperimentBase.store_embedding'),
        ],
        'driada/information/gcmi.py': [
            ('driada.information.gcmi.mi_mixture_gd', 'Function does not exist, remove from See Also'),
        ],
        'driada/information/info_base.py': [
            ('driada.information.info_base.get_kdtree', 'Function does not exist, remove from See Also'),
            ('driada.information.info_base.get_kdtree_query', 'Function does not exist, remove from See Also'),
        ],
        'driada/intense/intense_base.py': [
            ('driada.intense.intense_base.update', 'Function does not exist, remove from See Also'),
        ],
        'driada/network/net_base.py': [
            ('driada.network.net_base.calculate_free_entropy', 'Use driada.network.spectral.free_entropy instead'),
            ('driada.network.net_base.calculate_thermodynamic_entropy', 'Use driada.network.spectral.spectral_entropy instead'),
            ('driada.network.net_base.get_ipr', 'Method of Network class, use Network.get_ipr'),
            ('driada.network.net_base.get_z_values', 'Method of Network class, use Network.get_z_values'),
        ],
    }
    
    print("\n\nSEE ALSO FIXES NEEDED")
    print("=" * 80)
    print("\nFor each Python file, fix these incorrect See Also references:\n")
    
    for py_file, fixes in sorted(incorrect_refs.items()):
        print(f"\n{py_file}:")
        print("-" * len(py_file))
        
        for ref, fix in fixes:
            print(f"\n  • {ref}")
            print(f"    Fix: {fix}")


def main():
    """Main function."""
    generate_documentation_fixes()
    generate_see_also_fixes()
    
    print("\n\nSUMMARY")
    print("=" * 80)
    print("\n1. Add 28 missing function documentations to RST files")
    print("2. Fix 16 incorrect See Also references in Python files")
    print("3. Consider making private functions public if they're meant to be used")
    print("4. Update See Also sections to use correct class.method notation")


if __name__ == '__main__':
    main()