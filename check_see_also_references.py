#!/usr/bin/env python
"""Check if functions referenced in 'See Also' sections are documented in RST files."""

import os
import re
from pathlib import Path
from collections import defaultdict
import fnmatch


def extract_see_also_references(file_path):
    """Extract function references from See Also sections in a Python file."""
    references = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find See Also sections
    see_also_pattern = r'See Also\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Examples|Parameters|Returns|Raises|Notes|References)|(?:\n\s*""")|$)'
    matches = re.finditer(see_also_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        see_also_content = match.group(1)
        # Extract references like ~driada.module.function
        ref_pattern = r'~(driada\.[a-zA-Z0-9_.]+)'
        refs = re.findall(ref_pattern, see_also_content)
        references.extend(refs)
    
    return references


def get_module_mapping():
    """Create a mapping between Python modules and their corresponding RST files."""
    mapping = {
        'driada.dim_reduction.neural': 'api/dim_reduction/neural_methods.rst',
        'driada.dim_reduction.flexible_ae': 'api/dim_reduction/neural_methods.rst',
        'driada.dim_reduction.losses': 'api/dim_reduction/neural_methods.rst',
        'driada.dim_reduction.graph': 'api/dim_reduction/data_structures.rst',
        'driada.dim_reduction.manifold_metrics': 'api/dim_reduction/manifold_metrics.rst',
        'driada.dim_reduction': 'api/dim_reduction.rst',
        
        'driada.information.info_base': 'api/information/core.rst',
        'driada.information.entropy': 'api/information/entropy.rst',
        'driada.information.gcmi': 'api/information/estimators.rst',
        'driada.information.ksg': 'api/information/estimators.rst',
        'driada.information.time_series_types': 'api/information/utilities.rst',
        'driada.information': 'api/information.rst',
        
        'driada.experiment.exp_base': 'api/experiment/core.rst',
        'driada.experiment.exp_build': 'api/experiment/loading.rst',
        'driada.experiment.synthetic.experiment_generators': 'api/experiment/synthetic.rst',
        'driada.experiment.wavelet_event_detection': 'api/experiment/wavelet.rst',
        'driada.experiment.wavelet_ridge': 'api/experiment/wavelet.rst',
        'driada.experiment': 'api/experiment.rst',
        
        'driada.intense.intense_base': 'api/intense/base.rst',
        'driada.intense.pipelines': 'api/intense/pipelines.rst',
        'driada.intense': 'api/intense.rst',
        
        'driada.integration.manifold_analysis': 'api/integration.rst',
        
        'driada.network.graph_utils': 'api/network/graph_utils.rst',
        'driada.network.matrix_utils': 'api/network/matrix_utils.rst',
        'driada.network.net_base': 'api/network/core.rst',
        'driada.network.quantum': 'api/network/quantum.rst',
        'driada.network.randomization': 'api/network/randomization.rst',
        'driada.network.spectral': 'api/network/spectral.rst',
        'driada.network': 'api/network.rst',
        
        'driada.rsa.core': 'api/rsa/core.rst',
        'driada.rsa.core_jit': 'api/rsa/core.rst',
        'driada.rsa.integration': 'api/rsa/integration.rst',
        'driada.rsa': 'api/rsa.rst',
        
        'driada.utils.data': 'api/utils/data.rst',
        'driada.utils.gif': 'api/utils/visualization.rst',
        'driada.utils.jit': 'api/utils/misc.rst',
        'driada.utils.matrix': 'api/utils/matrix.rst',
        'driada.utils.neural': 'api/utils/signals.rst',
        'driada.utils.plot': 'api/utils/visualization.rst',
        'driada.utils': 'api/utils.rst',
    }
    return mapping


def check_function_in_rst(function_ref, rst_path):
    """Check if a function reference is documented in the RST file."""
    if not os.path.exists(rst_path):
        return False, f"RST file not found: {rst_path}"
    
    # Extract just the function name from the full reference
    parts = function_ref.split('.')
    function_name = parts[-1]
    
    with open(rst_path, 'r') as f:
        content = f.read()
    
    # Check for various documentation patterns
    patterns = [
        f'.. autofunction:: {function_ref}',
        f'.. automethod:: {function_ref}',
        f'.. autoclass:: {function_ref}',
        f'~{function_ref}',
        function_name  # Sometimes just the function name is used
    ]
    
    for pattern in patterns:
        if pattern in content:
            return True, "Found"
    
    return False, f"Not found in {rst_path}"


def main():
    """Main function to check all See Also references."""
    src_dir = Path('/Users/nikita/PycharmProjects/driada2/src')
    docs_dir = Path('/Users/nikita/PycharmProjects/driada2/docs')
    
    # Get all Python files
    py_files = list(src_dir.glob('**/*.py'))
    
    # Extract all See Also references
    all_references = defaultdict(list)
    
    print("Extracting See Also references...")
    for py_file in py_files:
        refs = extract_see_also_references(py_file)
        for ref in refs:
            all_references[ref].append(str(py_file))
    
    # Check each reference
    module_mapping = get_module_mapping()
    missing_references = []
    
    print("\nChecking references in RST files...")
    for ref, source_files in sorted(all_references.items()):
        # Find the appropriate RST file
        module_parts = ref.split('.')
        
        # Try to find the best matching module
        rst_file = None
        for i in range(len(module_parts) - 1, 1, -1):
            module_path = '.'.join(module_parts[:i])
            if module_path in module_mapping:
                rst_file = docs_dir / module_mapping[module_path]
                break
        
        if rst_file:
            is_found, message = check_function_in_rst(ref, rst_file)
            if not is_found:
                missing_references.append({
                    'reference': ref,
                    'rst_file': str(rst_file),
                    'source_files': source_files,
                    'message': message
                })
        else:
            missing_references.append({
                'reference': ref,
                'rst_file': 'No mapping found',
                'source_files': source_files,
                'message': 'No RST mapping defined for this module'
            })
    
    # Report results
    print("\n" + "="*80)
    print("MISSING REFERENCES REPORT")
    print("="*80)
    
    if not missing_references:
        print("✓ All See Also references are properly documented!")
    else:
        print(f"\nFound {len(missing_references)} missing references:\n")
        
        # Group by RST file
        by_rst = defaultdict(list)
        for ref in missing_references:
            by_rst[ref['rst_file']].append(ref)
        
        for rst_file, refs in sorted(by_rst.items()):
            print(f"\n{rst_file}:")
            print("-" * len(rst_file))
            
            for ref in refs:
                print(f"\n  • {ref['reference']}")
                print(f"    Message: {ref['message']}")
                print(f"    Referenced in:")
                for src in ref['source_files'][:3]:  # Show max 3 source files
                    print(f"      - {src.replace('/Users/nikita/PycharmProjects/driada2/', '')}")
                if len(ref['source_files']) > 3:
                    print(f"      ... and {len(ref['source_files']) - 3} more files")


if __name__ == '__main__':
    main()