#!/usr/bin/env python3
"""
Verify that all function/class references in documentation files
actually exist in the DRIADA codebase.

This script scans all .rst files for references like:
- :func:`~driada.module.function`
- :class:`~driada.module.Class`
- :meth:`~driada.module.Class.method`

And verifies each reference can be imported and accessed.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import importlib
from collections import defaultdict
from unittest.mock import MagicMock

# Add src to path to import driada
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock torch for documentation verification on systems without PyTorch
try:
    import torch
except ImportError:
    # Create comprehensive torch mock
    torch_mock = MagicMock()
    torch_mock.nn = MagicMock()
    torch_mock.nn.Module = MagicMock
    torch_mock.nn.functional = MagicMock()
    torch_mock.utils = MagicMock()
    torch_mock.utils.data = MagicMock()
    torch_mock.utils.data.Dataset = MagicMock
    torch_mock.optim = MagicMock()
    torch_mock.Tensor = MagicMock
    torch_mock.device = MagicMock
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    
    # Install the mock
    sys.modules['torch'] = torch_mock
    sys.modules['torch.nn'] = torch_mock.nn
    sys.modules['torch.nn.functional'] = torch_mock.nn.functional
    sys.modules['torch.utils'] = torch_mock.utils
    sys.modules['torch.utils.data'] = torch_mock.utils.data
    sys.modules['torch.optim'] = torch_mock.optim


def find_rst_files(docs_dir: Path) -> List[Path]:
    """Find all .rst files in the documentation directory."""
    return list(docs_dir.rglob("*.rst"))


def extract_references(file_path: Path) -> List[Tuple[int, str, str]]:
    """Extract all function/class references from an RST file.
    
    Returns:
        List of (line_number, ref_type, reference) tuples
    """
    references = []
    
    # Patterns to match Sphinx references
    patterns = [
        (r':func:`~?(driada\.[^`]+)`', 'func'),
        (r':class:`~?(driada\.[^`]+)`', 'class'),
        (r':meth:`~?(driada\.[^`]+)`', 'meth'),
        (r'.. autofunction:: (driada\.[^\s]+)', 'autofunction'),
        (r'.. autoclass:: (driada\.[^\s]+)', 'autoclass'),
        (r'.. automethod:: (driada\.[^\s]+)', 'automethod'),
    ]
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            for pattern, ref_type in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    references.append((line_num, ref_type, match))
    
    return references


def verify_reference(reference: str) -> Tuple[bool, str]:
    """Verify if a reference exists in the codebase.
    
    Returns:
        (exists, error_message)
    """
    try:
        # Split the reference into parts
        parts = reference.split('.')
        if len(parts) < 3:  # At least driada.module.item
            return False, f"Invalid reference format: {reference}"
        
        # Try to import the module
        module_path = '.'.join(parts[:-1])
        item_name = parts[-1]
        
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            return False, f"Cannot import module {module_path}: {e}"
        
        # Check if the item exists in the module
        if hasattr(module, item_name):
            return True, ""
        
        # Check if it's exported in __all__
        if hasattr(module, '__all__') and item_name in module.__all__:
            return True, ""
        
        # For methods, check if it's a method of a class
        if '.' in item_name:  # Could be Class.method
            class_name, method_name = item_name.split('.', 1)
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                if hasattr(cls, method_name):
                    return True, ""
        
        return False, f"'{item_name}' not found in {module_path}"
        
    except Exception as e:
        return False, f"Error verifying {reference}: {e}"


def check_module_exports(module_name: str) -> Set[str]:
    """Get all exported items from a module."""
    try:
        module = importlib.import_module(module_name)
        
        # Get __all__ if it exists
        if hasattr(module, '__all__'):
            exports = set(module.__all__)
        else:
            # Get all public attributes
            exports = {name for name in dir(module) if not name.startswith('_')}
        
        return exports
    except ImportError:
        return set()


def main():
    """Main verification function."""
    # Find documentation directory
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent / "docs"
    
    if not docs_dir.exists():
        print(f"Error: Documentation directory not found at {docs_dir}")
        sys.exit(1)
    
    print(f"Scanning documentation files in {docs_dir}")
    print("=" * 80)
    
    # Find all RST files
    rst_files = find_rst_files(docs_dir)
    print(f"Found {len(rst_files)} .rst files")
    
    # Track all issues
    all_issues = defaultdict(list)
    total_refs = 0
    valid_refs = 0
    
    # Process each file
    for rst_file in sorted(rst_files):
        relative_path = rst_file.relative_to(docs_dir)
        references = extract_references(rst_file)
        
        if not references:
            continue
        
        file_issues = []
        
        for line_num, ref_type, reference in references:
            total_refs += 1
            exists, error = verify_reference(reference)
            
            if exists:
                valid_refs += 1
            else:
                file_issues.append({
                    'line': line_num,
                    'type': ref_type,
                    'reference': reference,
                    'error': error
                })
        
        if file_issues:
            all_issues[str(relative_path)] = file_issues
    
    # Print summary
    print(f"\nVerification Summary:")
    print(f"Total references checked: {total_refs}")
    print(f"Valid references: {valid_refs}")
    print(f"Invalid references: {total_refs - valid_refs}")
    print(f"Files with issues: {len(all_issues)}")
    
    # Print detailed issues
    if all_issues:
        print("\n" + "=" * 80)
        print("ISSUES FOUND:")
        print("=" * 80)
        
        for file_path, issues in sorted(all_issues.items()):
            print(f"\n{file_path}: ({len(issues)} issues)")
            print("-" * len(file_path))
            
            for issue in issues:
                print(f"  Line {issue['line']}: {issue['type']} `{issue['reference']}`")
                print(f"    → {issue['error']}")
    
    # Generate report file
    report_path = script_dir / "doc_verification_report.txt"
    with open(report_path, 'w') as f:
        f.write("Documentation Reference Verification Report\n")
        f.write("==========================================\n\n")
        f.write(f"Total references checked: {total_refs}\n")
        f.write(f"Valid references: {valid_refs}\n")
        f.write(f"Invalid references: {total_refs - valid_refs}\n")
        f.write(f"Files with issues: {len(all_issues)}\n\n")
        
        if all_issues:
            f.write("Detailed Issues:\n")
            f.write("================\n\n")
            
            for file_path, issues in sorted(all_issues.items()):
                f.write(f"{file_path}:\n")
                for issue in issues:
                    f.write(f"  Line {issue['line']}: {issue['reference']} - {issue['error']}\n")
                f.write("\n")
    
    print(f"\nDetailed report saved to: {report_path}")
    
    # Exit with error code if issues found
    if all_issues:
        print("\n⚠️  Documentation references non-existent functions/classes!")
        sys.exit(1)
    else:
        print("\n✅ All documentation references are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()