#!/usr/bin/env python3
"""
Fix common issues in documentation examples.
"""

import re
from pathlib import Path
from typing import List, Tuple


def fix_experiment_constructor(code: str) -> str:
    """Fix Experiment() constructor calls to match current API."""
    # Pattern to match Experiment(neural_data) or similar simple calls
    pattern = r'exp = Experiment\(([^)]+)\)'
    
    def replace_func(match):
        args = match.group(1).strip()
        # If it's just a single argument (like neural_data), it needs fixing
        if ',' not in args and 'calcium=' not in args:
            return f'# Note: Experiment requires multiple parameters\n# exp = Experiment(signature, calcium, spikes, exp_identificators, static_features, dynamic_features)\n# For this example, assume exp is already created'
        return match.group(0)
    
    return re.sub(pattern, replace_func, code)


def add_exp_definition(code: str) -> str:
    """Add exp definition if it's used but not defined."""
    # Check if exp is used but not defined
    if 'exp.' in code and 'exp =' not in code and 'exp' not in code.split('=')[0]:
        # Check if there's already an import section
        if 'from driada' in code or 'import' in code:
            # Find the last import line
            lines = code.split('\n')
            last_import_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    last_import_idx = i
            
            if last_import_idx >= 0:
                # Add exp assumption after imports
                lines.insert(last_import_idx + 1, '')
                lines.insert(last_import_idx + 2, '# Assume exp is an Experiment object already created')
                lines.insert(last_import_idx + 3, '# exp = Experiment(...) # See Experiment docs for full parameters')
                return '\n'.join(lines)
        else:
            # Add at the beginning
            return '# Assume exp is an Experiment object already created\n# exp = Experiment(...) # See Experiment docs for full parameters\n\n' + code
    
    return code


def fix_code_block(code: str) -> str:
    """Apply all fixes to a code block."""
    # Apply fixes in order
    code = fix_experiment_constructor(code)
    code = add_exp_definition(code)
    
    return code


def process_rst_file(file_path: Path) -> bool:
    """Process an RST file and fix code blocks. Returns True if modified."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    lines = content.split('\n')
    modified = False
    
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('.. code-block:: python'):
            # Found a code block
            i += 1
            # Skip empty lines
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            if i < len(lines) and lines[i].strip():
                # Get indentation
                indent_line = lines[i]
                indent_level = len(indent_line) - len(indent_line.lstrip())
                indent_str = ' ' * indent_level
                
                # Extract code block
                code_start = i
                code_lines = []
                
                while i < len(lines):
                    if lines[i].strip() and not lines[i].startswith(indent_str):
                        break
                    if lines[i].strip():
                        code_lines.append(lines[i][indent_level:])
                    else:
                        code_lines.append('')
                    i += 1
                
                code_end = i
                
                # Fix the code
                original_code = '\n'.join(code_lines)
                fixed_code = fix_code_block(original_code)
                
                if fixed_code != original_code:
                    # Replace the code block
                    fixed_lines = fixed_code.split('\n')
                    # Re-indent
                    fixed_lines = [indent_str + line if line else '' for line in fixed_lines]
                    
                    # Replace in original lines
                    lines[code_start:code_end] = fixed_lines
                    modified = True
                    i = code_start + len(fixed_lines)
        else:
            i += 1
    
    if modified:
        # Write back
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
        return True
    
    return False


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='Fix documentation examples')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    parser.add_argument('path', nargs='?', default=None,
                       help='Path to specific file or directory')
    args = parser.parse_args()
    
    # Find docs directory
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent / "docs"
    
    # Files with issues from the analysis
    files_to_fix = [
        'api/dimensionality.rst',
        'api/experiment/core.rst',
        'api/experiment/reconstruction.rst',
        'api/experiment/synthetic.rst',
        'api/experiment/wavelet.rst',
        'api/integration.rst',
        'quickstart.rst'
    ]
    
    if args.path:
        # Process specific file/directory
        test_path = Path(args.path)
        if not test_path.is_absolute():
            test_path = docs_dir / test_path
        
        if test_path.is_file():
            files_to_fix = [test_path.relative_to(docs_dir)]
        else:
            # Process all rst files in directory
            files_to_fix = [f.relative_to(docs_dir) for f in test_path.rglob("*.rst")]
    
    print(f"Processing {len(files_to_fix)} documentation files...")
    
    modified_count = 0
    for rel_path in files_to_fix:
        file_path = docs_dir / rel_path
        if not file_path.exists():
            print(f"⚠️  File not found: {rel_path}")
            continue
        
        if args.dry_run:
            print(f"Would process: {rel_path}")
        else:
            if process_rst_file(file_path):
                print(f"✅ Fixed: {rel_path}")
                modified_count += 1
            else:
                print(f"  No changes needed: {rel_path}")
    
    if not args.dry_run:
        print(f"\nModified {modified_count} files")


if __name__ == "__main__":
    main()