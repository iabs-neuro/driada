#!/usr/bin/env python3
"""
Apply RST fixes to documentation files to make references clickable.
This script will add the missing autofunction/autoclass directives to RST files.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from datetime import datetime


def read_fixes_from_file(fixes_file: Path) -> Dict[str, List[str]]:
    """Read the generated fixes from the fixes file."""
    fixes = {}
    current_file = None
    current_fixes = []
    
    with open(fixes_file, 'r') as f:
        for line in f:
            # Check for file header
            if line.startswith('## File: docs/'):
                if current_file and current_fixes:
                    fixes[current_file] = current_fixes
                    current_fixes = []
                
                # Extract file path
                current_file = line.strip().replace('## File: docs/', '').strip()
            
            # Collect autofunction/autoclass directives
            elif line.startswith('.. auto'):
                current_fixes.append(line.rstrip())
                # Also collect indented lines after the directive
                in_directive = True
            elif line.strip() and in_directive and line.startswith('   '):
                current_fixes[-1] += '\n' + line.rstrip()
            elif line.strip() == '':
                in_directive = False
    
    # Don't forget the last file
    if current_file and current_fixes:
        fixes[current_file] = current_fixes
    
    return fixes


def find_section_to_insert(content: str, reference: str) -> int:
    """Find the appropriate section to insert the directive."""
    lines = content.split('\n')
    
    # Extract the type from reference (e.g., 'data', 'embedding', etc.)
    parts = reference.split('.')
    if len(parts) >= 3:
        ref_type = parts[2]  # e.g., 'data' from 'driada.dim_reduction.data.MVData'
    else:
        ref_type = 'general'
    
    # Look for appropriate section headers
    section_markers = [
        ('Core Classes', ['data', 'embedding', 'graph', 'dr_base']),
        ('Main Functions', ['sequences', 'dr_sequence']),
        ('Utilities', ['utils']),
        ('Neural Methods', ['neural', 'losses']),
        ('API Reference', []),  # Catch-all
    ]
    
    best_position = -1
    
    for i, line in enumerate(lines):
        # Check for section headers
        for section_name, keywords in section_markers:
            if section_name in line and (not keywords or any(k in ref_type for k in keywords)):
                # Found a matching section, look for the end of the section
                # (next header at same or higher level, or end of file)
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and (
                        lines[j].startswith('---') or 
                        lines[j].startswith('===') or
                        (j + 1 < len(lines) and (
                            lines[j + 1].startswith('---') or 
                            lines[j + 1].startswith('===')
                        ))
                    ):
                        # Found next section, insert before it
                        return j - 1
                
                # No next section found, use end of file
                return len(lines) - 1
    
    # If no appropriate section found, look for the end of the module documentation
    for i, line in enumerate(lines):
        if 'automodule::' in line:
            # Skip past the automodule directive and its options
            j = i + 1
            while j < len(lines) and (lines[j].strip() == '' or lines[j].startswith('   ')):
                j += 1
            return j
    
    # Default: insert at the end
    return len(lines) - 1


def apply_fixes_to_file(rst_file: Path, directives: List[str]) -> bool:
    """Apply fixes to a single RST file."""
    # Read the current content
    with open(rst_file, 'r') as f:
        content = f.read()
    
    # Check which directives are already present
    new_directives = []
    for directive in directives:
        # Extract the reference from the directive
        match = re.search(r':: (driada\.[^\s]+)', directive)
        if match:
            ref = match.group(1)
            # Check if this reference is already documented
            if ref not in content:
                new_directives.append(directive)
    
    if not new_directives:
        print(f"  All references already documented in {rst_file}")
        return False
    
    # Create backup
    backup_file = rst_file.with_suffix('.rst.bak')
    shutil.copy2(rst_file, backup_file)
    
    # Find appropriate insertion point
    lines = content.split('\n')
    
    # Look for an "API Reference" section or create one
    api_ref_index = -1
    for i, line in enumerate(lines):
        if 'API Reference' in line:
            api_ref_index = i
            break
    
    if api_ref_index == -1:
        # Create API Reference section
        # Find a good place to insert it (after main content, before indices)
        insert_index = len(lines) - 1
        for i, line in enumerate(lines):
            if 'Indices and tables' in line:
                insert_index = i - 1
                break
        
        # Add API Reference section
        new_section = [
            '',
            'API Reference',
            '-' * len('API Reference'),
            '',
            '.. note::',
            '   The following functions and classes are referenced in this module.',
            '',
        ]
        
        lines[insert_index:insert_index] = new_section
        api_ref_index = insert_index + 1
    
    # Find where to insert within the API Reference section
    insert_index = api_ref_index
    for i in range(api_ref_index + 1, len(lines)):
        if lines[i].strip() and not lines[i].startswith(' '):
            # Found next section
            insert_index = i - 1
            break
    else:
        insert_index = len(lines) - 1
    
    # Insert the new directives
    for directive in new_directives:
        # Add spacing
        lines.insert(insert_index, '')
        insert_index += 1
        
        # Add directive lines
        for line in directive.split('\n'):
            lines.insert(insert_index, line)
            insert_index += 1
    
    # Write the updated content
    with open(rst_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Applied {len(new_directives)} fixes to {rst_file}")
    return True


def apply_all_fixes(fixes_file: Path = None) -> None:
    """Apply all fixes to RST files."""
    if fixes_file is None:
        fixes_file = Path('/Users/nikita/PycharmProjects/driada2/rst_reference_fixes.txt')
    
    if not fixes_file.exists():
        print(f"Fixes file not found: {fixes_file}")
        return
    
    # Read fixes
    fixes = read_fixes_from_file(fixes_file)
    
    if not fixes:
        print("No fixes found in the fixes file.")
        return
    
    docs_dir = Path('/Users/nikita/PycharmProjects/driada2/docs')
    
    print(f"Applying fixes to {len(fixes)} RST files...")
    print("=" * 60)
    
    applied_count = 0
    for rst_path, directives in fixes.items():
        full_path = docs_dir / rst_path
        
        if not full_path.exists():
            print(f"Warning: RST file not found: {full_path}")
            continue
        
        print(f"\nProcessing: {rst_path}")
        if apply_fixes_to_file(full_path, directives):
            applied_count += 1
    
    print("\n" + "=" * 60)
    print(f"Summary: Applied fixes to {applied_count} files")
    
    # Create a summary file
    summary_file = Path('/Users/nikita/PycharmProjects/driada2/rst_fixes_applied.txt')
    with open(summary_file, 'w') as f:
        f.write(f"RST Fixes Applied - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total files processed: {len(fixes)}\n")
        f.write(f"Files updated: {applied_count}\n\n")
        f.write("Files processed:\n")
        for rst_path in sorted(fixes.keys()):
            f.write(f"  - {rst_path}\n")
    
    print(f"\nSummary written to: {summary_file}")
    print("\nIMPORTANT: Backup files created with .bak extension")
    print("To test the changes, rebuild the documentation:")
    print("  cd docs && make clean && make html")


def main():
    """Main function."""
    print("RST Documentation Fix Applicator")
    print("=" * 60)
    
    # Check if fixes file exists
    fixes_file = Path('/Users/nikita/PycharmProjects/driada2/rst_reference_fixes.txt')
    
    if not fixes_file.exists():
        print("No fixes file found. Please run fix_rst_references.py first.")
        return
    
    # Apply fixes
    apply_all_fixes(fixes_file)
    
    print("\nDone! Please rebuild documentation to verify changes.")


if __name__ == "__main__":
    main()