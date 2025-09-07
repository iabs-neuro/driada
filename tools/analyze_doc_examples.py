#!/usr/bin/env python3
"""
Analyze documentation examples to identify potential issues without running them.
This is faster and safer than executing all code blocks.
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def find_rst_files(docs_dir: Path, pattern: str = None) -> List[Path]:
    """Find RST files, optionally filtered by pattern."""
    all_files = list(docs_dir.rglob("*.rst"))
    if pattern:
        return [f for f in all_files if pattern in str(f)]
    return all_files


def extract_code_blocks(file_path: Path) -> List[Tuple[int, str]]:
    """Extract Python code blocks from RST file."""
    code_blocks = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.strip().startswith('.. code-block:: python'):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            if i < len(lines):
                indent_line = lines[i]
                if indent_line.strip():
                    indent_level = len(indent_line) - len(indent_line.lstrip())
                    
                    code_lines = []
                    start_line = i + 1
                    
                    while i < len(lines):
                        current_line = lines[i]
                        if current_line.strip():
                            current_indent = len(current_line) - len(current_line.lstrip())
                            if current_indent < indent_level:
                                break
                            code_lines.append(current_line[indent_level:])
                        else:
                            code_lines.append(current_line)
                        i += 1
                    
                    code = ''.join(code_lines).rstrip()
                    if code:
                        code_blocks.append((start_line, code))
        else:
            i += 1
    
    return code_blocks


def analyze_code_block(code: str, line_num: int) -> Dict:
    """Analyze a code block for potential issues."""
    issues = []
    imports = set()
    functions_called = set()
    classes_instantiated = set()
    
    try:
        # Parse the code
        tree = ast.parse(code)
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.add(f"{module}.{alias.name}" if module else alias.name)
            
            # Find function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    functions_called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle chained calls like exp.calcium.get_embedding
                    parts = []
                    current = node.func
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                        functions_called.add('.'.join(reversed(parts)))
                
                # Check for class instantiation
                if isinstance(node.func, ast.Name) and node.func.id[0].isupper():
                    classes_instantiated.add(node.func.id)
                    # Check number of arguments
                    n_args = len(node.args) + len(node.keywords)
                    if node.func.id == 'Experiment' and n_args < 6:
                        issues.append(f"Experiment() called with only {n_args} args, needs 6+")
    
    except SyntaxError as e:
        issues.append(f"Syntax error: {e}")
    except Exception as e:
        issues.append(f"Parse error: {e}")
    
    # Check for common issues
    if 'Experiment' in classes_instantiated and 'neural_data' in code:
        if not re.search(r'Experiment\s*\([^)]*calcium[^)]*\)', code):
            issues.append("Experiment() missing required 'calcium' parameter")
    
    # Check for undefined variables (simple heuristic)
    if 'exp.' in code and 'exp =' not in code and 'exp' not in code.split('=')[0]:
        issues.append("Using 'exp' without defining it")
    
    return {
        'line': line_num,
        'imports': imports,
        'functions': functions_called,
        'classes': classes_instantiated,
        'issues': issues,
        'code_preview': code.split('\n')[0][:80] + '...' if code else ''
    }


def analyze_rst_file(file_path: Path) -> Dict:
    """Analyze all code blocks in an RST file."""
    code_blocks = extract_code_blocks(file_path)
    
    results = {
        'file': file_path,
        'total_blocks': len(code_blocks),
        'blocks_with_issues': 0,
        'analyses': []
    }
    
    for line_num, code in code_blocks:
        analysis = analyze_code_block(code, line_num)
        if analysis['issues']:
            results['blocks_with_issues'] += 1
        results['analyses'].append(analysis)
    
    return results


def main():
    """Main analysis function."""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze documentation code examples')
    parser.add_argument('path', nargs='?', default=None,
                       help='Path to specific file or directory')
    args = parser.parse_args()
    
    # Find docs directory
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent / "docs"
    
    # Open report file for writing
    report_file = script_dir / "doc_analysis_report.txt"
    report_content = []
    
    # Determine files to analyze
    if args.path:
        test_path = Path(args.path)
        if not test_path.is_absolute():
            test_path = docs_dir / test_path
        
        if test_path.is_file():
            rst_files = [test_path] if test_path.suffix == '.rst' else []
        else:
            rst_files = list(test_path.rglob("*.rst"))
    else:
        rst_files = find_rst_files(docs_dir)
    
    header = f"Analyzing {len(rst_files)} documentation files..."
    print(header)
    print("=" * 80)
    report_content.append(header)
    report_content.append("=" * 80)
    
    # Analyze all files
    all_issues = []
    total_blocks = 0
    
    for rst_file in sorted(rst_files):
        relative_path = rst_file.relative_to(docs_dir)
        
        # Skip non-code files
        skip_patterns = ['changelog', 'contributing', 'license']
        if any(p in str(relative_path).lower() for p in skip_patterns):
            continue
        
        results = analyze_rst_file(rst_file)
        
        if results['blocks_with_issues'] > 0:
            all_issues.append(results)
            
        total_blocks += results['total_blocks']
        
        if results['total_blocks'] > 0:
            status = "✅" if results['blocks_with_issues'] == 0 else "⚠️"
            msg = f"{status} {relative_path}: {results['total_blocks']} blocks, " \
                  f"{results['blocks_with_issues']} with potential issues"
            print(msg)
            report_content.append(msg)
    
    # Print detailed issues
    if all_issues:
        report_content.append("")
        report_content.append("=" * 80)
        report_content.append("POTENTIAL ISSUES FOUND:")
        report_content.append("=" * 80)
        print("\n" + "=" * 80)
        print("POTENTIAL ISSUES FOUND:")
        print("=" * 80)
        
        for result in all_issues:
            relative_path = result['file'].relative_to(docs_dir)
            report_content.append(f"\n{relative_path}:")
            report_content.append("-" * len(str(relative_path)))
            print(f"\n{relative_path}:")
            print("-" * len(str(relative_path)))
            
            for analysis in result['analyses']:
                if analysis['issues']:
                    line_msg = f"\n  Line {analysis['line']}: {analysis['code_preview']}"
                    report_content.append(line_msg)
                    print(line_msg)
                    for issue in analysis['issues']:
                        issue_msg = f"    ⚠️  {issue}"
                        report_content.append(issue_msg)
                        print(issue_msg)
    
    summary1 = f"\n{total_blocks} total code blocks analyzed"
    summary2 = f"{len(all_issues)} files have potential issues"
    print(summary1)
    print(summary2)
    report_content.append(summary1)
    report_content.append(summary2)
    
    # Write report file
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_content))
    
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()