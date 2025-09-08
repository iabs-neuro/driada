#!/usr/bin/env python3
"""Verify documentation consistency by checking if function signatures match their docstrings."""

import ast
import inspect
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib
import json
from datetime import datetime


class DocConsistencyChecker:
    """Check consistency between code implementation and documentation."""
    
    def __init__(self, cache_file: str = ".doc_consistency_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.inconsistencies = []
        
    def _load_cache(self) -> dict:
        """Load the cache of function signatures."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save the cache of function signatures."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_function_signature_hash(self, node: ast.FunctionDef) -> str:
        """Get a hash of the function signature (name, args, return type)."""
        sig_parts = [node.name]
        
        # Add arguments
        for arg in node.args.args:
            sig_parts.append(arg.arg)
            if arg.annotation:
                sig_parts.append(ast.unparse(arg.annotation))
        
        # Add defaults
        defaults_offset = len(node.args.args) - len(node.args.defaults)
        for i, default in enumerate(node.args.defaults):
            sig_parts.append(f"default_{i + defaults_offset}={ast.unparse(default)}")
        
        # Add return annotation
        if node.returns:
            sig_parts.append(f"return={ast.unparse(node.returns)}")
        
        sig_str = "|".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()
    
    def _extract_docstring_params(self, docstring: str) -> Dict[str, str]:
        """Extract parameter descriptions from numpy-style docstring."""
        if not docstring:
            return {}
        
        params = {}
        in_params_section = False
        current_param = None
        
        for line in docstring.split('\n'):
            line = line.strip()
            
            if line == "Parameters":
                in_params_section = True
                continue
            elif line in ["Returns", "Raises", "Examples", "Notes", "See Also"]:
                in_params_section = False
                continue
                
            if in_params_section:
                # Check if this is a parameter definition
                param_match = re.match(r'^(\w+)\s*:\s*(.+)?$', line)
                if param_match:
                    current_param = param_match.group(1)
                    params[current_param] = param_match.group(2) or ""
                elif current_param and line:
                    # Continuation of previous parameter description
                    params[current_param] += " " + line
        
        return params
    
    def _check_function_consistency(self, node: ast.FunctionDef, filepath: str) -> List[str]:
        """Check if function signature matches its docstring."""
        issues = []
        
        # Get function arguments
        func_args = [arg.arg for arg in node.args.args if arg.arg != 'self']
        
        # Get docstring
        docstring = ast.get_docstring(node)
        if not docstring:
            # Skip functions without docstrings (might be internal)
            return issues
        
        # Extract documented parameters
        doc_params = self._extract_docstring_params(docstring)
        doc_param_names = set(doc_params.keys())
        func_arg_set = set(func_args)
        
        # Check for mismatches
        missing_in_doc = func_arg_set - doc_param_names
        extra_in_doc = doc_param_names - func_arg_set
        
        if missing_in_doc:
            issues.append(
                f"{filepath}:{node.lineno} - Function '{node.name}' has undocumented parameters: {missing_in_doc}"
            )
        
        if extra_in_doc:
            # Filter out common extras that might be valid
            extra_filtered = {p for p in extra_in_doc if p not in ['**kwargs', '*args']}
            if extra_filtered:
                issues.append(
                    f"{filepath}:{node.lineno} - Function '{node.name}' documents non-existent parameters: {extra_filtered}"
                )
        
        # Check if signature has changed
        sig_hash = self._get_function_signature_hash(node)
        cache_key = f"{filepath}:{node.name}"
        
        if cache_key in self.cache:
            if self.cache[cache_key]['signature'] != sig_hash:
                issues.append(
                    f"{filepath}:{node.lineno} - Function '{node.name}' signature has changed since last check"
                )
        
        # Update cache
        self.cache[cache_key] = {
            'signature': sig_hash,
            'last_checked': datetime.now().isoformat(),
            'has_docstring': bool(docstring)
        }
        
        return issues
    
    def _check_class_consistency(self, node: ast.ClassDef, filepath: str) -> List[str]:
        """Check class and its methods for consistency."""
        issues = []
        
        # Check class docstring
        class_docstring = ast.get_docstring(node)
        if not class_docstring and not node.name.startswith('_'):
            issues.append(
                f"{filepath}:{node.lineno} - Public class '{node.name}' is missing docstring"
            )
        
        # Check methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Skip private methods and special methods
                if not item.name.startswith('_') or item.name in ['__init__', '__call__']:
                    issues.extend(self._check_function_consistency(item, filepath))
        
        return issues
    
    def check_file(self, filepath: str) -> List[str]:
        """Check a single Python file for documentation consistency."""
        issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Only check top-level functions and public methods
                    if not node.name.startswith('_'):
                        issues.extend(self._check_function_consistency(node, filepath))
                elif isinstance(node, ast.ClassDef):
                    issues.extend(self._check_class_consistency(node, filepath))
        
        except Exception as e:
            issues.append(f"{filepath}: Error parsing file - {e}")
        
        return issues
    
    def check_directory(self, directory: Path) -> Tuple[List[str], Dict[str, int]]:
        """Check all Python files in a directory for consistency."""
        all_issues = []
        stats = {
            'files_checked': 0,
            'issues_found': 0,
            'functions_checked': 0,
            'signature_changes': 0
        }
        
        for py_file in directory.rglob('*.py'):
            # Skip test files and __pycache__
            if '__pycache__' in str(py_file) or 'test_' in py_file.name:
                continue
            
            issues = self.check_file(str(py_file))
            if issues:
                all_issues.extend(issues)
                stats['issues_found'] += len(issues)
            
            stats['files_checked'] += 1
        
        # Count signature changes
        for issue in all_issues:
            if 'signature has changed' in issue:
                stats['signature_changes'] += 1
        
        stats['functions_checked'] = len(self.cache)
        
        return all_issues, stats
    
    def generate_report(self, issues: List[str], stats: Dict[str, int]) -> str:
        """Generate a consistency report."""
        report = []
        report.append("=== Documentation Consistency Report ===")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("=== Statistics ===")
        report.append(f"Files checked: {stats['files_checked']}")
        report.append(f"Functions tracked: {stats['functions_checked']}")
        report.append(f"Issues found: {stats['issues_found']}")
        report.append(f"Signature changes detected: {stats['signature_changes']}")
        report.append("")
        
        if issues:
            report.append("=== Issues Found ===")
            # Group issues by file
            issues_by_file = {}
            for issue in sorted(issues):
                filepath = issue.split(':')[0]
                if filepath not in issues_by_file:
                    issues_by_file[filepath] = []
                issues_by_file[filepath].append(issue)
            
            for filepath, file_issues in issues_by_file.items():
                report.append(f"\n{filepath}:")
                for issue in file_issues:
                    report.append(f"  - {issue}")
        else:
            report.append("✅ No documentation consistency issues found!")
        
        return "\n".join(report)


def main():
    """Main function to run the consistency checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check documentation consistency in Python code")
    parser.add_argument(
        "path",
        nargs="?",
        default="src",
        help="Path to check (default: src)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output report to file"
    )
    parser.add_argument(
        "--update-cache",
        action="store_true",
        help="Update the cache without reporting issues"
    )
    
    args = parser.parse_args()
    
    # Get the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    check_path = project_root / args.path
    
    # Create checker
    checker = DocConsistencyChecker(
        cache_file=str(project_root / ".doc_consistency_cache.json")
    )
    
    # Run checks
    print(f"Checking documentation consistency in: {check_path}")
    issues, stats = checker.check_directory(check_path)
    
    # Save cache
    checker._save_cache()
    
    if args.update_cache:
        print("Cache updated successfully!")
        return
    
    # Generate report
    report = checker.generate_report(issues, stats)
    
    # Output report
    print(report)
    
    if args.output:
        output_path = project_root / args.output
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    
    # Exit with error code if issues found
    if issues and not args.update_cache:
        exit(1)


if __name__ == "__main__":
    main()