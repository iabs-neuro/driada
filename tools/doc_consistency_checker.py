#!/usr/bin/env python3
"""
Documentation Consistency Checker for Driada Project.

This tool verifies that docstrings are consistent with their implementation
by checking function signatures, parameters, return types, and detecting
when implementations have changed without corresponding documentation updates.
"""

import ast
import hashlib
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import difflib
import inspect


class DocConsistencyChecker:
    """Check documentation consistency with implementation."""
    
    def __init__(self, cache_file: str = ".doc_consistency_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.inconsistencies = []
        
    def _load_cache(self) -> Dict:
        """Load the cache of previous implementation signatures."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save the current implementation signatures to cache."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_function_signature_hash(self, node: ast.FunctionDef) -> str:
        """Generate a hash of the function signature and body."""
        # Create a string representation of the function
        sig_parts = [node.name]
        
        # Add arguments
        args = node.args
        for arg in args.args:
            sig_parts.append(f"arg:{arg.arg}")
        if args.vararg:
            sig_parts.append(f"*{args.vararg.arg}")
        if args.kwarg:
            sig_parts.append(f"**{args.kwarg.arg}")
        
        # Add return type if present
        if node.returns:
            sig_parts.append(f"returns:{ast.dump(node.returns)}")
        
        # Add a simplified version of the function body
        # (to detect implementation changes)
        body_str = ast.dump(node)
        sig_parts.append(body_str)
        
        # Create hash
        signature = "|".join(sig_parts)
        return hashlib.md5(signature.encode()).hexdigest()
    
    def _extract_docstring_params(self, docstring: str) -> Dict[str, str]:
        """Extract parameters from NumPy-style docstring."""
        if not docstring:
            return {}
        
        params = {}
        in_params_section = False
        current_param = None
        lines = docstring.split('\n')
        
        # Section headers that end parameter parsing
        section_headers = [
            'parameters', 'parameters:', 'args:', 'arguments:',
            'returns', 'returns:', 'yields:', 'yields:',
            'raises', 'raises:', 'notes', 'notes:',
            'examples', 'examples:', 'see also', 'see also:',
            'references', 'references:', 'attributes', 'attributes:',
            'warnings', 'warnings:', 'todo', 'todo:'
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            line_lower = line.lower()
            
            # Check for section headers
            if line_lower in section_headers:
                if line_lower in ['parameters', 'parameters:', 'args:', 'arguments:']:
                    in_params_section = True
                    # Skip the separator line (usually -----)
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith('-'):
                        i += 1
                else:
                    in_params_section = False
                i += 1
                continue
            
            if in_params_section and line:
                # NumPy style: parameter line doesn't start with whitespace
                # Parameters in NumPy style are at the beginning of the line (no indent)
                original_line = lines[i]  # Keep original to check indentation
                
                if not original_line.startswith((' ', '\t')):
                    # Check if this is a parameter definition
                    # It either has ' : ' for typed params or is *args/**kwargs
                    if ' : ' in line:
                        # Standard parameter with type annotation
                        parts = line.split(' : ', 1)
                        param_name = parts[0].strip()
                        
                        # Skip if it looks like a bullet point or special formatting
                        # But allow asterisks at the beginning for *args, **kwargs
                        if param_name and not param_name.lstrip('*').startswith(('-', '*', '+', '•', '·', '◦')):
                            param_type = parts[1].strip() if len(parts) > 1 else ''
                            # Keep the parameter name as-is (including asterisks for *args, **kwargs)
                            params[param_name] = param_type
                            current_param = param_name
                    elif line.startswith('*'):
                        # Special case for *args, **kwargs, **method_kwargs etc. without type annotation
                        # Must be a valid parameter name (not just * or ** or bullet points)
                        param_name = line.strip()
                        # Extract the name part after * or **
                        name_part = param_name.lstrip('*')
                        # Check if it's a valid Python identifier and not a bullet point pattern
                        if name_part and name_part.isidentifier() and not line.startswith('***'):
                            params[param_name] = ''
                            current_param = param_name
            
            i += 1
        
        return params
    
    def _check_function_consistency(self, filepath: str, node: ast.FunctionDef) -> List[Tuple[str, str]]:
        """Check consistency between function implementation and documentation.
        
        Returns list of tuples (issue_type, issue_description) where issue_type is:
        - 'critical': Missing docstrings, undocumented params, non-existent params
        - 'warning': Implementation changes
        """
        issues = []
        
        # Get function signature including *args and **kwargs
        func_args = {arg.arg for arg in node.args.args if arg.arg != 'self'}
        
        # Add keyword-only arguments
        func_args.update(arg.arg for arg in node.args.kwonlyargs)
        
        # Add *args and **kwargs if present
        if node.args.vararg:
            func_args.add(node.args.vararg.arg)
        if node.args.kwarg:
            func_args.add(node.args.kwarg.arg)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        if not docstring and func_args:
            issues.append(("critical", f"Missing docstring for function with parameters"))
            return issues
        
        if docstring:
            # Extract documented parameters
            doc_params = set(self._extract_docstring_params(docstring).keys())
            
            # For comparison, we need to handle the case where docstring might have
            # *args/**kwargs while function has args/kwargs
            doc_params_normalized = set()
            for param in doc_params:
                # Remove leading asterisks for comparison
                normalized = param.lstrip('*')
                doc_params_normalized.add(normalized)
            
            # Check for undocumented parameters
            undocumented = func_args - doc_params_normalized
            if undocumented:
                issues.append(("critical", f"Undocumented parameters: {', '.join(sorted(undocumented))}"))
            
            # Check for documented but non-existent parameters
            non_existent = doc_params_normalized - func_args
            if non_existent:
                issues.append(("critical", f"Documented but non-existent parameters: {', '.join(sorted(non_existent))}"))
        
        # Check if implementation has changed
        sig_hash = self._get_function_signature_hash(node)
        cache_key = f"{filepath}::{node.name}"
        
        if cache_key in self.cache:
            if self.cache[cache_key]['hash'] != sig_hash:
                issues.append(("warning", f"Implementation changed since last check (consider updating docs)"))
                self.cache[cache_key]['last_change'] = datetime.now().isoformat()
        
        # Update cache
        self.cache[cache_key] = {
            'hash': sig_hash,
            'last_checked': datetime.now().isoformat(),
            'last_change': self.cache.get(cache_key, {}).get('last_change', datetime.now().isoformat())
        }
        
        return issues
    
    def check_file(self, filepath: str) -> List[Tuple[str, int, str, List[Tuple[str, str]]]]:
        """Check a single Python file for documentation consistency."""
        file_issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    issues = self._check_function_consistency(filepath, node)
                    if issues:
                        file_issues.append((
                            filepath,
                            node.lineno,
                            node.name,
                            issues
                        ))
                elif isinstance(node, ast.ClassDef):
                    # Check class docstring
                    class_docstring = ast.get_docstring(node)
                    if not class_docstring:
                        file_issues.append((
                            filepath,
                            node.lineno,
                            node.name,
                            [("critical", "Missing class docstring")]
                        ))
        
        except Exception as e:
            file_issues.append((
                filepath,
                0,
                "FILE_ERROR",
                [("critical", f"Error parsing file: {str(e)}")]
            ))
        
        return file_issues
    
    def check_directory(self, directory: str, exclude_patterns: List[str] = None) -> Dict[str, List]:
        """Check all Python files in a directory recursively."""
        exclude_patterns = exclude_patterns or ['__pycache__', '.git', 'build', 'dist']
        all_issues = {}
        
        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    issues = self.check_file(filepath)
                    if issues:
                        all_issues[filepath] = issues
        
        return all_issues
    
    def generate_report(self, issues: Dict[str, List], format: str = 'text') -> str:
        """Generate a report of all issues found."""
        if format == 'json':
            # Convert tuples to dict for JSON serialization
            json_issues = {}
            for filepath, file_issues in issues.items():
                json_issues[filepath] = []
                for fp, line, func, issue_list in file_issues:
                    json_issues[filepath].append({
                        'line': line,
                        'function': func,
                        'issues': [{'type': itype, 'message': msg} for itype, msg in issue_list]
                    })
            return json.dumps(json_issues, indent=2)
        
        # Text format
        report_lines = []
        report_lines.append("Documentation Consistency Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if not issues:
            report_lines.append("✅ No documentation inconsistencies found!")
        else:
            # Count issues by type
            critical_count = 0
            warning_count = 0
            
            for file_issues in issues.values():
                for _, _, _, issue_list in file_issues:
                    for issue_type, _ in issue_list:
                        if issue_type == 'critical':
                            critical_count += 1
                        else:
                            warning_count += 1
            
            total_issues = critical_count + warning_count
            report_lines.append(f"Found {total_issues} issues in {len(issues)} files:")
            report_lines.append(f"  ❌ Critical Issues: {critical_count}")
            report_lines.append(f"  ⚠️  Warnings: {warning_count}")
            report_lines.append("")
            
            # Group issues by type
            critical_files = {}
            warning_files = {}
            
            for filepath, file_issues in issues.items():
                for fp, line, func_name, issue_list in file_issues:
                    for issue_type, issue_msg in issue_list:
                        if issue_type == 'critical':
                            if filepath not in critical_files:
                                critical_files[filepath] = []
                            critical_files[filepath].append((line, func_name, issue_msg))
                        else:
                            if filepath not in warning_files:
                                warning_files[filepath] = []
                            warning_files[filepath].append((line, func_name, issue_msg))
            
            # Print critical issues first
            if critical_files:
                report_lines.append("\n❌ CRITICAL ISSUES - Must Fix")
                report_lines.append("=" * 40)
                for filepath, file_issues in sorted(critical_files.items()):
                    report_lines.append(f"\n📄 {filepath}")
                    for line, func_name, issue_msg in sorted(file_issues):
                        report_lines.append(f"  Line {line} - {func_name}():")
                        report_lines.append(f"    ❌ {issue_msg}")
            
            # Print warnings after
            if warning_files:
                report_lines.append("\n\n⚠️  WARNINGS - Implementation Changes")
                report_lines.append("=" * 40)
                for filepath, file_issues in sorted(warning_files.items()):
                    report_lines.append(f"\n📄 {filepath}")
                    for line, func_name, issue_msg in sorted(file_issues):
                        report_lines.append(f"  Line {line} - {func_name}():")
                        report_lines.append(f"    ⚠️  {issue_msg}")
        
        return "\n".join(report_lines)


def main():
    """Main entry point for the documentation consistency checker."""
    parser = argparse.ArgumentParser(description="Check documentation consistency in Python code")
    parser.add_argument('path', help='File or directory to check')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                        help='Output format (default: text)')
    parser.add_argument('--save-cache', action='store_true',
                        help='Save implementation signatures to cache')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear the cache before checking')
    parser.add_argument('--exclude', nargs='+', default=['__pycache__', '.git', 'build', 'dist'],
                        help='Patterns to exclude from directory search')
    
    args = parser.parse_args()
    
    checker = DocConsistencyChecker()
    
    if args.clear_cache and os.path.exists(checker.cache_file):
        os.remove(checker.cache_file)
        checker.cache = {}
    
    # Check if path is file or directory
    if os.path.isfile(args.path):
        issues = {args.path: checker.check_file(args.path)}
    elif os.path.isdir(args.path):
        issues = checker.check_directory(args.path, args.exclude)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)
    
    # Generate and print report
    report = checker.generate_report(issues, args.format)
    print(report)
    
    # Save cache if requested
    if args.save_cache:
        checker._save_cache()
        print(f"\n💾 Cache saved to {checker.cache_file}")
    
    # Exit with error code only if critical issues found
    has_critical = False
    if issues:
        for file_issues in issues.values():
            for _, _, _, issue_list in file_issues:
                if any(issue_type == 'critical' for issue_type, _ in issue_list):
                    has_critical = True
                    break
            if has_critical:
                break
    
    sys.exit(1 if has_critical else 0)


if __name__ == '__main__':
    main()