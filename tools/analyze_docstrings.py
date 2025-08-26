#!/usr/bin/env python3
"""Analyze functions in DRIADA codebase to find missing or incomplete docstrings."""

import ast
import os
from pathlib import Path
from typing import List, Dict, Tuple
import textwrap


class DocstringAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze function docstrings."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.results = []
        
    def visit_FunctionDef(self, node):
        """Visit function definitions and analyze their docstrings."""
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Analyze function
        issues = []
        params = []
        
        # Get function parameters
        for arg in node.args.args:
            if arg.arg != 'self':
                params.append(arg.arg)
        
        # Get keyword-only args
        for arg in node.args.kwonlyargs:
            params.append(arg.arg)
            
        # Check if function has docstring
        if not docstring:
            issues.append("Missing docstring")
        else:
            # Parse docstring sections
            has_params_section = "Parameters" in docstring
            has_returns_section = "Returns" in docstring
            
            # Check for missing parameter documentation
            if params and not has_params_section:
                issues.append("Missing Parameters section")
            elif params and has_params_section:
                # Check if all params are documented
                missing_params = []
                for param in params:
                    if f"{param} :" not in docstring and f"{param}:" not in docstring:
                        missing_params.append(param)
                if missing_params:
                    issues.append(f"Missing parameter docs: {', '.join(missing_params)}")
                    
            # Check for returns section if function returns something
            has_return = any(isinstance(n, ast.Return) and n.value is not None 
                           for n in ast.walk(node))
            if has_return and not has_returns_section and node.name != '__init__':
                issues.append("Missing Returns section")
                
        if issues:
            self.results.append({
                'function': node.name,
                'line': node.lineno,
                'params': params,
                'issues': issues,
                'is_public': not node.name.startswith('_')
            })
            
        self.generic_visit(node)
        
        
def analyze_file(filepath: Path) -> List[Dict]:
    """Analyze a single Python file for docstring issues."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        analyzer = DocstringAnalyzer(str(filepath))
        analyzer.visit(tree)
        
        return analyzer.results
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return []


def get_function_importance(filepath: str, func_name: str) -> int:
    """Score function importance based on various factors."""
    score = 0
    
    # Public functions are more important
    if not func_name.startswith('_'):
        score += 3
        
    # Functions in key modules are more important
    if 'information' in filepath:
        if any(key in filepath for key in ['info_base', 'gcmi', 'ksg', 'entropy']):
            score += 5
            
    # Key function names
    key_functions = ['get_mi', 'get_1d_mi', 'get_multi_mi', 'conditional_mi',
                     'mi_gg', 'gcmi_cc', 'gccmi_ccd', 'nonparam_mi_cc',
                     'nonparam_entropy_c', 'entropy_d', 'get_tdmi',
                     'interaction_information']
    if func_name in key_functions:
        score += 10
        
    # Functions with many parameters are likely important
    return score


def main():
    """Main function to analyze DRIADA codebase."""
    # Define paths to analyze
    base_path = Path("/Users/nikita/PycharmProjects/driada2/src/driada")
    
    # Key directories to analyze
    directories = [
        base_path / "information",
        base_path / "models",
        base_path / "experiment"
    ]
    
    all_issues = []
    
    for directory in directories:
        if not directory.exists():
            continue
            
        for py_file in directory.glob("**/*.py"):
            # Skip __init__ and test files
            if py_file.name == '__init__.py' or 'test' in py_file.name:
                continue
                
            issues = analyze_file(py_file)
            if issues:
                for issue in issues:
                    issue['filepath'] = str(py_file)
                    issue['module'] = py_file.parent.name
                    issue['importance'] = get_function_importance(str(py_file), issue['function'])
                    all_issues.append(issue)
                    
    # Sort by importance and filter to top issues
    all_issues.sort(key=lambda x: x['importance'], reverse=True)
    
    # Focus on public functions with high importance
    critical_issues = [i for i in all_issues if i['is_public'] and i['importance'] >= 5]
    
    print(f"# DRIADA Critical Documentation Issues\n")
    print(f"Found {len(all_issues)} total issues, {len(critical_issues)} critical\n")
    
    # Group by module
    by_module = {}
    for issue in critical_issues[:20]:  # Top 20 critical issues
        module_path = issue['filepath'].split('/driada/')[-1]
        if module_path not in by_module:
            by_module[module_path] = []
        by_module[module_path].append(issue)
        
    # Print results
    for module_path, issues in by_module.items():
        print(f"\n## {module_path}")
        for issue in issues:
            print(f"\n### {issue['function']} (line {issue['line']}, importance: {issue['importance']})")
            print(f"Parameters: {', '.join(issue['params']) if issue['params'] else 'None'}")
            print(f"Issues: {'; '.join(issue['issues'])}")


if __name__ == "__main__":
    main()