#!/usr/bin/env python
"""Analyze docstring coverage across the DRIADA codebase.

This tool examines Python files to determine:
1. Which functions/classes have docstrings
2. Quality of existing docstrings
3. Module-level documentation coverage
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re


class DocstringAnalyzer(ast.NodeVisitor):
    """Analyze docstrings in Python AST."""
    
    def __init__(self):
        self.classes = []
        self.functions = []
        self.methods = []
        self.current_class = None
        
    def visit_ClassDef(self, node):
        """Visit class definition."""
        has_docstring = (
            node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)
        )
        docstring = node.body[0].value.value if has_docstring else None
        
        self.classes.append({
            'name': node.name,
            'lineno': node.lineno,
            'has_docstring': has_docstring,
            'docstring': docstring,
            'docstring_quality': self._analyze_docstring_quality(docstring) if docstring else None
        })
        
        # Track current class for methods
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        has_docstring = (
            node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)
        )
        docstring = node.body[0].value.value if has_docstring else None
        
        func_info = {
            'name': node.name,
            'lineno': node.lineno,
            'has_docstring': has_docstring,
            'docstring': docstring,
            'docstring_quality': self._analyze_docstring_quality(docstring) if docstring else None,
            'is_private': node.name.startswith('_'),
            'is_test': node.name.startswith('test_')
        }
        
        if self.current_class:
            self.methods.append({**func_info, 'class': self.current_class})
        else:
            self.functions.append(func_info)
            
    def _analyze_docstring_quality(self, docstring: str) -> Dict[str, bool]:
        """Analyze the quality of a docstring."""
        if not docstring:
            return {}
            
        quality = {
            'has_description': len(docstring.strip()) > 0,
            'has_params': bool(re.search(r':param\s+\w+:|Parameters\s*\n\s*-+|Args:', docstring)),
            'has_returns': bool(re.search(r':returns?:|Returns\s*\n\s*-+|Returns:', docstring)),
            'has_examples': bool(re.search(r'Example[s]?\s*\n\s*-+|>>>', docstring)),
            'has_raises': bool(re.search(r':raises?:|Raises\s*\n\s*-+|Raises:', docstring)),
            'is_numpy_style': bool(re.search(r'Parameters\s*\n\s*-+|Returns\s*\n\s*-+', docstring)),
            'is_google_style': bool(re.search(r'Args:|Returns:|Raises:', docstring)),
            'is_sphinx_style': bool(re.search(r':param|:returns?:|:raises?:', docstring)),
        }
        
        # Calculate overall quality score
        quality['score'] = sum([
            quality['has_description'] * 2,  # Description is important
            quality['has_params'] * 2,        # Parameters are important
            quality['has_returns'] * 1,
            quality['has_examples'] * 1,
            quality['has_raises'] * 0.5
        ])
        
        return quality


def analyze_file(filepath: Path) -> Dict:
    """Analyze a single Python file for docstring coverage."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        analyzer = DocstringAnalyzer()
        analyzer.visit(tree)
        
        # Check for module-level docstring
        module_docstring = ast.get_docstring(tree)
        
        return {
            'filepath': filepath,
            'module_docstring': module_docstring is not None,
            'classes': analyzer.classes,
            'functions': analyzer.functions,
            'methods': analyzer.methods,
        }
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None


def analyze_module(module_path: Path) -> Dict:
    """Analyze all Python files in a module."""
    files_data = []
    
    for py_file in module_path.rglob("*.py"):
        # Skip __pycache__ and test files
        if '__pycache__' in str(py_file) or 'test_' in py_file.name:
            continue
            
        file_data = analyze_file(py_file)
        if file_data:
            files_data.append(file_data)
            
    return {
        'module': module_path.name,
        'files': files_data
    }


def calculate_stats(module_data: Dict) -> Dict:
    """Calculate statistics for a module."""
    total_classes = 0
    classes_with_docstrings = 0
    total_functions = 0
    functions_with_docstrings = 0
    total_methods = 0
    methods_with_docstrings = 0
    quality_scores = []
    
    for file_data in module_data['files']:
        # Classes
        for cls in file_data['classes']:
            total_classes += 1
            if cls['has_docstring']:
                classes_with_docstrings += 1
                if cls['docstring_quality']:
                    quality_scores.append(cls['docstring_quality']['score'])
                    
        # Functions
        for func in file_data['functions']:
            if not func['is_private']:  # Skip private functions
                total_functions += 1
                if func['has_docstring']:
                    functions_with_docstrings += 1
                    if func['docstring_quality']:
                        quality_scores.append(func['docstring_quality']['score'])
                        
        # Methods
        for method in file_data['methods']:
            if not method['is_private'] and method['name'] != '__init__':
                total_methods += 1
                if method['has_docstring']:
                    methods_with_docstrings += 1
                    if method['docstring_quality']:
                        quality_scores.append(method['docstring_quality']['score'])
    
    total_items = total_classes + total_functions + total_methods
    items_with_docstrings = classes_with_docstrings + functions_with_docstrings + methods_with_docstrings
    
    return {
        'total_classes': total_classes,
        'classes_with_docstrings': classes_with_docstrings,
        'class_coverage': (classes_with_docstrings / total_classes * 100) if total_classes > 0 else 0,
        'total_functions': total_functions,
        'functions_with_docstrings': functions_with_docstrings,
        'function_coverage': (functions_with_docstrings / total_functions * 100) if total_functions > 0 else 0,
        'total_methods': total_methods,
        'methods_with_docstrings': methods_with_docstrings,
        'method_coverage': (methods_with_docstrings / total_methods * 100) if total_methods > 0 else 0,
        'total_items': total_items,
        'items_with_docstrings': items_with_docstrings,
        'overall_coverage': (items_with_docstrings / total_items * 100) if total_items > 0 else 0,
        'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
        'max_quality_score': 6.5  # Maximum possible score
    }


def find_undocumented_items(module_data: Dict) -> List[Dict]:
    """Find all undocumented classes, functions, and methods."""
    undocumented = []
    
    for file_data in module_data['files']:
        filepath = file_data['filepath']
        
        # Classes
        for cls in file_data['classes']:
            if not cls['has_docstring']:
                undocumented.append({
                    'type': 'class',
                    'name': cls['name'],
                    'file': filepath.name,
                    'line': cls['lineno']
                })
                
        # Functions
        for func in file_data['functions']:
            if not func['has_docstring'] and not func['is_private']:
                undocumented.append({
                    'type': 'function',
                    'name': func['name'],
                    'file': filepath.name,
                    'line': func['lineno']
                })
                
        # Methods
        for method in file_data['methods']:
            if not method['has_docstring'] and not method['is_private'] and method['name'] != '__init__':
                undocumented.append({
                    'type': 'method',
                    'name': f"{method['class']}.{method['name']}",
                    'file': filepath.name,
                    'line': method['lineno']
                })
                
    return undocumented


def find_well_documented_items(module_data: Dict) -> List[Dict]:
    """Find well-documented items (score >= 4)."""
    well_documented = []
    
    for file_data in module_data['files']:
        filepath = file_data['filepath']
        
        # Check all items
        all_items = (
            [(c, 'class') for c in file_data['classes']] +
            [(f, 'function') for f in file_data['functions']] +
            [(m, 'method') for m in file_data['methods']]
        )
        
        for item, item_type in all_items:
            if item['has_docstring'] and item['docstring_quality']['score'] >= 4:
                name = item['name']
                if item_type == 'method':
                    name = f"{item['class']}.{item['name']}"
                    
                well_documented.append({
                    'type': item_type,
                    'name': name,
                    'file': filepath.name,
                    'line': item['lineno'],
                    'score': item['docstring_quality']['score'],
                    'has_params': item['docstring_quality']['has_params'],
                    'has_returns': item['docstring_quality']['has_returns'],
                    'has_examples': item['docstring_quality']['has_examples']
                })
                
    return well_documented


def main():
    """Main analysis function."""
    src_path = Path("src/driada")
    
    # Modules to analyze
    modules = [
        'dim_reduction',
        'information', 
        'intense',
        'integration',
        'experiment',
        'utils',
        'network',
        'rsa',
        'gdrive'
    ]
    
    print("# DRIADA Docstring Coverage Analysis\n")
    print(f"Analyzing modules in {src_path}")
    print("=" * 80)
    
    all_stats = {}
    
    for module_name in modules:
        module_path = src_path / module_name
        if not module_path.exists():
            print(f"\nModule '{module_name}' not found at {module_path}")
            continue
            
        print(f"\n## Module: {module_name}")
        print("-" * 40)
        
        # Analyze module
        module_data = analyze_module(module_path)
        stats = calculate_stats(module_data)
        all_stats[module_name] = stats
        
        # Print statistics
        print(f"Overall coverage: {stats['overall_coverage']:.1f}%")
        print(f"  Classes: {stats['classes_with_docstrings']}/{stats['total_classes']} ({stats['class_coverage']:.1f}%)")
        print(f"  Functions: {stats['functions_with_docstrings']}/{stats['total_functions']} ({stats['function_coverage']:.1f}%)")
        print(f"  Methods: {stats['methods_with_docstrings']}/{stats['total_methods']} ({stats['method_coverage']:.1f}%)")
        print(f"Average quality score: {stats['avg_quality_score']:.1f}/{stats['max_quality_score']}")
        
        # Find undocumented items
        undocumented = find_undocumented_items(module_data)
        if undocumented[:5]:  # Show first 5
            print("\nTop undocumented items:")
            for item in undocumented[:5]:
                print(f"  - {item['type']} {item['name']} ({item['file']}:{item['line']})")
                
        # Find well-documented items
        well_documented = find_well_documented_items(module_data)
        if well_documented[:3]:  # Show first 3
            print("\nWell-documented examples:")
            for item in well_documented[:3]:
                features = []
                if item['has_params']: features.append('params')
                if item['has_returns']: features.append('returns')
                if item['has_examples']: features.append('examples')
                print(f"  - {item['type']} {item['name']} (score: {item['score']}, has: {', '.join(features)})")
    
    # Summary
    print("\n" + "=" * 80)
    print("## SUMMARY")
    print("=" * 80)
    
    # Sort modules by coverage
    sorted_modules = sorted(all_stats.items(), key=lambda x: x[1]['overall_coverage'], reverse=True)
    
    print("\nModules by documentation coverage:")
    for module, stats in sorted_modules:
        coverage = stats['overall_coverage']
        quality = stats['avg_quality_score']
        status = "✅" if coverage >= 80 else "⚠️" if coverage >= 60 else "❌"
        print(f"  {status} {module:15} {coverage:5.1f}% coverage, {quality:.1f} avg quality")
        
    # Overall statistics
    total_items = sum(s['total_items'] for s in all_stats.values())
    documented_items = sum(s['items_with_docstrings'] for s in all_stats.values())
    overall_coverage = (documented_items / total_items * 100) if total_items > 0 else 0
    
    print(f"\nOverall project documentation coverage: {overall_coverage:.1f}%")
    print(f"Total items: {total_items}, Documented: {documented_items}")
    
    # Recommendations
    print("\n## RECOMMENDATIONS")
    print("-" * 40)
    
    priority_modules = [m for m, s in sorted_modules if s['overall_coverage'] < 60]
    if priority_modules:
        print("\nHigh priority modules (< 60% coverage):")
        for module in priority_modules:
            print(f"  - {module}")
            
    print("\nNext steps:")
    print("1. Focus on documenting public APIs (classes and main functions)")
    print("2. Ensure all docstrings include Parameters and Returns sections")
    print("3. Add examples to commonly used functions")
    print("4. Use consistent docstring style (NumPy style recommended)")


if __name__ == "__main__":
    main()