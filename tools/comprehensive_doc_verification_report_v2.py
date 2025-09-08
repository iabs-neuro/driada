#!/usr/bin/env python3
"""
Improved DOC_VERIFIED verification report for DRIADA project.
Properly matches DOC_VERIFIED labels to their corresponding entities.
"""

import os
import ast
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class VerificationAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze entities and their DOC_VERIFIED status."""
    
    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.entities = []  # List of (type, name, line, has_doc_verified)
        self.current_class = None
        self.class_has_doc_verified = {}  # Track which classes have DOC_VERIFIED
        
    def visit_FunctionDef(self, node):
        """Count function definitions and check for DOC_VERIFIED."""
        if self.current_class:
            name = f"{self.current_class}.{node.name}"
            entity_type = "method"
            
            # Skip __init__ if parent class has DOC_VERIFIED
            if (node.name == '__init__' and 
                self.current_class in self.class_has_doc_verified and 
                self.class_has_doc_verified[self.current_class]):
                # Check if __init__ has its own substantial docstring
                init_docstring = ast.get_docstring(node)
                if not init_docstring or len(init_docstring.strip()) < 50:
                    # Skip this __init__ - it's covered by class documentation
                    return
        else:
            name = node.name
            entity_type = "function"
            
        # Check if this function has DOC_VERIFIED
        has_doc_verified = self._has_doc_verified(node)
        self.entities.append((entity_type, name, node.lineno, has_doc_verified))
        
        # Don't visit nested functions inside this function
        # to avoid counting them as separate entities
        
    def visit_AsyncFunctionDef(self, node):
        """Count async function definitions."""
        if self.current_class:
            name = f"{self.current_class}.{node.name}"
            entity_type = "method"
            
            # Skip __init__ if parent class has DOC_VERIFIED (same logic as FunctionDef)
            if (node.name == '__init__' and 
                self.current_class in self.class_has_doc_verified and 
                self.class_has_doc_verified[self.current_class]):
                init_docstring = ast.get_docstring(node)
                if not init_docstring or len(init_docstring.strip()) < 50:
                    return
        else:
            name = node.name
            entity_type = "function"
            
        has_doc_verified = self._has_doc_verified(node)
        self.entities.append((entity_type, name, node.lineno, has_doc_verified))
        
    def visit_ClassDef(self, node):
        """Count class definitions."""
        name = node.name
        has_doc_verified = self._has_doc_verified(node)
        self.entities.append(("class", name, node.lineno, has_doc_verified))
        
        # Track if this class has DOC_VERIFIED
        self.class_has_doc_verified[name] = has_doc_verified
        
        # Visit methods inside the class
        old_class = self.current_class
        self.current_class = node.name
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)
        self.current_class = old_class
        
    def _has_doc_verified(self, node) -> bool:
        """Check if a node's docstring contains DOC_VERIFIED."""
        # Get the docstring
        docstring = ast.get_docstring(node, clean=False)
        if docstring and 'DOC_VERIFIED' in docstring:
            return True
            
        # Also check for DOC_VERIFIED comment right after the docstring
        # This handles cases where DOC_VERIFIED is outside the docstring
        if hasattr(node, 'body') and node.body:
            # Find the line after the docstring
            first_stmt = node.body[0]
            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, (ast.Str, ast.Constant)):
                # This is the docstring, check lines after it
                docstring_end_line = first_stmt.end_lineno
                # Check next few lines for DOC_VERIFIED
                for i in range(docstring_end_line, min(docstring_end_line + 5, len(self.source_lines))):
                    if i < len(self.source_lines) and 'DOC_VERIFIED' in self.source_lines[i]:
                        return True
                        
        return False

def analyze_file(file_path: str) -> Optional[Dict]:
    """Analyze a single Python file for DOC_VERIFIED status."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            source_lines = content.splitlines()
        
        # Parse AST
        tree = ast.parse(content, filename=file_path)
        analyzer = VerificationAnalyzer(source_lines)
        
        # Only visit top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                analyzer.visit(node)
        
        # Count verified entities
        total_entities = len(analyzer.entities)
        verified_entities = sum(1 for _, _, _, has_doc in analyzer.entities if has_doc)
        
        # Separate counts by type
        functions = sum(1 for t, _, _, _ in analyzer.entities if t == "function")
        classes = sum(1 for t, _, _, _ in analyzer.entities if t == "class")
        methods = sum(1 for t, _, _, _ in analyzer.entities if t == "method")
        
        return {
            'functions': functions,
            'classes': classes,
            'methods': methods,
            'total_entities': total_entities,
            'verified': verified_entities,
            'percentage': (verified_entities / total_entities * 100) if total_entities > 0 else 0,
            'remaining': total_entities - verified_entities,
            'is_complete': verified_entities >= total_entities and total_entities > 0,
            'entities': analyzer.entities
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def create_comprehensive_report():
    """Generate comprehensive verification report."""
    all_files = []
    
    # Scan all Python files
    for root, dirs, files in os.walk('src/driada'):
        # Skip test directories
        if 'test' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                
                # Skip __init__.py files unless they have substantial content
                if file == '__init__.py':
                    with open(file_path, 'r') as f:
                        if len(f.read().strip()) < 50:
                            continue
                
                analysis = analyze_file(file_path)
                if analysis and analysis['total_entities'] > 0:
                    # Extract module name
                    rel_path = os.path.relpath(file_path, 'src/driada')
                    module = rel_path.split(os.sep)[0] if os.sep in rel_path else 'root'
                    
                    all_files.append({
                        'module': module,
                        'file': file,
                        'path': file_path,
                        'functions': analysis['functions'],
                        'classes': analysis['classes'],
                        'methods': analysis['methods'],
                        'total_entities': analysis['total_entities'],
                        'verified': analysis['verified'],
                        'percentage': analysis['percentage'],
                        'remaining': analysis['remaining'],
                        'is_complete': analysis['is_complete'],
                        'entities': analysis['entities']
                    })
    
    return all_files

def print_report(files_data: List[Dict]):
    """Print the comprehensive verification report."""
    # Sort by module, then by completion status, then by percentage
    files_data.sort(key=lambda x: (
        x['module'],
        not x['is_complete'],  # Complete files first
        -x['percentage'],
        x['file']
    ))
    
    print('=' * 120)
    print('DRIADA IMPROVED DOC_VERIFIED VERIFICATION REPORT')
    print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 120)
    
    # Show files with missing DOC_VERIFIED
    incomplete_files = [f for f in files_data if not f['is_complete']]
    if incomplete_files:
        print()
        print('📋 FILES NEEDING ATTENTION:')
        print('-' * 120)
        for f in incomplete_files:  # Show all incomplete files
            print(f"\n{f['module']}/{f['file']} - {f['remaining']} entities need DOC_VERIFIED:")
            # Show which entities are missing
            for entity_type, name, line, has_doc in f['entities']:
                if not has_doc:
                    print(f"  ❌ {entity_type:8s} {name:40s} (line {line})")
    
    # Summary statistics
    print()
    print('=' * 120)
    print('SUMMARY STATISTICS:')
    print('-' * 120)
    
    total_files = len(files_data)
    complete_files = sum(1 for f in files_data if f['is_complete'])
    total_entities = sum(f['total_entities'] for f in files_data)
    verified_entities = sum(f['verified'] for f in files_data)
    
    print(f"Total files analyzed: {total_files}")
    print(f"Fully verified files: {complete_files} ({complete_files/total_files*100:.1f}%)")
    print(f"Total entities: {total_entities}")
    print(f"Verified entities: {verified_entities} ({verified_entities/total_entities*100:.1f}%)")
    print(f"Remaining entities: {total_entities - verified_entities}")

def print_debug_for_file(file_path: str):
    """Debug a specific file to see entity detection."""
    print(f"\nDEBUG ANALYSIS FOR: {file_path}")
    print("-" * 80)
    
    analysis = analyze_file(file_path)
    if analysis:
        print(f"Total entities: {analysis['total_entities']}")
        print(f"Verified: {analysis['verified']}")
        print(f"Functions: {analysis['functions']}, Classes: {analysis['classes']}, Methods: {analysis['methods']}")
        print("\nEntity details:")
        for entity_type, name, line, has_doc in analysis['entities']:
            status = "✅" if has_doc else "❌"
            print(f"  {status} {entity_type:8s} {name:40s} (line {line})")

if __name__ == '__main__':
    # Debug specific files first
    print("\n" + "="*80)
    print("DEBUGGING SPECIFIC FILES:")
    print("="*80)
    
    print_debug_for_file('src/driada/intense/intense_base.py')
    print_debug_for_file('src/driada/dimensionality/intrinsic.py')
    
    # Then run full report
    print("\n" + "="*80)
    print("FULL REPORT:")
    print("="*80)
    
    files_data = create_comprehensive_report()
    print_report(files_data)