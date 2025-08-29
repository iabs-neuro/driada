#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, 'tools')
from comprehensive_doc_verification_report_v2 import analyze_file
from collections import defaultdict

# Get all Python files
all_files = []
for root, dirs, files in os.walk('src/driada'):
    if 'test' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py') and not file.startswith('test_'):
            file_path = os.path.join(root, file)
            if file == '__init__.py':
                with open(file_path, 'r') as f:
                    if len(f.read().strip()) < 50:
                        continue
            all_files.append(file_path)

# Analyze and show all files with remaining entities
files_with_remaining = []
total_remaining = 0

for file_path in all_files:
    analysis = analyze_file(file_path)
    if analysis and analysis['remaining'] > 0:
        files_with_remaining.append((file_path, analysis['remaining']))
        total_remaining += analysis['remaining']

# Sort by number of remaining entities
files_with_remaining.sort(key=lambda x: x[1], reverse=True)

print("ALL FILES WITH UNVERIFIED ENTITIES:")
print("="*80)
for file_path, count in files_with_remaining:
    rel_path = os.path.relpath(file_path, 'src/driada')
    print(f"{rel_path}: {count} entities")

print(f"\nTotal files with unverified entities: {len(files_with_remaining)}")
print(f"Total remaining entities: {total_remaining}")