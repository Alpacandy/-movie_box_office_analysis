#!/usr/bin/env python3
"""
Simple script to check fixed notebook content
"""

import json
import os

# Path to the fixed notebook file
notebook_path = r'C:\羊驼\pro\analysis\movie_box_office_analysis\notebooks\01_data_acquisition_fixed.ipynb'

print(f"Reading fixed notebook: {notebook_path}")
print(f"File exists: {os.path.exists(notebook_path)}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Total cells: {len(notebook.get('cells', []))}")

# Check each code cell
for i, cell in enumerate(notebook.get('cells', [])):
    if cell['cell_type'] == 'code':
        source = '\n'.join(cell['source'])
        print(f"\nCode Cell {i+1}:")
        print(source[:200] + '...' if len(source) > 200 else source)