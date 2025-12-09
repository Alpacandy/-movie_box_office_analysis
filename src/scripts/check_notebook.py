#!/usr/bin/env python3
"""
Simple script to check notebook content
"""

import json
import os

# 使用相对路径定位notebook文件
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
notebook_path = os.path.join(project_root, "notebooks", "01_data_acquisition.ipynb")

print(f"Reading notebook: {notebook_path}")
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