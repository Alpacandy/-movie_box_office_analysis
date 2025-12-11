#!/usr/bin/env python3
"""
Script to verify original notebook content has been updated
"""

import json

# Path to the original notebook file
notebook_path = r'C:\羊驼\pro\analysis\movie_box_office_analysis\notebooks\01_data_acquisition.ipynb'

print(f"Verifying original notebook content: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Get all code cells
code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']

print(f"Total code cells: {len(code_cells)}")

# Check first code cell (should not have subprocess import)
first_cell = code_cells[0]
first_cell_content = '\n'.join(first_cell['source'])
print("\nFirst code cell content (should NOT have subprocess import):")
print(first_cell_content)

# Check third code cell (should use direct import instead of subprocess)
third_cell = code_cells[2]
third_cell_content = '\n'.join(third_cell['source'])
print("\nThird code cell content (should use direct import):")
print(third_cell_content)