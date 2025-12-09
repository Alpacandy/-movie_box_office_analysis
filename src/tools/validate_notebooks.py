#!/usr/bin/env python3
"""
Enhanced Notebook Validation Script

This script validates that Jupyter notebooks in the project:
1. Don't reference non-existent files like src/data_preprocessing.py
2. Use the correct modular imports
3. Don't use outdated subprocess calls to run scripts
4. Follow the current project structure
5. Use the appropriate class names
6. Don't have unnecessary imports
7. Provide correct next steps in conclusions
"""

import os
import sys
import json
import re

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Define valid module mappings
VALID_MODULE_MAPPINGS = {
    'data_preprocessing': {
        'old_script': 'src/data_preprocessing.py',
        'new_import': 'from src.data_preprocessing import DataPreprocessing',
        'class_name': 'DataPreprocessing'
    },
    'data_acquisition': {
        'old_script': 'src/data_acquisition.py',
        'new_import': 'from src.data_acquisition import DataAcquisition',
        'class_name': 'DataAcquisition'
    },
    'eda_analysis': {
        'old_script': 'src/eda_analysis.py',
        'new_import': 'from src.eda_analysis import EDAnalysis',
        'class_name': 'EDAnalysis'
    },
    'feature_engineering': {
        'old_script': 'src/feature_engineering.py',
        'new_import': 'from src.feature_engineering import FeatureEngineering',
        'class_name': 'FeatureEngineering'
    },
    'modeling': {
        'old_script': 'src/modeling.py',
        'new_import': 'from src.modeling import TraditionalModeling',
        'class_name': 'TraditionalModeling'
    },
    'deep_learning': {
        'old_script': 'src/deep_learning.py',
        'new_import': 'from src.deep_learning import DeepLearningModeling',
        'class_name': 'DeepLearningModeling'
    },
    'visualization': {
        'old_script': 'src/visualization.py',
        'new_import': 'from src.visualization import MovieVisualizer',
        'class_name': 'MovieVisualizer'
    }
}

# Define files to exclude from validation
EXCLUDED_FILES = {
    '06_deep_learning_fixed.ipynb',
    'test_fixed.ipynb',
    'test_simple_notebook.ipynb',
    'test_simple_output.ipynb',
    'valid_deep_learning.ipynb',
    'new_deep_learning.ipynb'
}

def validate_notebook(notebook_path):
    """Validate a single notebook file"""
    print(f"\nValidating notebook: {notebook_path}")

    issues = []

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Print notebook structure for debugging
        print(f"  Total cells: {len(notebook.get('cells', []))}")

        # Collect all code cell content for global checks
        all_code = ''
        code_cells = []
        code_cell_idx = 0
        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'code':
                code_cell_idx += 1
                source = '\n'.join(cell['source'])
                code_cells.append((code_cell_idx, source))
                all_code += source + '\n'

                print(f"  Code Cell {code_cell_idx} content preview: {source[:100]}...")

        # 1. Check for unnecessary subprocess import globally
        has_subprocess_import = False
        import_cell_idx = 0
        for idx, source in code_cells:
            if 'import subprocess' in source:
                has_subprocess_import = True
                import_cell_idx = idx
                break

        if has_subprocess_import and not re.search(r'subprocess\.run\s*\(', all_code):
            issues.append({
                'cell': import_cell_idx,
                'issue': 'Unnecessary import: subprocess',
                'fix': 'Remove unused subprocess import'
            })

        # Check each code cell for other issues
        for code_cell_idx, source in code_cells:
            # 2. Check for references to old single-file structure
            for module, mappings in VALID_MODULE_MAPPINGS.items():
                if re.search(re.escape(mappings['old_script']), source):
                    issues.append({
                        'cell': code_cell_idx,
                        'issue': f'References old single-file structure: {mappings["old_script"]}',
                        'fix': f'Use modular import: {mappings["new_import"]}'
                    })

            # 3. Check for subprocess calls to run scripts
            subprocess_matches = re.finditer(r'subprocess\.run\([^)]*?([a-zA-Z_]+\.py)', source)
            for match in subprocess_matches:
                script_name = match.group(1)
                module_name = script_name.replace('.py', '')
                if module_name in VALID_MODULE_MAPPINGS:
                    mappings = VALID_MODULE_MAPPINGS[module_name]
                    issues.append({
                        'cell': code_cell_idx,
                        'issue': f'Uses subprocess to run {script_name}',
                        'fix': f'Import and use {mappings["class_name"]} class directly'
                    })

            # 4. Check for proper modular imports
            for module, mappings in VALID_MODULE_MAPPINGS.items():
                import_pattern = re.escape(mappings['new_import'].split(' import ')[0])
                if re.search(import_pattern, source):
                    if mappings['class_name'] not in source:
                        issues.append({
                            'cell': code_cell_idx,
                            'issue': f'Incomplete import: Missing {mappings["class_name"]} usage',
                            'fix': f'Use {mappings["new_import"]} and instantiate {mappings["class_name"]} class'
                        })

            # 5. Check for incorrect next steps in conclusions
            if any(phrase in source.lower() for phrase in ['下一步操作', 'next step', 'conclusion']):
                for module, mappings in VALID_MODULE_MAPPINGS.items():
                    if f'directly execute {mappings["old_script"]}' in source:
                        issues.append({
                            'cell': code_cell_idx,
                            'issue': f'Outdated next step: Suggests executing {mappings["old_script"]}',
                            'fix': f'Suggest using {mappings["class_name"]} class instead'
                        })

    except json.JSONDecodeError:
        issues.append({
            'cell': 0,
            'issue': 'Invalid JSON format',
            'fix': 'Ensure notebook is a valid JSON file'
        })
    except Exception as e:
        issues.append({
            'cell': 0,
            'issue': f'Error parsing notebook: {e}',
            'fix': 'Check notebook file integrity'
        })

    return issues

def validate_all_notebooks(notebooks_dir):
    """Validate all notebooks in a directory"""
    print(f"\n=== Validating notebooks in {notebooks_dir} ===")

    all_issues = []

    for file in os.listdir(notebooks_dir):
        if file.endswith('.ipynb') and not file.startswith('.'):
            # Skip excluded files
            if file in EXCLUDED_FILES:
                print(f"  ⏩ Skipping excluded notebook: {file}")
                continue

            notebook_path = os.path.join(notebooks_dir, file)
            issues = validate_notebook(notebook_path)

            if issues:
                all_issues.extend([(notebook_path, issue) for issue in issues])
                print(f"  ❌ Found {len(issues)} issues in {file}")
                for issue in issues:
                    print(f"    Cell {issue['cell']}: {issue['issue']}")
                    print(f"    Fix: {issue['fix']}")
            else:
                print(f"  ✓ No issues found in {file}")

    return all_issues

def main():
    """Main function"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Notebook Validation Script')
    parser.add_argument('--notebook', type=str, help='Path to a specific notebook to validate')
    parser.add_argument('--list-excluded', action='store_true', help='List excluded notebook files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # List excluded files if requested
    if args.list_excluded:
        print("Excluded notebook files:")
        for file in sorted(EXCLUDED_FILES):
            print(f"  - {file}")
        sys.exit(0)

    # Determine notebooks directory
    notebooks_dir = os.path.join(project_root, 'notebooks')

    if not os.path.exists(notebooks_dir):
        print(f"Error: Notebooks directory not found at {notebooks_dir}")
        sys.exit(1)

    # Validate specific notebook if provided
    if args.notebook:
        notebook_path = args.notebook
        if not os.path.isabs(notebook_path):
            notebook_path = os.path.join(project_root, notebook_path)

        if not os.path.exists(notebook_path):
            print(f"Error: Notebook not found at {notebook_path}")
            sys.exit(1)

        issues = validate_notebook(notebook_path)
        all_issues = [(notebook_path, issue) for issue in issues] if issues else []
    else:
        # Validate all notebooks
        all_issues = validate_all_notebooks(notebooks_dir)

    # Print summary
    print("\n=== Validation Summary ===")

    if args.notebook:
        print(f"Notebook checked: {os.path.basename(notebook_path)}")
    else:
        total_notebooks = len([f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb') and not f.startswith('.')])
        checked_notebooks = total_notebooks - len(EXCLUDED_FILES)
        print(f"Total notebooks: {total_notebooks}")
        print(f"Excluded notebooks: {len(EXCLUDED_FILES)}")
        print(f"Notebooks checked: {checked_notebooks}")

    print(f"Total issues found: {len(all_issues)}")

    if all_issues:
        print("\nIssues were found. Please update the notebooks to match the current project structure.")
        sys.exit(1)
    else:
        print("\n✓ All checked notebooks are valid!")
        sys.exit(0)

if __name__ == "__main__":
    main()