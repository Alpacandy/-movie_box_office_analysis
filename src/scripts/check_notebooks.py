import json
import os

# 使用相对路径定位notebooks目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
notebooks_dir = os.path.join(project_root, "notebooks")

for filename in os.listdir(notebooks_dir):
    if filename.endswith(".ipynb"):
        file_path = os.path.join(notebooks_dir, filename)
        print(f"Checking {filename}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ {filename} is a valid JSON file")
        except json.JSONDecodeError as e:
            print(f"✗ {filename} has JSON syntax error: {e}")
        except Exception as e:
            print(f"✗ {filename} has error: {e}")
        print()
