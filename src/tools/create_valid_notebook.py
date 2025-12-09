import nbformat
import json
import os

# 创建一个新的notebook对象
nb = nbformat.v4.new_notebook()

# 添加markdown单元格
markdown_cell = nbformat.v4.new_markdown_cell("# 深度学习模型训练和评估")
nb.cells.append(markdown_cell)

# 添加code单元格
code_cells = [
    "import os\nimport sys\nimport pandas as pd\nimport subprocess",
    "project_dir = '.'\nscript_path = os.path.join(project_dir, 'movie_box_office_analysis', 'src', 'deep_learning.py')\nprint('执行脚本:', script_path)\n\nif os.path.exists(script_path):\n    result = subprocess.run([sys.executable, script_path], cwd=project_dir, capture_output=True, text=True)\n    print('脚本输出:')\n    print(result.stdout)\n    if result.stderr:\n        print('错误信息:')\n        print(result.stderr)\nelse:\n    print('脚本不存在')",
    "# 查看models目录\nmodels_dir = os.path.join(project_dir, 'movie_box_office_analysis', 'results', 'models')\nprint('models目录文件:')\nif os.path.exists(models_dir):\n    for f in os.listdir(models_dir):\n        print(f'  {f}')\nelse:\n    print('  models目录不存在')",
    "# 查看charts目录\ncharts_dir = os.path.join(project_dir, 'movie_box_office_analysis', 'results', 'charts')\nprint('charts目录文件:')\nif os.path.exists(charts_dir):\n    for f in os.listdir(charts_dir):\n        print(f'  {f}')\nelse:\n    print('  charts目录不存在')",
    "# 加载性能数据\nperf_file = os.path.join(project_dir, 'movie_box_office_analysis', 'results', 'deep_learning_performance.csv')\nif os.path.exists(perf_file):\n    df = pd.read_csv(perf_file)\n    print('性能数据:')\n    print(df)\nelse:\n    print('性能数据文件不存在')"
]

for code in code_cells:
    code_cell = nbformat.v4.new_code_cell(code)
    nb.cells.append(code_cell)

# 设置metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.8.0"
    }
}

# 保存notebook文件
notebook_path = "c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks\\06_deep_learning.ipynb"
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"已成功修复06_deep_learning.ipynb")
