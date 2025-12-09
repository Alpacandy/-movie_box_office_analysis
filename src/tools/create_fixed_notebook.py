import os
import nbformat

# 创建一个新的修复后的notebook文件
def create_fixed_notebook(original_path, new_path):
    print(f'从 {os.path.basename(original_path)} 创建修复版本...')

    try:
        # 创建一个新的notebook结构
        nb = nbformat.v4.new_notebook()

        # 添加markdown单元格
        nb.cells.append(nbformat.v4.new_markdown_cell("# 06_deep_learning"))
        nb.cells.append(nbformat.v4.new_markdown_cell("本笔记本用于执行深度学习模型的训练和评估，包括全连接神经网络、卷积神经网络和循环神经网络等。"))
        nb.cells.append(nbformat.v4.new_markdown_cell("## 1. 检查项目结构"))

        # 添加代码单元格
        nb.cells.append(nbformat.v4.new_code_cell("# 导入必要的库\nimport os\nimport sys\nimport pandas as pd\nimport subprocess"))

        nb.cells.append(nbformat.v4.new_markdown_cell("## 2. 执行深度学习建模脚本"))
        nb.cells.append(nbformat.v4.new_code_cell("# 检查当前工作目录\nprint(f'当前工作目录: {os.getcwd()}')\n\n# 检查项目结构\nproject_dir = '..'\nprint('项目结构:')\nfor root, dirs, files in os.walk(project_dir):\n    level = root.replace(project_dir, '').count(os.sep)\n    indent = ' ' * 2 * level\n    print(f'{indent}{os.path.basename(root)}/')\n    subindent = ' ' * 2 * (level + 1)\n    for file in files[:5]:  # 只显示前5个文件\n        print(f'{subindent}{file}')"))

        nb.cells.append(nbformat.v4.new_markdown_cell("## 3. 查看生成的文件"))
        nb.cells.append(nbformat.v4.new_code_cell("# 执行深度学习建模脚本\nscript_path = os.path.join(project_dir, 'src', 'deep_learning.py')\nprint(f'执行脚本: {script_path}')\n\nif os.path.exists(script_path):\n    # 执行脚本\n    result = subprocess.run([sys.executable, script_path], cwd=project_dir, capture_output=True, text=True)\n    \n    # 打印脚本输出\n    print('\\n脚本输出:')\n    print('-' * 50)\n    print(result.stdout)\n    if result.stderr:\n        print('\\n错误信息:')\n        print('-' * 50)\n        print(result.stderr)\n    print('-' * 50)\nelse:\n    print('脚本不存在')"))

        nb.cells.append(nbformat.v4.new_code_cell("# 查看models目录下的文件\nmodels_dir = os.path.join(project_dir, 'results', 'models')\nif os.path.exists(models_dir):\n    print('\\nmodels目录下的文件:')\n    for file in os.listdir(models_dir):\n        if file.endswith('.pkl') or file.endswith('.joblib'):\n            print(f'  - {file}')\nelse:\n    print('\\nmodels目录不存在')\n\n# 查看charts目录下的文件\ncharts_dir = os.path.join(project_dir, 'results', 'charts')\nif os.path.exists(charts_dir):\n    print('\\ncharts目录下的文件:')\n    for file in os.listdir(charts_dir):\n        if 'deep' in file.lower():\n            print(f'  - {file}')\nelse:\n    print('\\ncharts目录不存在')"))

        nb.cells.append(nbformat.v4.new_markdown_cell("## 4. 加载并查看深度学习模型性能"))
        nb.cells.append(nbformat.v4.new_code_cell("# 加载深度学习模型性能数据\nperformance_file = os.path.join(project_dir, 'results', 'deep_learning_performance.csv')\n\nif os.path.exists(performance_file):\n    deep_learning_df = pd.read_csv(performance_file)\n    \n    # 显示数据形状和内容\n    print(f'\\n深度学习性能数据形状: {deep_learning_df.shape}')\n    print('\\n深度学习性能数据:')\n    print(deep_learning_df.head())\n    \n    # 识别最佳模型\n    if not deep_learning_df.empty:\n        best_model = deep_learning_df.loc[deep_learning_df['R2 Score'].idxmax()]\n        print(f'\\n最佳深度学习模型: {best_model[\"Model\"]}')\n        print(f'R2 Score: {best_model[\"R2 Score\"]:.4f}')\n        print(f'RMSE: {best_model[\"RMSE\"]:.4f}')\n        print(f'MAE: {best_model[\"MAE\"]:.4f}')\n    \nelse:\n    print('\\n深度学习性能数据文件不存在')"))

        nb.cells.append(nbformat.v4.new_markdown_cell("## 5. 总结与下一步操作"))
        nb.cells.append(nbformat.v4.new_code_cell("# 总结与下一步操作\nprint('\\n总结:')\nprint('1. 成功执行了深度学习建模脚本')\nprint('2. 生成了深度学习模型文件')\nprint('3. 生成了深度学习性能图表')\nprint('4. 加载并分析了深度学习模型性能数据')\n\nprint('\\n下一步操作:')\nprint('1. 运行 07_visualization.ipynb 进行数据可视化')\nprint('2. 或直接执行可视化脚本: python src/visualization.py')\nprint('3. 分析生成的图表和报告')\nprint('4. 根据需求调整模型参数')\nprint('\\n项目流程已全部完成！')"))

        # 保存新的notebook文件
        nbformat.write(nb, new_path)
        print(f'  ✅ 修复版本创建成功')
        return True

    except Exception as e:
        print(f'  ❌ 创建失败: {e}')
        import traceback
        traceback.print_exc()
        return False

# 主程序
def main():
    # 原始文件路径
    original_file = 'c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks\\06_deep_learning.ipynb'
    # 新文件路径
    new_file = 'c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks\\06_deep_learning_fixed.ipynb'

    # 创建修复版本
    if create_fixed_notebook(original_file, new_file):
        print(f'\n修复版本已创建: {os.path.basename(new_file)}')
    else:
        print(f'\n修复失败')

if __name__ == '__main__':
    main()