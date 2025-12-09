import os
import nbformat

# 检查目录
notebooks_dir = 'c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks'

# 获取所有.ipynb文件
notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]

# 检查每个notebook文件
for notebook_file in notebook_files:
    notebook_path = os.path.join(notebooks_dir, notebook_file)
    print(f'检查文件: {notebook_file}')

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        print(f'  ✅ nbformat解析成功')
    except Exception as e:
        print(f'  ❌ nbformat解析错误: {e}')

print('\n检查完成！')