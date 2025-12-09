import os
import json
import nbformat

# 检查目录
notebooks_dir = 'c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks'

# 获取所有.ipynb文件
notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]

# 修复每个notebook文件
for notebook_file in notebook_files:
    notebook_path = os.path.join(notebooks_dir, notebook_file)
    print(f'修复文件: {notebook_file}')

    try:
        # 使用nbformat读取文件
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 重新写入文件，确保JSON格式正确
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        print(f'  ✅ 修复成功')

    except Exception as e:
        print(f'  ❌ 修复失败: {e}')

print('\n修复完成！')