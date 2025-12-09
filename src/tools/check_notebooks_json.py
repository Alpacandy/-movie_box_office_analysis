import os
import json

# 检查目录
notebooks_dir = 'c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks'

# 获取所有.ipynb文件
notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]

# 检查每个notebook文件的JSON格式
for notebook_file in notebook_files:
    notebook_path = os.path.join(notebooks_dir, notebook_file)
    print(f'检查文件: {notebook_file}')

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f'  ✅ JSON格式正确')
    except json.JSONDecodeError as e:
        print(f'  ❌ JSON格式错误: {e}')
    except Exception as e:
        print(f'  ❌ 其他错误: {e}')

print('\n检查完成！')