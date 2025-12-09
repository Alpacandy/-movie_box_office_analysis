import os
import json

# 检查目录
notebooks_dir = 'c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks'

# 获取所有.ipynb文件
notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]

# 检查每个notebook文件中的字符串字面量
for notebook_file in notebook_files:
    notebook_path = os.path.join(notebooks_dir, notebook_file)
    print(f'检查文件: {notebook_file}')

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查文件中是否存在未转义的换行符在字符串中
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            # 检查是否在字符串字面量内部
            in_string = False
            escape = False

            for char in line:
                if escape:
                    escape = False
                    continue

                if char == '\\':
                    escape = True
                    continue

                if char == '"' or char == "'":
                    in_string = not in_string

            # 如果行末仍在字符串中，说明可能有问题
            if in_string:
                print(f'  ❌ 第{line_num}行可能存在未终止的字符串字面量')
                print(f'     行内容: {line[:100]}...')

        print(f'  ✅ 字符串字面量检查完成')

    except Exception as e:
        print(f'  ❌ 检查失败: {e}')

print('\n所有文件检查完成！')