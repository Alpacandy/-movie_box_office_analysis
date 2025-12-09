import os
import json

# 检查目录
notebooks_dir = 'c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks'

# 获取所有.ipynb文件
notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]

# 修复每个notebook文件中的换行符问题
for notebook_file in notebook_files:
    notebook_path = os.path.join(notebooks_dir, notebook_file)
    print(f'修复文件: {notebook_file}')

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复字符串中的未转义换行符
        # 将所有不在转义序列中的\n替换为\\n
        # 创建一个新的内容字符串
        new_content = ''
        i = 0
        in_string = False
        escape = False

        while i < len(content):
            char = content[i]

            if escape:
                # 如果是转义序列，直接添加
                new_content += char
                escape = False
                i += 1
                continue

            if char == '\\':
                # 开始转义序列
                new_content += char
                escape = True
                i += 1
                continue

            if char == '"' or char == "'":
                # 字符串边界
                new_content += char
                in_string = not in_string
                i += 1
                continue

            if in_string and char == '\n':
                # 在字符串中发现换行符，需要转义
                new_content += '\\n'
                i += 1
                continue

            # 其他字符直接添加
            new_content += char
            i += 1

        # 将修复后的内容写回文件
        with open(notebook_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f'  ✅ 修复成功')

    except Exception as e:
        print(f'  ❌ 修复失败: {e}')
        import traceback
        traceback.print_exc()

print('\n所有文件修复完成！')