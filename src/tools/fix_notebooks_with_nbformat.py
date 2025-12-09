import os
import nbformat

# 检查目录
notebooks_dir = 'c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks'

# 获取所有.ipynb文件
notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]

# 使用nbformat修复每个notebook文件
for notebook_file in notebook_files:
    notebook_path = os.path.join(notebooks_dir, notebook_file)
    print(f'修复文件: {notebook_file}')

    try:
        # 首先，我们需要手动修复文件中的语法错误
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 简单修复：替换所有未转义的换行符为转义的换行符
        # 这是一个保守的修复，只处理明显的问题
        import re

        # 修复字符串中的\n为\\n
        # 使用正则表达式查找所有在字符串中的\n
        # 这个正则表达式匹配在引号内的\n，并将其替换为\\n
        # 注意：这是一个简化的实现，可能无法处理所有情况
        def fix_string_newlines(text):
            in_single_quote = False
            in_double_quote = False
            escaped = False
            result = []

            for char in text:
                if escaped:
                    # 如果是转义字符，直接添加
                    result.append(char)
                    escaped = False
                    continue

                if char == '\\':
                    # 开始转义序列
                    result.append(char)
                    escaped = True
                    continue

                if char == "'" and not in_double_quote:
                    # 单引号字符串边界
                    result.append(char)
                    in_single_quote = not in_single_quote
                    continue

                if char == '"' and not in_single_quote:
                    # 双引号字符串边界
                    result.append(char)
                    in_double_quote = not in_double_quote
                    continue

                if (in_single_quote or in_double_quote) and char == 'n' and result and result[-1] == '\\':
                    # 在字符串中发现\n序列，确保它被正确转义
                    # 检查前一个字符是否已经是\
                    if len(result) > 1 and result[-2] == '\\':
                        # 已经是\\n，不需要修改
                        result.append(char)
                    else:
                        # 添加一个额外的\
                        result.append('\\')
                        result.append(char)
                else:
                    # 其他字符直接添加
                    result.append(char)

            return ''.join(result)

        # 修复内容
        fixed_content = fix_string_newlines(content)

        # 尝试使用JSON解析修复后的内容
        import json
        json_data = json.loads(fixed_content)

        # 使用nbformat重新写入文件
        nb = nbformat.from_dict(json_data)
        nbformat.write(nb, notebook_path)

        print(f'  ✅ 修复成功')

    except Exception as e:
        print(f'  ❌ 修复失败: {e}')
        import traceback
        traceback.print_exc()

print('\n所有文件修复完成！')