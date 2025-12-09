import os
import re

# 修复notebook文件中的JSON数组逗号问题
def fix_notebook_file(file_path):
    print(f'修复文件: {file_path}')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复1: 确保所有JSON数组元素都有逗号分隔符
        # 查找所有在"之后的换行符，确保前面有逗号
        fixed_content = content

        # 使用正则表达式查找所有在"之后的换行符，确保前面有逗号
        # 匹配模式: "...\n"...
        pattern = r'"(.*?)\\n"(?!,)'  # 匹配"...\n"但后面没有逗号的情况

        def replace_with_comma(match):
            return match.group(0) + ','

        fixed_content = re.sub(pattern, replace_with_comma, fixed_content, flags=re.DOTALL)

        # 修复2: 确保字符串中的\n都正确转义为\\n
        # 创建状态机来处理转义序列和字符串
        in_string = False
        escape = False
        result = []

        for char in fixed_content:
            if escape:
                result.append(char)
                escape = False
            elif char == '\\':
                result.append(char)
                escape = True
            elif char == '"':
                result.append(char)
                in_string = not in_string
            elif in_string and char == '\n':
                # 在字符串中发现换行符，需要转义
                result.append('\\n')
            else:
                result.append(char)

        fixed_content = ''.join(result)

        # 修复3: 移除JSON数组末尾多余的逗号
        # 匹配模式: ,\n    ]
        fixed_content = re.sub(r',\\n\s*\]', r'\\n    ]', fixed_content)

        # 保存修复后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        print(f'  ✅ 文件修复成功')

        # 尝试使用JSON解析验证
        import json
        try:
            json.loads(fixed_content)
            print(f'  ✅ JSON解析验证成功')
            return True
        except json.JSONDecodeError as e:
            print(f'  ❌ JSON解析验证失败: {e}')
            return False

    except Exception as e:
        print(f'  ❌ 修复失败: {e}')
        import traceback
        traceback.print_exc()
        return False

# 主程序
def main():
    # 检查目录
    notebooks_dir = 'c:\\羊驼\\pro\\analysis\\movie_box_office_analysis\\notebooks'

    # 获取所有.ipynb文件
    notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]

    # 修复每个notebook文件
    for notebook_file in notebook_files:
        notebook_path = os.path.join(notebooks_dir, notebook_file)
        fix_notebook_file(notebook_path)

    print('\n所有文件修复完成！')

if __name__ == '__main__':
    main()