import os
import json
import nbformat

# 修复单个notebook文件

def fix_notebook(file_path):
    print(f'修复文件: {file_path}')

    try:
        # 1. 尝试直接使用json模块读取
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. 尝试解析JSON
        try:
            data = json.loads(content)
            print(f'  ✅ JSON解析成功')
        except json.JSONDecodeError as e:
            print(f'  ❌ JSON解析失败: {e}')
            # 3. 尝试修复JSON格式
            print(f'  🔧 尝试修复JSON格式...')

            # 简单的修复：确保所有字符串都正确闭合
            # 这个方法可能不适用于所有情况，但可以处理一些常见问题
            import re

            # 修复1：确保所有字符串都有闭合引号
            # 这个正则表达式会尝试修复未闭合的字符串
            fixed_content = content

            # 修复2：确保所有\n都正确转义
            # 将所有不在转义序列中的\n替换为\\n
            # 创建一个状态机来处理转义序列和字符串
            in_string = False
            escape = False
            result = []

            for char in content:
                if escape:
                    result.append(char)
                    escape = False
                elif char == '\\':
                    result.append(char)
                    escape = True
                elif char == '"' or char == "'":
                    result.append(char)
                    in_string = not in_string
                elif in_string and char == '\n':
                    # 在字符串中发现换行符，需要转义
                    result.append('\\n')
                else:
                    result.append(char)

            fixed_content = ''.join(result)

            # 尝试再次解析
            try:
                data = json.loads(fixed_content)
                print(f'  ✅ 修复后JSON解析成功')
            except json.JSONDecodeError as e2:
                print(f'  ❌ 修复后JSON解析仍失败: {e2}')
                return False

        # 4. 使用nbformat重新写入
        nb = nbformat.from_dict(data)
        nbformat.write(nb, file_path)
        print(f'  ✅ 使用nbformat重写成功')
        return True

    except Exception as e:
        print(f'  ❌ 修复过程中发生错误: {e}')
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
        fix_notebook(notebook_path)

    print('\n所有文件修复完成！')

if __name__ == '__main__':
    main()