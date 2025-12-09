import os
import json

# 手动修复notebook文件的JSON格式
def manual_fix_notebook(file_path):
    print(f'修复文件: {file_path}')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. 将所有\n替换为\\n，然后将所有\\n替换为\\\n
        # 这是一个简单但有效的方法，因为在JSON字符串中，\n必须被转义为\\n
        # 步骤1: 先将所有\n替换为\\n
        # 这里我们使用简单的字符串替换，而不是状态机，因为状态机在处理复杂JSON时可能有问题

        # 首先，将所有\n替换为\\n
        # 注意：这会影响整个文件，包括不在字符串中的\n
        # 然后，我们需要将JSON结构中的\n恢复回来

        # 步骤1: 将所有\n替换为\\n
        temp_content = content.replace('\n', '\\n')

        # 步骤2: 将JSON结构中的\\n恢复为\n
        # 查找所有在"...\\n..."之外的\\n并将其替换为\n
        # 创建一个状态机来处理
        in_string = False
        escape = False
        result = []

        i = 0
        while i < len(temp_content):
            char = temp_content[i]

            if escape:
                result.append(char)
                escape = False
                i += 1
                continue

            if char == '\\':
                result.append(char)
                escape = True
                i += 1
                continue

            if char == '"':
                result.append(char)
                in_string = not in_string
                i += 1
                continue

            if not in_string and char == '\\' and i + 1 < len(temp_content) and temp_content[i+1] == 'n':
                # 在字符串之外发现\\n，需要恢复为\n
                result.append('\n')
                i += 2  # 跳过\\n
            else:
                result.append(char)
                i += 1

        fixed_content = ''.join(result)

        # 2. 确保所有JSON数组元素都有逗号分隔符
        # 使用正则表达式查找所有在"...\n"..."之间的内容，并确保后面有逗号

        import re

        # 匹配模式: "...\n"..." (没有逗号)
        pattern = r'"([^"\\]*\\n[^"\\]*)"(?!,)\\s*(?="[^"\\]*\\n)'

        def add_comma(match):
            return match.group(0) + ','

        fixed_content = re.sub(pattern, add_comma, fixed_content, flags=re.DOTALL)

        # 3. 尝试解析JSON
        try:
            data = json.loads(fixed_content)
            print(f'  ✅ JSON解析成功')
        except json.JSONDecodeError as e:
            print(f'  ❌ JSON解析失败: {e}')

            # 尝试更激进的修复
            print(f'  🔧 尝试更激进的修复...')

            # 将所有内容视为字符串，然后重新构建JSON
            # 这个方法比较极端，但可能有效

            # 读取文件内容，将其视为Python代码，然后使用eval()解析
            # 注意：这是一个危险的操作，仅用于修复已知结构的文件

            try:
                # 替换所有\n为\\n
                temp_content2 = content.replace('\n', '\\n')

                # 使用eval()解析
                data = eval(temp_content2)
                print(f'  ✅ 使用eval()解析成功')

                # 重新转换为JSON
                fixed_content = json.dumps(data, ensure_ascii=False, indent=2)
                print(f'  ✅ 重新转换为JSON成功')

            except Exception as e2:
                print(f'  ❌ 激进修复失败: {e2}')
                return False

        # 4. 保存修复后的文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        print(f'  ✅ 文件修复成功')
        return True

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
        manual_fix_notebook(notebook_path)

    print('\n所有文件修复完成！')

if __name__ == '__main__':
    main()