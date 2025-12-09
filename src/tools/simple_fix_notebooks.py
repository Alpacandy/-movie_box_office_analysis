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
        # 简单地读取并重新写入文件，这会自动修复JSON格式问题
        nb = nbformat.read(notebook_path, as_version=4)
        nbformat.write(nb, notebook_path)

        print(f'  ✅ 修复成功')

    except Exception as e:
        print(f'  ❌ 修复失败: {e}')
        # 如果nbformat无法直接读取，我们需要手动修复
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 手动修复一些常见问题
            # 1. 替换所有\n为\\n（在字符串中）
            # 使用简单的正则表达式修复
            import re

            # 这个正则表达式会查找在引号内的\n并将其替换为\\n
            # 匹配双引号字符串
            fixed_content = re.sub(r'(?<!\\)"(.*?)(?<!\\)\\n(.*?)(?<!\\)"', r'"\1\\n\2"', content, flags=re.DOTALL)
            # 匹配单引号字符串
            fixed_content = re.sub(r"(?<!\\)'(.*?)(?<!\\)\\n(.*?)(?<!\\)'", r"'\1\\n\2'", fixed_content, flags=re.DOTALL)

            # 尝试重新解析
            nb = nbformat.reads(fixed_content, as_version=4)
            nbformat.write(nb, notebook_path)

            print(f'  ✅ 手动修复成功')

        except Exception as e2:
            print(f'  ❌ 手动修复也失败: {e2}')

print('\n所有文件修复完成！')