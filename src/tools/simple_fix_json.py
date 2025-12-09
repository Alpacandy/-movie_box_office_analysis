import json
import os
import re
import nbformat

def fix_notebook_json(file_path):
    """
    修复notebook文件的JSON格式问题
    """
    print(f'修复文件: {os.path.basename(file_path)}')

    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复字符串中的未转义换行符
        # 这个正则表达式会匹配字符串中的换行符并转义它们
        def fix_string(match):
            string_content = match.group(1)
            # 转义换行符
            string_content = string_content.replace('\n', '\\n')
            # 转义双引号
            string_content = string_content.replace('"', '\\"')
            return f'"{string_content}"'

        # 使用正则表达式修复所有字符串
        fixed_content = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', fix_string, content)

        # 尝试解析JSON
        data = json.loads(fixed_content)

        # 使用nbformat写入正确的JSON
        nbformat.write(nbformat.from_dict(data), file_path)

        print(f'  ✅ 修复成功')
        return True

    except Exception as e:
        print(f'  ❌ 修复失败: {e}')
        return False

def main():
    # 修复指定的notebook文件
    file_path = 'movie_box_office_analysis/notebooks/06_deep_learning.ipynb'

    if os.path.exists(file_path):
        fix_notebook_json(file_path)
    else:
        print(f'文件不存在: {file_path}')

if __name__ == '__main__':
    main()