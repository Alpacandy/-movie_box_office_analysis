#!/usr/bin/env python3
"""
清理代码文件中空白行包含的空格
"""
import os

def clean_whitespace(file_path):
    """
    清理单个文件中空白行包含的空格
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 清理空白行中的空格
        cleaned_lines = []
        for line in lines:
            if line.strip() == '':
                # 如果是空白行，替换为真正的空行
                cleaned_lines.append('\n')
            else:
                cleaned_lines.append(line)

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        print(f"清理完成: {file_path}")
        return True
    except Exception as e:
        print(f"清理失败 {file_path}: {e}")
        return False

def main():
    """
    主函数，清理指定目录下的Python文件
    """
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 计算项目根目录
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))

    # 需要清理的目录和文件
    target_dirs = [os.path.join(project_root, "src"), project_root]

    # 遍历目录下的所有Python文件
    for target_dir in target_dirs:
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    # 跳过隐藏目录中的文件
                    if "__pycache__" not in file_path and ".git" not in file_path:
                        clean_whitespace(file_path)

if __name__ == "__main__":
    main()