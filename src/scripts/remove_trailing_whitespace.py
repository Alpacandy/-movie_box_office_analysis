#!/usr/bin/env python3
"""
移除文件中的行尾空格
"""
import os

def remove_trailing_whitespace(file_path):
    """移除文件中的行尾空格"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 移除每行末尾的空格
        cleaned_lines = [line.rstrip() + '\n' for line in lines]

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        print(f"已成功移除 {file_path} 中的行尾空格")
        return True
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")
        return False

if __name__ == "__main__":
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 计算项目根目录
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    # 构建要处理的文件的绝对路径
    file_path = os.path.join(project_root, "src/data_preprocessing/__init__.py")
    remove_trailing_whitespace(file_path)
