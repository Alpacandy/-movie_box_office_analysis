import os
import nbformat

# 修复notebook文件
def fix_notebook(file_path):
    print(f'修复文件: {os.path.basename(file_path)}')

    try:
        # 直接使用nbformat读取并写入文件
        # 这通常能自动修复JSON格式问题
        nb = nbformat.read(file_path, as_version=4)
        nbformat.write(nb, file_path)

        print(f'  ✅ 修复成功')
        return True

    except Exception as e:
        print(f'  ❌ 修复失败: {e}')
        return False

# 主程序
def main():
    # 修复指定的notebook文件
    file_path = 'movie_box_office_analysis/notebooks/06_deep_learning.ipynb'

    if os.path.exists(file_path):
        fix_notebook(file_path)
    else:
        print(f'文件不存在: {file_path}')

if __name__ == '__main__':
    main()