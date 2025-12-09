import os
import shutil

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
# 设置results文件夹路径为项目根目录下的results目录
results_path = os.path.join(project_root, "results")

# 确保文件夹存在
if os.path.exists(results_path):
    # 遍历results文件夹中的所有内容
    for item in os.listdir(results_path):
        item_path = os.path.join(results_path, item)
        try:
            # 如果是文件，直接删除
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"已删除文件: {item_path}")
            # 如果是文件夹，递归删除
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"已删除文件夹: {item_path}")
        except Exception as e:
            print(f"删除失败: {item_path}, 错误: {e}")
    print("已清空results文件夹中的所有内容。")
else:
    print(f"文件夹不存在: {results_path}")