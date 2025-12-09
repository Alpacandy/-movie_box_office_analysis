import os
import datetime

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录
project_root = os.path.abspath(os.path.join(script_dir, '../..'))

# 删除13:50之前生成的所有文件
target_time = datetime.datetime(2025, 12, 7, 13, 50, 0)
# 设置results文件夹路径为项目根目录下的results目录
results_path = os.path.join(project_root, "results")

# 搜索所有13:50之前的文件
old_files = []
for root, dirs, files in os.walk(results_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        if file_time < target_time:
            old_files.append(file_path)

print(f"找到 {len(old_files)} 个需要删除的文件：")
for file_path in old_files:
    print(f"- {file_path}")

# 删除这些文件
if old_files:
    for file_path in old_files:
        try:
            os.remove(file_path)
            print(f"已删除: {file_path}")
        except Exception as e:
            print(f"删除失败: {file_path}, 错误: {e}")
    print(f"已删除所有13:50之前生成的文件。")
else:
    print("没有找到需要删除的文件。")