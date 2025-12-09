import pandas as pd
import os

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
# 构建数据文件路径
data_path = os.path.join(project_root, 'data/processed/cleaned_movie_data.csv')

# 加载处理后的数据
data = pd.read_csv(data_path)

# 打印所有列
print('所有可用列:')
print(data.columns.tolist())
print()

# 找出文本列
text_cols = [col for col in data.columns if data[col].dtype == 'object']
print('文本列:')
print(text_cols)
print()

# 显示每列的示例
print('文本列示例:')
for col in text_cols:
    print(f'\n{col}:')
    print(data[col].head(3))
    print('-' * 50)
