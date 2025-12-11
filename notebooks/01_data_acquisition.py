#!/usr/bin/env python
# coding: utf-8

# # 01_data_acquisition
# 
# 本笔记本用于执行电影数据的获取操作，包括从Kaggle和pandas GitHub下载电影数据集。

# In[ ]:


# 导入必要的库
import os
import sys
import pandas as pd
import subprocess


# ## 1. 检查项目结构

# In[ ]:


# 检查当前工作目录
print(f'当前工作目录: {os.getcwd()}')

# 检查项目结构
project_dir = '..'
print('项目结构:')
for root, dirs, files in os.walk(project_dir):
    level = root.replace(project_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # 只显示前5个文件
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... 等 {len(files) - 5} 个文件')


# ## 2. 执行数据获取脚本

# In[ ]:


# 执行数据获取脚本
src_dir = os.path.join(project_dir, 'src')
os.chdir(src_dir)

print('正在执行数据获取脚本...')
result = subprocess.run(['python', 'data_acquisition.py', '--skip-download'], capture_output=True, text=True)

# 打印脚本输出
print('脚本输出:')
print('-' * 50)
print(result.stdout)

if result.stderr:
    print('错误信息:')
    print('-' * 50)
    print(result.stderr)

print('返回代码:', result.returncode)


# ## 3. 查看获取的数据

# In[ ]:


# 查看raw目录下的文件
raw_dir = os.path.join(project_dir, 'data', 'raw')
print('raw目录下的文件:')
for file in os.listdir(raw_dir):
    print(f'  - {file}')

# 查看processed目录下的文件
processed_dir = os.path.join(project_dir, 'data', 'processed')
print('processed目录下的文件:')
for file in os.listdir(processed_dir):
    print(f'  - {file}')


# ## 4. 加载并查看数据示例

# In[ ]:


# 加载并查看TMDB电影数据
tmdb_movies_path = os.path.join(raw_dir, 'tmdb_5000_movies.csv')
if os.path.exists(tmdb_movies_path):
    tmdb_movies = pd.read_csv(tmdb_movies_path)
    print(f'TMDB电影数据形状: {tmdb_movies.shape}')
    print('
TMDB电影数据示例:')
    display(tmdb_movies.head())
else:
    print('TMDB电影数据文件不存在')

# 加载并查看TMDB演职员数据
tmdb_credits_path = os.path.join(raw_dir, 'tmdb_5000_credits.csv')
if os.path.exists(tmdb_credits_path):
    tmdb_credits = pd.read_csv(tmdb_credits_path)
    print(f'
TMDB演职员数据形状: {tmdb_credits.shape}')
    print('
TMDB演职员数据示例:')
    display(tmdb_credits.head())
else:
    print('TMDB演职员数据文件不存在')

# 加载并查看pandas电影数据
pandas_movies_path = os.path.join(raw_dir, 'movies.csv')
if os.path.exists(pandas_movies_path):
    pandas_movies = pd.read_csv(pandas_movies_path)
    print(f'
pandas电影数据形状: {pandas_movies.shape}')
    print('
pandas电影数据示例:')
    display(pandas_movies.head())
else:
    print('pandas电影数据文件不存在')


# ## 5. 结论

# In[ ]:


# 总结数据获取结果
print('数据获取操作完成！')
print('
下一步操作:')
print('1. 运行02_data_preprocessing.ipynb进行数据预处理')
print('2. 或直接执行src/data_preprocessing.py脚本')

