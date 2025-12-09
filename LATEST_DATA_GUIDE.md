# 获取2017年至今最新电影数据指南

由于TMDB API连接超时，无法通过代码自动获取2017年至今的最新电影数据，我们提供以下手动获取方案：

## 方案一：下载Full TMDB Movies Dataset 2023（推荐）

Kaggle上有一个更新的Full TMDB Movies Dataset 2023，包含93万部电影，数据更新至2023年：

**下载地址**：https://www.kaggle.com/datasets/akshaypawar7/millions-of-movies-full-tmdb-dataset-2023

### 操作步骤：

1. 访问上述Kaggle链接并下载数据集（需要Kaggle账号）
2. 解压下载的文件，将`tmdb_movies.csv`文件复制到项目的`data/raw`目录
3. 重命名为`tmdb_latest_movies_2017_2023.csv`

## 方案二：下载TMDB 5000 Movie Dataset（基础版）

如果无法访问Full TMDB数据集，可以下载原始的TMDB 5000 Movie Dataset：

**下载地址**：https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

### 操作步骤：

1. 访问上述Kaggle链接并下载数据集
2. 解压下载的文件，将以下两个文件复制到项目的`data/raw`目录：
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv`

## 方案三：使用MovieLens数据集（替代方案）

MovieLens 1M Dataset包含100万条用户评分数据，但电影信息可能较旧：

### 自动下载MovieLens数据集：

```bash
python src/data_acquisition.py --skip-download
```

## 后续处理步骤

1. 数据下载完成后，运行数据预处理脚本：

```bash
python src/data_preprocessing.py
```

2. 运行特征工程脚本：

```bash
python src/feature_engineering.py
```

3. 运行模型训练脚本：

```bash
python src/modeling.py
```

4. 运行可视化脚本：

```bash
python src/visualization.py
```

## 注意事项

1. 如果使用Full TMDB Movies Dataset 2023，可能需要修改`data_acquisition.py`中的数据加载逻辑
2. 如果需要更详细的票房数据，可以尝试结合Box Office Mojo数据：

```bash
python src/data_acquisition.py --use-box-office-mojo
```

3. 如果需要IMDb评分数据，可以使用OMDb API：

```bash
python src/data_acquisition.py --use-imdb --imdb-api-key=YOUR_API_KEY
```

4. 所有脚本都支持`--help`参数查看详细使用说明：

```bash
python src/data_acquisition.py --help
```
