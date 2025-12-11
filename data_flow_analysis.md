# 电影票房分析项目 - 数据获取与处理流程分析

## 项目概述
本项目用于电影票房数据的获取、处理和分析，采用模块化设计，将各个功能拆分为独立的模块，便于管理和维护。

## 数据获取模块

### 核心文件：`src/data_acquisition.py`

#### 主要功能
- 从Kaggle下载TMDB 5000 Movie Dataset
- 从MovieLens下载电影数据集
- 从TMDB API获取最新电影数据
- 从OMDb API获取IMDb数据
- 从Box Office Mojo爬取年度票房数据
- 合并多个数据源的数据

#### 关键类与方法

1. **DataAcquisition类**（第44行定义）
   - 项目数据获取的核心类，整合了所有数据获取功能

2. **数据下载方法**
   - `download_tmdb_dataset()`（第85行）：从Kaggle下载TMDB 5000 Movie Dataset
   - `download_pandas_movie_dataset()`（第137行）：从MovieLens下载数据集

3. **数据加载方法**
   - `load_tmdb_data()`（第170行）：加载TMDB数据集
   - `load_pandas_movie_data()`（第204行）：加载pandas电影数据集

4. **API数据获取**
   - `get_tmdb_data()`（第231行）：从TMDB API获取电影数据
   - `get_omdb_data()`（第325行）：从OMDb API获取IMDb数据

5. **网络爬虫**
   - `scrape_box_office_mojo()`（第367行）：从Box Office Mojo爬取年度票房数据

6. **数据合并与保存**
   - `merge_tmdb_data()`（第403行）：合并TMDB数据集
   - `merge_multiple_datasets()`（第432行）：合并多个数据源
   - `enrich_data_with_box_office_mojo()`（第454行）：使用Box Office Mojo数据丰富现有数据集
   - `save_data()`（第519行）：保存处理后的数据

7. **主流程**
   - `run()`（第541行）：执行完整的数据获取流程
   - `main()`（第588行）：命令行接口，处理参数并执行数据获取

## 数据处理模块

### 核心目录：`src/data_preprocessing/`

该目录包含多个子模块，负责数据加载、清洗、缺失值处理、特征提取和异常值检测等功能。

#### 1. `data_loader.py` - 数据加载模块

- **DataLoader类**（第19行定义）
  - `load_data()`（第43行）：加载数据文件
  - `optimize_dtypes()`（第81行）：优化数据类型以减少内存使用
  - `save_checkpoint()`（第129行）：保存检查点
  - `load_checkpoint()`（第153行）：加载检查点

#### 2. `data_cleaner.py` - 数据清洗模块

- **DataCleaner类**（第20行定义）
  - `clean_data()`（第43行）：执行完整的数据清洗流程
  - `filter_data_by_date()`（第71行）：根据日期过滤数据
  - `filter_data_by_budget()`（第101行）：过滤预算异常的数据
  - `filter_data_by_revenue()`（第134行）：过滤票房异常的数据

#### 3. `missing_value_handler.py` - 缺失值处理模块

- **MissingValueHandler类**（第18行定义）
  - `handle_missing_values()`（第41行）：处理缺失值的主方法
  - `detect_outliers()`（第122行）：检测异常值
  - `remove_outliers()`（第148行）：移除异常值

#### 4. `feature_extractor.py` - 特征提取模块

（未完全查看，但从目录结构和项目设计推断其功能）

#### 5. `outlier_detector.py` - 异常值检测模块

- **OutlierDetector类**（第21行定义）
  - `detect_outliers()`（第44行）：检测异常值
  - `remove_outliers()`（第101行）：移除异常值
  - `cap_outliers()`（第131行）：对异常值进行盖帽处理
  - `winsorize_outliers()`（第152行）：对异常值进行Winsorize处理
  - `get_outlier_statistics()`（第174行）：获取异常值统计信息
  - `visualize_outliers()`（第195行）：可视化异常值

#### 6. `__init__.py` - 模块入口

（未完全查看，但从项目设计推断其会导出DataPreprocessing类，整合所有预处理功能）

## 数据处理流程

根据notebooks/02_data_preprocessing.ipynb中的示例，数据处理的典型流程如下：

1. **数据加载**：`preprocessor.load_data()`
2. **数据清洗**：`preprocessor.clean_data(data)`
3. **缺失值处理**：`preprocessor.handle_missing_values(cleaned_data)`
4. **特征提取**：`preprocessor.extract_features(processed_data)`
5. **数据保存**：`preprocessor.save_processed_data(processed_data)`

## 数据流转图

```
数据获取模块 (data_acquisition.py) → 原始数据 (data/raw/) → 数据处理模块 (data_preprocessing/) → 处理后数据 (data/processed/)
```

## 关键配置

### API密钥配置
- TMDB API密钥：第22行配置
- OMDb API密钥：第23行配置

### 数据源配置
- 支持的数据源列表：第25-29行配置

### 数据路径配置
- 原始数据保存路径：第31行配置
- 处理后数据保存路径：第32行配置

## 运行方式

### 数据获取
```python
from src.data_acquisition import DataAcquisition

data_acquirer = DataAcquisition()
data_acquirer.run()
```

### 数据处理
```python
from src.data_preprocessing import DataPreprocessing

preprocessor = DataPreprocessing()
data = preprocessor.load_data()
cleaned_data = preprocessor.clean_data(data)
processed_data = preprocessor.handle_missing_values(cleaned_data)
processed_data = preprocessor.extract_features(processed_data)
preprocessor.save_processed_data(processed_data)
```

## 总结

本项目的数据获取和处理模块设计清晰，功能完整，采用模块化设计便于扩展和维护。数据获取模块负责从多个数据源获取数据并进行合并，数据处理模块则负责对原始数据进行清洗、缺失值处理、特征提取等操作，为后续的分析和建模提供高质量的数据。