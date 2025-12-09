import os
import sys
import pandas as pd
import numpy as np
import ast
import time
import pickle
import statistics
import argparse
import yaml
from datetime import datetime
from tqdm import tqdm
import traceback

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入本地模块和其他库
from src.utils.logging_config import get_logger
from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .feature_extractor import FeatureExtractor

# 尝试导入Dask，用于大规模数据处理
try:
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# 获取当前脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))


class DataPreprocessing:
    def __init__(self, base_dir=None, config_file=None):
        # 使用项目根目录作为基础路径
        self.base_dir = os.path.join(project_root, "data") if base_dir is None else base_dir
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.performance_data = []  # 存储性能数据

        # 先配置日志，再加载配置
        self.setup_logging()

        # 加载配置
        self.config = self.load_config(config_file)

        os.makedirs(self.processed_dir, exist_ok=True)

        # 实例化独立的功能模块
        self.data_loader = DataLoader(base_dir=self.base_dir, config=self.config)
        self.data_cleaner = DataCleaner(config=self.config)
        self.feature_extractor = FeatureExtractor(config=self.config)

        self.logger.info(f"数据预处理对象初始化完成，基础目录: {base_dir}")
        self.logger.info("已初始化模块: DataLoader, DataCleaner, FeatureExtractor")

    def setup_logging(self):
        """配置日志系统"""
        # 使用统一的日志配置
        self.logger = get_logger('DataPreprocessing')

    def load_config(self, config_file=None):
        """加载配置文件

        Args:
            config_file: 配置文件路径

        Returns:
            dict: 配置字典
        """
        # 默认配置
        default_config = {
            'data_loading': {
                'filename': "tmdb_merged.csv",
                'use_chunks': False,
                'chunksize': 200000,
                'optimize_memory': True,
                'use_dask': True
            },
            'data_cleaning': {
                'remove_duplicates': True,
                'min_budget': 100,
                'min_revenue': 100,
                'max_future_years': 0
            },
            'missing_values': {
                'threshold_high_missing': 0.7,
                'threshold_medium_missing': 0.3,
                'use_group_imputation': True
            },
            'feature_extraction': {
                'extract_json_features': True,
                'extract_datetime_features': True,
                'calculate_financial_metrics': True
            },
            'outliers': {
                'detection_columns': ['budget', 'revenue', 'runtime', 'return_on_investment'],
                'treatment': 'keep'
            },
            'performance_monitoring': {
                'enabled': True,
                'log_to_file': False,
                'log_level': 'INFO'
            },
            'checkpoints': {
                'enabled': True,
                'checkpoint_dir': "checkpoints"
            },
            'validation': {
                'enabled': True,
                'deep_check': False,
                'report_path': "validation_report.txt"
            }
        }

        # 如果提供了配置文件路径，则加载配置
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)

                # 更新默认配置
                self._update_config(default_config, user_config)
                self.logger.info(f"成功加载配置文件: {config_file}")
            except Exception as e:
                self.logger.error(f"加载配置文件失败: {e}")
                self.logger.info("使用默认配置")

        return default_config

    def _update_config(self, default, user):
        """递归更新配置字典

        Args:
            default: 默认配置字典
            user: 用户配置字典
        """
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._update_config(default[key], value)
            else:
                default[key] = value

    @staticmethod
    def performance_monitor(func):
        """性能监控装饰器

        Args:
            func: 要监控的函数

        Returns:
            包装后的函数
        """
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            start_memory = 0  # 可以添加内存监控

            try:
                result = func(self, *args, **kwargs)
                success = True
            except Exception as e:
                # 记录错误但重新抛出异常，以便调用者处理
                success = False
                error = str(e)
                self.logger.error(f"{func.__name__} - 执行失败: {error}")
                self.logger.exception("异常详细信息:")

                # 记录性能数据
                end_time = time.time()
                execution_time = end_time - start_time

                performance_record = {
                    'method_name': func.__name__,
                    'start_time': datetime.fromtimestamp(start_time),
                    'end_time': datetime.fromtimestamp(end_time),
                    'execution_time': execution_time,
                    'success': success,
                    'input_size': len(args[0]) if args and hasattr(args[0], '__len__') else None,
                    'memory_used_mb': (start_memory - start_memory) / 1024 / 1024 if start_memory else None,
                    'error': error
                }

                self.performance_data.append(performance_record)
                self.logger.info(f"{func.__name__} - 执行时间: {execution_time:.4f}秒")

                # 重新抛出异常，以便调用者处理
                raise

            end_time = time.time()
            end_memory = 0  # 可以添加内存监控
            execution_time = end_time - start_time

            # 记录性能数据
            performance_record = {
                'method_name': func.__name__,
                'start_time': datetime.fromtimestamp(start_time),
                'end_time': datetime.fromtimestamp(end_time),
                'execution_time': execution_time,
                'success': success,
                'input_size': len(args[0]) if args and hasattr(args[0], '__len__') else None,
                'memory_used_mb': (end_memory - start_memory) / 1024 / 1024 if start_memory and end_memory else None
            }

            self.performance_data.append(performance_record)
            self.logger.info(f"{func.__name__} - 执行时间: {execution_time:.4f}秒")

            return result

        return wrapper

    def get_performance_report(self):
        """生成性能报告

        Returns:
            性能报告字符串
        """
        if not self.performance_data:
            return "暂无性能数据"

        report = ["\n--- 性能监控报告 ---",
                  f"总执行方法数: {len(self.performance_data)}",
                  f"报告生成时间: {datetime.now()}",
                  ""]

        # 按方法分组统计
        method_stats = {}
        for record in self.performance_data:
            method_name = record['method_name']
            if method_name not in method_stats:
                method_stats[method_name] = {
                    'count': 0,
                    'total_time': 0,
                    'success_count': 0,
                    'times': []
                }

            method_stats[method_name]['count'] += 1
            method_stats[method_name]['total_time'] += record['execution_time']
            method_stats[method_name]['times'].append(record['execution_time'])

            if record['success']:
                method_stats[method_name]['success_count'] += 1

        # 添加详细统计
        report.append("方法详细统计:")
        report.append("-" * 60)
        report.append("{:<30} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
            "方法名", "调用次数", "成功次数", "总时间(秒)", "平均时间(秒)", "中位数时间(秒)"))
        report.append("-" * 60)

        for method_name, stats in sorted(method_stats.items()):
            avg_time = stats['total_time'] / stats['count']
            median_time = statistics.median(stats['times'])
            report.append("{:<30} {:>10} {:>10} {:>12.4f} {:>12.4f} {:>12.4f}".format(
                method_name, stats['count'], stats['success_count'],
                stats['total_time'], avg_time, median_time))

        report.append("-" * 60)

        # 总体统计
        total_time = sum(record['execution_time'] for record in self.performance_data)
        successful_records = [r for r in self.performance_data if r['success']]
        success_rate = len(successful_records) / len(self.performance_data) * 100

        report.append(f"总执行时间: {total_time:.4f}秒")
        report.append(f"成功率: {success_rate:.2f}%")

        if len(self.performance_data) > 1:
            # 计算瓶颈
            sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
            top_3 = sorted_methods[:3]
            report.append("\n性能瓶颈(耗时最长的3个方法):")
            for i, (method_name, stats) in enumerate(top_3, 1):
                percentage = (stats['total_time'] / total_time) * 100
                report.append(f"{i}. {method_name}: {stats['total_time']:.4f}秒 ({percentage:.2f}%)")

        return "\n".join(report)

    def clear_performance_data(self):
        """清除性能数据"""
        self.performance_data = []

    @performance_monitor
    def optimize_dtypes(self, data):
        """优化数据类型以减少内存使用

        Args:
            data: pandas DataFrame

        Returns:
            优化后的数据框
        """
        # 优化数值类型
        for col in data.select_dtypes(include=['int64', 'float64']).columns:
            # 对于整数类型
            if data[col].dtype == 'int64':
                # 检查最小值和最大值，选择合适的整数类型
                min_val = data[col].min()
                max_val = data[col].max()

                if min_val >= -128 and max_val <= 127:
                    data[col] = data[col].astype('int8')
                elif min_val >= -32768 and max_val <= 32767:
                    data[col] = data[col].astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    data[col] = data[col].astype('int32')
            # 对于浮点类型
            elif data[col].dtype == 'float64':
                # 大多数电影数据不需要高精度，使用float32
                data[col] = data[col].astype('float32')

        # 优化分类类型
        for col in data.select_dtypes(include=['object']).columns:
            # 检查唯一值比例，如果低于50%，使用category类型
            if len(data[col].unique()) / len(data[col]) < 0.5:
                data[col] = data[col].astype('category')

        return data

    def retry_operation(self, func, max_retries=3, delay=1, *args, **kwargs):
        """重试操作装饰器

        Args:
            func: 要执行的函数
            max_retries: 最大重试次数
            delay: 重试间隔时间(秒)

        Returns:
            函数执行结果
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"操作失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"{delay}秒后重试...")
                    time.sleep(delay)
                else:
                    self.logger.error("已达到最大重试次数")
                    raise

    def load_checkpoint(self, checkpoint_name):
        """加载处理进度检查点

        Args:
            checkpoint_name: 检查点名称

        Returns:
            检查点数据或None
        """
        checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_name}.pkl")

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.logger.info(f"已加载检查点: {checkpoint_file}")
                return checkpoint
            except Exception as e:
                self.logger.error(f"加载检查点失败: {e}")
        return None

    def save_checkpoint(self, checkpoint_name, data):
        """保存处理进度检查点

        Args:
            checkpoint_name: 检查点名称
            data: 要保存的检查点数据
        """
        checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_name}.pkl")

        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"已保存检查点: {checkpoint_file}")
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")

    def clear_checkpoint(self, checkpoint_name):
        """清除处理进度检查点

        Args:
            checkpoint_name: 检查点名称
        """
        checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_name}.pkl")

        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                self.logger.info(f"已清除检查点: {checkpoint_file}")
            except Exception as e:
                self.logger.error(f"清除检查点失败: {e}")

    @performance_monitor
    def load_data(self, filename=None, use_chunks=None, chunksize=None, optimize_memory=None, use_dask=None):
        """加载原始数据，使用DataLoader模块

        Args:
            filename: 数据文件名
            use_chunks: 是否使用分块加载
            chunksize: 分块大小
            optimize_memory: 是否优化内存使用
            use_dask: 是否使用Dask进行并行处理

        Returns:
            如果use_dask为True，返回Dask DataFrame；如果use_chunks为True，返回DataFrame迭代器；否则返回完整DataFrame
        """
        # 使用配置中的默认值，如果没有提供参数
        config = self.config.get('data_loading', {})
        filename = filename or config.get('filename', 'tmdb_merged.csv')
        use_chunks = use_chunks if use_chunks is not None else config.get('use_chunks', False)
        chunksize = chunksize or config.get('chunksize', 200000)
        optimize_memory = optimize_memory if optimize_memory is not None else config.get('optimize_memory', True)
        use_dask = use_dask if use_dask is not None else config.get('use_dask', True)

        self.logger.info(f"调用DataLoader加载数据: {filename}")
        return self.data_loader.load_data(filename=filename, use_chunks=use_chunks, chunksize=chunksize,
                                         optimize_memory=optimize_memory, use_dask=use_dask)

    @performance_monitor
    def clean_data(self, data):
        """数据清洗，使用DataCleaner模块

        Args:
            data: 输入数据（Pandas或Dask DataFrame）

        Returns:
            清洗后的数据
        """
        self.logger.info("调用DataCleaner进行数据清洗...")
        return self.data_cleaner.clean_data(data)

    @performance_monitor
    def extract_features(self, data):
        """特征提取，使用FeatureExtractor模块

        Args:
            data: 输入数据（Pandas或Dask DataFrame）

        Returns:
            包含提取特征的数据
        """
        self.logger.info("调用FeatureExtractor进行特征提取...")
        return self.feature_extractor.extract_features(data)

    @performance_monitor
    def handle_missing_values(self, data):
        """处理缺失值并进行数据质量检查"""
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在处理缺失值和数据质量检查...")

        # 获取数据大小信息
        if is_dask:
            # Dask需要计算才能获取准确值
            total_rows = len(data)
            columns_count = len(data.columns)
        else:
            total_rows = len(data)
            columns_count = data.shape[1]

        # 显示缺失值统计
        missing_values = data.isnull().sum()

        # 处理Dask的延迟计算
        if is_dask:
            missing_values = missing_values.compute()

        missing_values = missing_values[missing_values > 0]

        if len(missing_values) > 0:
            self.logger.info(f"缺失值列: {missing_values.to_dict()}")

            # 计算缺失率
            missing_rates = (missing_values / total_rows) * 100
            missing_rates = missing_rates.round(2)
            self.logger.info(f"缺失率: {missing_rates.to_dict()}")
        else:
            self.logger.info("未发现缺失值")
            return data

        # 1. 删除缺失值过多的列（缺失率超过70%）
        threshold = total_rows * 0.7
        columns_before = columns_count

        # 必须保留的核心列，包括财务相关字段
        essential_columns = ['budget', 'revenue']

        if is_dask:
            # Dask不支持axis参数，手动计算每列缺失值并筛选
            # 计算每列非空值数量
            non_null_counts = data.count().compute()
            # 筛选出非空值数量大于等于阈值的列
            keep_columns = non_null_counts[non_null_counts >= threshold].index.tolist()
            # 确保必须保留的列也被包含
            for col in essential_columns:
                if col in data.columns and col not in keep_columns:
                    keep_columns.append(col)
            # 重新选择列
            data = data[keep_columns]
        else:
            # 保存必须保留的列
            essential_data = data[essential_columns] if all(col in data.columns for col in essential_columns) else None

            # Pandas使用dropna方法
            data = data.dropna(axis=1, thresh=threshold)

            # 将必须保留的列添加回来
            if essential_data is not None:
                for col in essential_columns:
                    if col not in data.columns:
                        data[col] = essential_data[col]

        columns_after = len(data.columns)
        if columns_after < columns_before:
            self.logger.info(f"删除高缺失率列: {columns_before - columns_after} 列")

        # 2. 智能缺失值填充策略
        # 分离数值型和非数值型特征
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns

        # 2.1 数值型特征处理
        for col in numeric_cols:
            # 计算缺失率
            if is_dask:
                missing_count = data[col].isnull().sum().compute()
            else:
                missing_count = data[col].isnull().sum()

            if missing_count > 0:
                missing_rate = (missing_count / total_rows) * 100

                if missing_rate < 10:
                    # 低缺失率：使用中位数填充
                    median_val = data[col].median()
                    if is_dask:
                        median_val = median_val.compute()
                    data.loc[:, col] = data[col].fillna(median_val)
                    self.logger.info(f"  {col}: 缺失率 {missing_rate:.1f}%，使用中位数 {median_val:.2f} 填充")
                elif missing_rate < 30:
                    # 中等缺失率：使用分组填充
                    # Dask不支持复杂的分组变换，使用中位数填充
                    if is_dask:
                        median_val = data[col].median().compute()
                        data.loc[:, col] = data[col].fillna(median_val)
                        self.logger.info(f"  {col}: 缺失率 {missing_rate:.1f}%，Dask模式下使用中位数填充")
                    else:
                        if 'release_year' in data.columns:
                            # 按年份分组填充
                            data.loc[:, col] = data.groupby('release_year')[col].transform(
                                lambda x: x.fillna(x.median())
                            )
                        elif 'genres_list' in data.columns:
                            # 按类型分组填充
                            data.loc[:, col] = data.groupby('genres_list')[col].transform(
                                lambda x: x.fillna(x.median())
                            )
                        else:
                            # 回退到中位数填充
                            median_val = data[col].median()
                            data.loc[:, col] = data[col].fillna(median_val)
                        self.logger.info(f"  {col}: 缺失率 {missing_rate:.1f}%，使用分组策略填充")
                else:
                    # 高缺失率：填充为-1并添加缺失标记
                    data.loc[:, f'{col}_missing'] = data[col].isnull().astype(int)
                    data.loc[:, col] = data[col].fillna(-1)
                    self.logger.info(f"  {col}: 缺失率 {missing_rate:.1f}%，填充为-1并添加缺失标记")

        # 2.2 分类特征处理
        for col in categorical_cols:
            # 计算缺失率
            if is_dask:
                missing_count = data[col].isnull().sum().compute()
            else:
                missing_count = data[col].isnull().sum()

            if missing_count > 0:
                missing_rate = (missing_count / total_rows) * 100

                if missing_rate < 20:
                    # 使用众数填充
                    if is_dask:
                        # Dask模式下众数计算较复杂，使用'Unknown'填充
                        data.loc[:, col] = data[col].fillna('Unknown')
                        self.logger.info(f"  {col}: 缺失率 {missing_rate:.1f}%，Dask模式下使用'Unknown'填充")
                    else:
                        mode_val = data[col].mode().iloc[0]
                        data.loc[:, col] = data[col].fillna(mode_val)
                        self.logger.info(f"  {col}: 缺失率 {missing_rate:.1f}%，使用众数 '{mode_val}' 填充")
                else:
                    # 使用Unknown填充
                    data.loc[:, col] = data[col].fillna('Unknown')
                    self.logger.info(f"  {col}: 缺失率 {missing_rate:.1f}%，使用'Unknown'填充")

        # 2.3 日期时间特征处理
        for col in datetime_cols:
            # 计算缺失率
            if is_dask:
                missing_count = data[col].isnull().sum().compute()
            else:
                missing_count = data[col].isnull().sum()

            if missing_count > 0:
                # 日期时间特征较少，直接删除或标记
                data = data.dropna(subset=[col])
                self.logger.info(f"  {col}: 删除日期时间缺失的记录")

        # 3. 特殊特征处理
        # 处理runtime为0或异常小的值
        if 'runtime' in data.columns:
            if is_dask:
                # Dask模式下使用不同的处理方式
                median_runtime = data[((data['runtime'] > 0) &
                                      (data['runtime'] >= 10))]['runtime'].median().compute()
                data = data.assign(
                    runtime=data['runtime'].where(
                        ~((data['runtime'] == 0) | (data['runtime'] < 10)),
                        median_runtime
                    )
                )
                self.logger.info(f"  runtime: Dask模式下使用中位数 {median_runtime:.1f} 填充无效值")
            else:
                zero_runtime = len(data[(data['runtime'] == 0) | (data['runtime'] < 10)])
                if zero_runtime > 0:
                    median_runtime = data[(data['runtime'] > 0) & (data['runtime'] >= 10)]['runtime'].median()
                    data.loc[(data['runtime'] == 0) | (data['runtime'] < 10), 'runtime'] = median_runtime
                    self.logger.info(f"  runtime: {zero_runtime} 个无效值使用中位数 {median_runtime:.1f} 填充")

        # 处理budget和revenue为0的情况
        financial_cols = ['budget', 'revenue']
        for col in financial_cols:
            if col in data.columns:
                if is_dask:
                    # Dask模式下添加标记
                    data[f'{col}_zero'] = (data[col] == 0).astype(int)
                    self.logger.info(f"  {col}: Dask模式下添加0值标记列")
                else:
                    zero_count = len(data[data[col] == 0])
                    if zero_count > 0:
                        # 对于财务数据，0可能是有效缺失值，添加标记
                        data[f'{col}_zero'] = (data[col] == 0).astype(int)
                        self.logger.info(f"  {col}: 发现 {zero_count} 个0值，添加标记列")

        # 4. 数据质量检查报告
        self.logger.info("\n--- 数据质量检查报告 ---")

        # 获取处理后的数据大小
        if is_dask:
            self.logger.info(f"处理后记录数: {len(data)}")
            self.logger.info(f"处理后特征数: {len(data.columns)}")
        else:
            self.logger.info(f"处理后记录数: {len(data)}")
            self.logger.info(f"处理后特征数: {data.shape[1]}")

        # 检查重复值 - Dask不支持直接的duplicated().sum()
        if not is_dask:
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                self.logger.warning(f"仍存在重复记录: {duplicate_count} 条")
        else:
            self.logger.info("Dask模式下跳过重复值检查")

        # 检查数据类型一致性
        for col in data.columns:
            if data[col].dtype == 'object':
                if is_dask:
                    # Dask不支持nunique()直接计算，跳过此检查
                    continue
                else:
                    unique_count = data[col].nunique()
                    if unique_count == 1:
                        self.logger.warning(f"{col} 列所有值相同")

        # 最终缺失值检查
        final_missing = data.isnull().sum()
        if is_dask:
            final_missing = final_missing.compute()
            total_final_missing = final_missing.sum()
        else:
            total_final_missing = final_missing.sum()

        self.logger.info(f"最终缺失值总数: {total_final_missing}")

        if total_final_missing > 0:
            if is_dask:
                # Dask模式下使用更简单的填充
                data = data.fillna(-1)
                self.logger.info("Dask模式下使用-1填充剩余缺失值")
            else:
                final_missing_cols = data.columns[data.isnull().any()]
                self.logger.info(f"仍存在缺失值的列: {list(final_missing_cols)}")
                # 最后尝试使用更保守的填充
                data = data.fillna(method='ffill').fillna(method='bfill')
                self.logger.info("使用前向/后向填充处理剩余缺失值")

        self.logger.info("缺失值处理和数据质量检查完成")
        return data

    @performance_monitor
    def convert_datetime(self, data):
        """转换日期时间格式"""
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在处理日期时间...")

        if is_dask:
            # Dask支持pd.to_datetime，但需要使用dask的方式
            from dask.dataframe import to_datetime
            data['release_date'] = to_datetime(data['release_date'], errors='coerce')
        else:
            # 转换release_date为datetime格式
            data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

        # 提取时间特征
        data['release_year'] = data['release_date'].dt.year
        data['release_month'] = data['release_date'].dt.month
        data['release_day'] = data['release_date'].dt.day
        data['release_quarter'] = data['release_date'].dt.quarter
        data['release_dayofweek'] = data['release_date'].dt.dayofweek  # 0=周一, 6=周日

        # 计算是否为周末 - 适配Dask
        if is_dask:
            data['is_weekend'] = (data['release_dayofweek'] >= 5).astype(int)
        else:
            data['is_weekend'] = data['release_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

        # 计算电影年龄
        current_year = datetime.now().year
        data['movie_age'] = current_year - data['release_year']

        return data

    @performance_monitor
    def extract_json_features(self, data, filename=None):
        """从JSON字段中提取特征
        
        Args:
            data: 输入数据
            filename: 文件名，用于确定数据格式
            
        Returns:
            提取特征后的数据
        """
        self.logger.info("正在提取特征...")
        
        # 确保关键列是字符串类型，避免Categorical类型导致的问题
        json_columns = ['genres', 'production_companies', 'production_countries', 
                      'spoken_languages', 'keywords', 'crew', 'cast']
        for col in json_columns:
            if col in data.columns:
                # 将Categorical类型转换为字符串
                if hasattr(data[col].dtype, 'categories'):
                    data[col] = data[col].astype(str)
                # 确保非Categorical类型也是字符串
                elif data[col].dtype != 'object':
                    data[col] = data[col].astype(str)

        # 判断数据格式
        is_new_format = (filename == "TMDB_movie_dataset_v11.csv" or
                        ('genres' in data.columns and
                         data['genres'].str.startswith('Action', na=False).any()))

        if is_new_format:
            self.logger.info("检测到新数据格式，使用逗号分隔格式处理...")

            # 1. 提取电影类型（逗号分隔） - 优化版：减少apply使用
            data['genres_list'] = data['genres'].str.split(',').apply(
                lambda x: [g.strip() for g in x] if isinstance(x, list) else []
            )
            data['main_genre'] = data['genres_list'].str[0].fillna('Unknown')
            data['genre_count'] = data['genres_list'].str.len().fillna(0).astype(int)

            # 2. 提取制作公司（逗号分隔）
            data['production_companies_list'] = data['production_companies'].str.split(',').apply(
                lambda x: [c.strip() for c in x] if isinstance(x, list) else []
            )
            data['production_company_count'] = data['production_companies_list'].str.len().fillna(0).astype(int)

            # 3. 提取制作国家（逗号分隔）
            data['production_countries_list'] = data['production_countries'].str.split(',').apply(
                lambda x: [c.strip() for c in x] if isinstance(x, list) else []
            )

            # 4. 提取语言（逗号分隔）
            data['spoken_languages_list'] = data['spoken_languages'].str.split(',').apply(
                lambda x: [lang.strip() for lang in x] if isinstance(x, list) else []
            )

            # 5. 提取关键词（逗号分隔）
            if 'keywords' in data.columns:
                data['keywords_list'] = data['keywords'].str.split(',').apply(
                    lambda x: [k.strip() for k in x] if isinstance(x, list) else []
                )
                data['keyword_count'] = data['keywords_list'].str.len().fillna(0).astype(int)

            # 新格式没有crew和cast信息，添加默认值
            if 'crew' not in data.columns:
                data['director'] = 'Unknown'
                data['top_3_actors'] = data.apply(lambda _: [], axis=1)
                data['actor_count'] = 0

        else:
            # 旧格式：JSON格式处理
            self.logger.info("使用JSON格式处理...")

            # 定义安全的eval函数
            def safe_eval(x):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []

            # 定义通用的特征提取函数，避免重复代码和重复的safe_eval调用
            def extract_field_names(json_str):
                try:
                    data = safe_eval(json_str)
                    if isinstance(data, list):
                        return [item['name'] for item in data]
                    return []
                except Exception:
                    return []

            # 1. 提取电影类型
            data['genres_list'] = data['genres'].apply(extract_field_names)
            data['main_genre'] = data['genres_list'].str[0].fillna('Unknown')
            data['genre_count'] = data['genres_list'].str.len().fillna(0).astype(int)

            # 2. 提取导演（仅当crew列存在时）
            if 'crew' in data.columns:
                def extract_director(crew_str):
                    crew = safe_eval(crew_str)
                    directors = [member['name'] for member in crew if member['job'] == 'Director']
                    return directors[0] if directors else 'Unknown'

                data['director'] = data['crew'].apply(extract_director)
            else:
                data['director'] = 'Unknown'

            # 3. 提取演员（前3位）（仅当cast列存在时）
            if 'cast' in data.columns:
                def extract_top_cast(cast_str, top_n=3):
                    cast = safe_eval(cast_str)
                    return [actor['name'] for actor in cast[:top_n]]

                data['top_3_actors'] = data['cast'].apply(extract_top_cast)
                data['actor_count'] = data['cast'].apply(lambda x: len(extract_field_names(x)))
            else:
                data['top_3_actors'] = data.apply(lambda _: [], axis=1)
                data['actor_count'] = 0

            # 4. 提取制作公司
            data['production_companies_list'] = data['production_companies'].apply(extract_field_names)
            data['production_company_count'] = data['production_companies_list'].str.len().fillna(0).astype(int)

            # 5. 提取关键词
            if 'keywords' in data.columns:
                data['keywords_list'] = data['keywords'].apply(extract_field_names)
                data['keyword_count'] = data['keywords_list'].str.len().fillna(0).astype(int)

            # 6. 提取制作国家
            data['production_countries_list'] = data['production_countries'].apply(extract_field_names)
            data['production_country_count'] = data['production_countries_list'].str.len().fillna(0).astype(int)

            # 7. 提取语言
            data['spoken_languages_list'] = data['spoken_languages'].apply(extract_field_names)
            data['spoken_language_count'] = data['spoken_languages_list'].str.len().fillna(0).astype(int)

        return data

    @performance_monitor
    def calculate_financial_metrics(self, data):
        """计算财务指标"""
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在计算财务指标...")

        # 检查必要的财务列是否存在
        required_columns = ['revenue', 'budget']
        if all(col in data.columns for col in required_columns):
            # 1. 票房回报率 - 处理0预算情况
            # 计算正确的投资回报率：(revenue - budget) / budget
            data['return_on_investment'] = (data['revenue'] - data['budget']) / data['budget']

            # 处理特殊情况以符合测试期望
            if is_dask:
                # Dask使用where条件
                # 预算为0时返回0.0
                data['return_on_investment'] = data['return_on_investment'].where(
                    data['budget'] != 0,
                    0.0
                )
            else:
                # Pandas使用loc条件
                # 预算为0时返回0.0
                data.loc[data['budget'] == 0, 'return_on_investment'] = 0.0

            # 2. 利润
            data['profit'] = data['revenue'] - data['budget']

            # 3. 人均票房（假设平均票价为10美元）
            data['estimated_tickets'] = data['revenue'] / 10
        else:
            self.logger.warning("缺少必要的财务数据列(revenue或budget)，跳过财务指标计算")

        return data

    @performance_monitor
    def encode_categorical_features(self, data):
        """编码分类特征"""
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在编码分类特征...")

        # Dask不支持复杂的apply操作，跳过分类特征编码
        if is_dask:
            self.logger.info("Dask模式下跳过复杂分类特征编码")
            return data

        # 1. 主要制作公司标识
        major_studios = ['Warner Bros. Pictures', 'Universal Pictures', 'Paramount Pictures',
                        '20th Century Fox', 'Walt Disney Pictures', 'Sony Pictures',
                        'Columbia Pictures', 'Lionsgate', 'DreamWorks SKG']

        data['has_major_studio'] = data['production_companies_list'].apply(
            lambda x: 1 if any(studio in x for studio in major_studios) else 0
        )

        # 2. 电影类型编码
        data = self._one_hot_encode_genres(data)

        return data

    def _one_hot_encode_genres(self, data):
        """对电影类型进行独热编码"""
        # 获取所有电影类型
        all_genres = set()
        for genres in data['genres_list']:
            all_genres.update(genres)

        # 独热编码
        for genre in all_genres:
            data[f'genre_{genre.replace(" ", "_").lower()}'] = data['genres_list'].apply(
                lambda x: 1 if genre in x else 0
            )

        return data

    @performance_monitor
    def detect_outliers(self, data, columns=['budget', 'revenue', 'runtime', 'return_on_investment'], treatment='keep'):
        """检测并可选地处理异常值

        Args:
            data: 输入数据
            columns: 要检查异常值的列
            treatment: 异常值处理方式 ('keep', 'remove', 'cap')
                - 'keep': 保留异常值
                - 'remove': 删除异常值
                - 'cap': 将异常值限制在边界值

        Returns:
            处理后的数据和异常值信息
        """
        self.logger.info(f"正在检测异常值 (处理方式: {treatment})...")

        is_dask = hasattr(data, 'compute')
        outliers = {}
        original_shape = len(data) if is_dask else data.shape[0]

        for col in columns:
            if col in data.columns:
                try:
                    # 计算四分位数 - 适配Dask和Pandas
                    if is_dask:
                        Q1 = data[col].quantile(0.25).compute()
                        Q3 = data[col].quantile(0.75).compute()
                    else:
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)

                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # 检测异常值
                    outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)

                    if is_dask:
                        # Dask需要compute获取实际数量
                        outlier_count = outlier_mask.sum().compute()
                    else:
                        outlier_count = outlier_mask.sum()

                    outliers[col] = outlier_count

                    # 只在异常值数量大于0时记录详细信息，减少日志输出
                    if outlier_count > 0:
                        self.logger.info(f"{col} 异常值: {outlier_count:,} 个 ({outlier_count / original_shape * 100:.1f}%)")
                        self.logger.info(f"  正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")

                    # 处理异常值
                    if treatment == 'remove':
                        if is_dask:
                            data = data[~outlier_mask]
                            self.logger.info(f"  删除异常值后剩余: {len(data)} 行")
                        else:
                            data = data[~outlier_mask]
                            self.logger.info(f"  删除异常值后剩余: {len(data):,} 行")
                    elif treatment == 'cap':
                        if is_dask:
                            # Dask使用where条件
                            data[col] = data[col].where(data[col] >= lower_bound, lower_bound)
                            data[col] = data[col].where(data[col] <= upper_bound, upper_bound)
                        else:
                            data.loc[outlier_mask & (data[col] < lower_bound), col] = lower_bound
                            data.loc[outlier_mask & (data[col] > upper_bound), col] = upper_bound
                        self.logger.info("  异常值已限制在正常范围内")

                except Exception as e:
                    self.logger.warning(f"检测{col}列异常值时出错: {e}")
                    outliers[col] = 0

        if treatment == 'remove' and not is_dask:
            data = data.reset_index(drop=True)

        return data, outliers

    def save_processed_data(self, data, filename="cleaned_movie_data.csv"):
        """保存处理后的数据"""
        file_path = os.path.join(self.processed_dir, filename)

        if hasattr(data, 'compute'):  # 检查是否为Dask DataFrame
            self.logger.info("正在使用Dask并行保存数据...")
            with ProgressBar():
                data.to_csv(file_path, index=False, single_file=True)
        else:
            data.to_csv(file_path, index=False)

        self.logger.info(f"处理后的数据已保存到: {file_path}")
        return file_path

    def merge_related_datasets(self, main_data, datasets_to_merge=None, use_checkpoint=False):
        """合并相关数据集

        Args:
            main_data: 主数据集
            datasets_to_merge: 要合并的数据集列表，可以是['credits', 'keywords', 'ratings', 'movielens_1m']
            use_checkpoint: 是否使用检查点功能

        Returns:
            合并后的数据集
        """
        if datasets_to_merge is None:
            datasets_to_merge = []

        merged_data = main_data.copy()

        # 检查是否有检查点
        checkpoint_name = f"merge_{hash(str(datasets_to_merge))}"
        if use_checkpoint:
            checkpoint = self.load_checkpoint(checkpoint_name)
            if checkpoint:
                # 恢复已合并的数据集和剩余需要合并的数据集
                merged_data = checkpoint['merged_data']
                processed_datasets = checkpoint['processed_datasets']
                remaining_datasets = [d for d in datasets_to_merge if d not in processed_datasets]
                self.logger.info(f"从检查点恢复，已完成 {len(processed_datasets)} 个数据集合并，剩余 {len(remaining_datasets)} 个")
                datasets_to_merge = remaining_datasets
            else:
                processed_datasets = []
        else:
            processed_datasets = []

        for dataset_name in datasets_to_merge:
            self.logger.info(f"\n正在合并{dataset_name}数据集...")

            # 定义数据集文件映射
            dataset_files = {
                'credits': 'credits.csv',
                'keywords': 'keywords.csv',
                'ratings': 'ratings.csv',
                'movielens_1m': 'movies.csv'  # MovieLens 1M数据集
            }

            if dataset_name not in dataset_files:
                self.logger.warning(f"不支持的数据集 {dataset_name}")
                continue

            filename = dataset_files[dataset_name]
            file_path = os.path.join(self.raw_dir, filename)

            if not os.path.exists(file_path):
                # 尝试在ml-1m子目录中查找MovieLens数据集
                if dataset_name == 'movielens_1m':
                    ml_1m_dir = os.path.join(self.raw_dir, 'ml-1m')
                    file_path = os.path.join(ml_1m_dir, 'movies.dat')
                    if not os.path.exists(file_path):
                        self.logger.warning(f"MovieLens 1M数据集文件不存在: {file_path}")
                        continue
                else:
                    self.logger.warning(f"文件 {file_path} 不存在")
                    continue

            try:
                # 加载MovieLens 1M数据集的特殊处理
                if dataset_name == 'movielens_1m':
                    self.logger.info("加载MovieLens 1M数据集...")

                    # MovieLens 1M movies.dat文件格式：MovieID::Title::Genres
                    ml_movies = pd.read_csv(
                        file_path, sep='::', engine='python',
                        names=['ml_movie_id', 'ml_title', 'ml_genres'],
                        encoding='latin-1'
                    )

                    # 处理MovieLens电影标题，提取年份
                    ml_movies['ml_year'] = ml_movies['ml_title'].str.extract(r'\((\d{4})\)$', expand=False)
                    ml_movies['ml_title_clean'] = ml_movies['ml_title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()

                    # 处理评分数据
                    ratings_file = os.path.join(os.path.dirname(file_path), 'ratings.dat')
                    if os.path.exists(ratings_file):
                        self.logger.info("加载MovieLens 1M评分数据...")
                        ml_ratings = pd.read_csv(
                            ratings_file, sep='::', engine='python',
                            names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                            encoding='latin-1'
                        )

                        # 计算电影平均评分和评分数量
                        ml_rating_stats = ml_ratings.groupby('MovieID').agg({
                            'Rating': ['mean', 'count']
                        }).reset_index()
                        ml_rating_stats.columns = ['ml_movie_id', 'ml_avg_rating', 'ml_rating_count']

                        # 合并评分统计到电影数据
                        ml_movies = ml_movies.merge(ml_rating_stats, on='ml_movie_id', how='left')

                    # 合并到主数据集（使用标题匹配）
                    # 清洗主数据集的标题
                    merged_data['title_clean'] = merged_data['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip().str.lower()
                    ml_movies['ml_title_clean_lower'] = ml_movies['ml_title_clean'].str.lower()

                    # 合并数据集
                    merged_data = merged_data.merge(ml_movies, left_on='title_clean', right_on='ml_title_clean_lower', how='left')

                    # 删除临时列
                    merged_data = merged_data.drop(['title_clean', 'ml_title_clean_lower'], axis=1)

                    self.logger.info(f"MovieLens 1M数据集合并完成，新添加特征: {[col for col in ml_movies.columns if col not in merged_data.columns]}")

                # 其他数据集的常规处理
                else:
                    # 使用重试机制加载数据集
                    def load_dataset():
                        if dataset_name == 'credits':
                            # credits文件较大，使用分块加载
                            chunks = []
                            for chunk in pd.read_csv(file_path, chunksize=100000, encoding='utf-8'):
                                chunks.append(chunk)
                            return pd.concat(chunks, ignore_index=True)
                        else:
                            return pd.read_csv(file_path, encoding='utf-8')

                    related_data = self.retry_operation(load_dataset, max_retries=3, delay=2)

                    self.logger.info(f"{dataset_name}数据集形状: {related_data.shape}")

                    # 确保id列类型一致
                    merged_data['id'] = pd.to_numeric(merged_data['id'], errors='coerce')
                    related_data['id'] = pd.to_numeric(related_data['id'], errors='coerce')

                    # 合并数据集
                    merged_data = pd.merge(merged_data, related_data, on='id', how='left')
                    self.logger.info(f"合并后数据集形状: {merged_data.shape}")

                # 标记该数据集已处理
                processed_datasets.append(dataset_name)

                # 保存检查点
                if use_checkpoint:
                    checkpoint_data = {
                        'merged_data': merged_data,
                        'processed_datasets': processed_datasets,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_checkpoint(checkpoint_name, checkpoint_data)

            except Exception as e:
                self.logger.error(f"合并{dataset_name}数据集失败: {e}")
                traceback.print_exc()

        # 清除已完成的检查点
        if use_checkpoint:
            self.clear_checkpoint(checkpoint_name)

        return merged_data

    def validate_data(self, data, deep_check=True):
        """验证处理后的数据质量

        Args:
            data: 要验证的数据
            deep_check: 是否执行深度检查（包括一致性和业务规则验证）

        Returns:
            bool: 数据是否通过验证
        """
        self.logger.info("正在验证数据质量...")

        # 验证结果字典
        validation_results = {
            'basic_integrity': True,
            'data_types': True,
            'consistency': True,
            'business_rules': True
        }

        issues = []

        # 1. 基本数据完整性检查
        self.logger.info("\n1. 基本数据完整性检查:")
        self.logger.info(f"   数据形状: {data.shape}")
        self.logger.info(f"   数据类型数量: {len(data.dtypes.unique())}")

        # 检查重复行 - 排除包含列表类型的列
        non_list_columns = []
        for col in data.columns:
            # 检查列中是否包含列表类型
            try:
                # 如果列中包含列表，尝试哈希时会出错
                hash(data[col].iloc[0] if len(data) > 0 else '')
                non_list_columns.append(col)
            except TypeError:
                # 跳过包含列表类型的列
                continue

        duplicate_rows = data.duplicated(subset=non_list_columns).sum()
        if duplicate_rows > 0:
            self.logger.warning(f"警告: 发现 {duplicate_rows} 行重复数据")
            self.logger.warning(f"   重复行数: {duplicate_rows} (警告)")
        else:
            self.logger.info("   重复行数: 0")

        # 检查关键列的缺失值
        critical_columns = ['id', 'title', 'release_date', 'budget', 'revenue']
        self.logger.info("\n   关键列缺失值检查:")
        for col in critical_columns:
            if col in data.columns:
                missing = data[col].isnull().sum()
                missing_pct = (missing / len(data)) * 100
                if missing > 0:
                    issues.append(f"警告: {col} 列有 {missing} 个缺失值 ({missing_pct:.2f}%)")
                    self.logger.warning(f"   - {col}: {missing} 个缺失值 ({missing_pct:.2f}%)")
                else:
                    self.logger.info(f"   - {col}: 无缺失值")

        # 2. 数据类型验证
        self.logger.info("\n2. 数据类型验证:")
        expected_dtypes = {
            'id': 'int32',
            'budget': 'int64',
            'revenue': 'int64',
            'runtime': 'float32',
            'vote_average': 'float32',
            'vote_count': 'int32',
            'popularity': 'float32'
        }

        for col, expected_dtype in expected_dtypes.items():
            if col in data.columns:
                actual_dtype = str(data[col].dtype)
                if actual_dtype != expected_dtype:
                    issues.append(f"警告: {col} 列数据类型不匹配 (预期: {expected_dtype}, 实际: {actual_dtype})")
                    self.logger.warning(f"   - {col}: 预期 {expected_dtype}, 实际 {actual_dtype}")
                else:
                    self.logger.info(f"   - {col}: 数据类型正确 ({expected_dtype})")

        if deep_check:
            # 3. 数据一致性检查
            self.logger.info("\n3. 数据一致性检查:")

            # 检查release_date格式
            if 'release_date' in data.columns:
                try:
                    data['release_date'].apply(lambda x: isinstance(x, (datetime, np.datetime64)) or pd.to_datetime(x, errors='raise'))
                    self.logger.info("   - release_date: 格式正确")
                except Exception:
                    issues.append("错误: release_date 列包含无效日期格式")
                    validation_results['consistency'] = False
                    self.logger.error("   - release_date: 包含无效日期格式 (错误)")

            # 检查预算和票房的一致性
            if 'budget' in data.columns and 'revenue' in data.columns:
                # 计算预算回报率
                data['roi'] = data['revenue'] / data['budget']
                data['roi'] = data['roi'].replace([np.inf, -np.inf], np.nan)

                # 检查异常回报率（过低或过高）
                extreme_roi = data[(data['roi'] > 100) | (data['roi'] < 0.01)]
                if len(extreme_roi) > 0:
                    issues.append(f"警告: 发现 {len(extreme_roi)} 条异常回报率记录")
                    self.logger.warning(f"   - ROI: 发现 {len(extreme_roi)} 条异常记录")

                # 检查负预算或负票房
                negative_values = data[(data['budget'] < 0) | (data['revenue'] < 0)]
                if len(negative_values) > 0:
                    issues.append(f"错误: 发现 {len(negative_values)} 条负预算或负票房记录")
                    validation_results['consistency'] = False
                    self.logger.error(f"   - 预算/票房: 发现 {len(negative_values)} 条负值记录 (错误)")

            # 检查评分一致性
            if 'vote_count' in data.columns and 'vote_average' in data.columns:
                # 低投票数的高评分电影
                low_vote_high_score = data[(data['vote_count'] < 10) & (data['vote_average'] > 8)]
                if len(low_vote_high_score) > 0:
                    issues.append(f"警告: 发现 {len(low_vote_high_score)} 条低投票数但高评分的电影")
                    self.logger.warning(f"   - 评分: 发现 {len(low_vote_high_score)} 条低投票数高评分记录")

            # 4. 业务规则验证
            self.logger.info("\n4. 业务规则验证:")

            # 检查电影时长合理性
            if 'runtime' in data.columns:
                # 筛选异常时长的电影（过短或过长）
                abnormal_runtime = data[(data['runtime'] < 30) | (data['runtime'] > 300)]
                if len(abnormal_runtime) > 0:
                    issues.append(f"警告: 发现 {len(abnormal_runtime)} 条异常时长的电影")
                    self.logger.warning(f"   - 时长: 发现 {len(abnormal_runtime)} 条异常记录 (小于30分钟或大于300分钟)")

            # 检查上映日期范围
            if 'release_date' in data.columns:
                try:
                    release_dates = pd.to_datetime(data['release_date'], errors='coerce')

                    # 检查未来日期
                    future_dates = release_dates[release_dates > datetime.now()]
                    if len(future_dates) > 0:
                        issues.append(f"警告: 发现 {len(future_dates)} 条未来上映日期记录")
                        self.logger.warning(f"   - 上映日期: 发现 {len(future_dates)} 条未来日期记录")

                    # 检查过旧日期
                    old_dates = release_dates[release_dates < pd.to_datetime('1900-01-01')]
                    if len(old_dates) > 0:
                        issues.append(f"警告: 发现 {len(old_dates)} 条1900年前的上映日期记录")
                        self.logger.warning(f"   - 上映日期: 发现 {len(old_dates)} 条1900年前的记录")
                except Exception as e:
                    self.logger.error(f"   - 上映日期: 验证失败: {e}")

            # 检查预算合理性
            if 'budget' in data.columns:
                # 过低预算（低于1000美元）
                low_budget = data[(data['budget'] > 0) & (data['budget'] < 1000)]
                if len(low_budget) > 0:
                    issues.append(f"警告: 发现 {len(low_budget)} 条过低预算记录")
                    self.logger.warning(f"   - 预算: 发现 {len(low_budget)} 条过低预算记录 (<$1000)")

        # 5. 关键列数据范围检查
        self.logger.info("\n5. 关键列数据范围检查:")
        key_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
        for col in key_columns:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                mean_val = data[col].mean()
                self.logger.info(f"   - {col}: 最小值={min_val:.2f}, 最大值={max_val:.2f}, 平均值={mean_val:.2f}")

        # 6. 生成验证报告
        self.logger.info(f"\n{'-' * 50}")
        self.logger.info("验证报告总结:")
        self.logger.info(f"  总记录数: {len(data)}")
        self.logger.info(f"  总列数: {len(data.columns)}")
        self.logger.info(f"  发现问题: {len(issues)}")

        if issues:
            self.logger.info("\n  详细问题:")
            for i, issue in enumerate(issues, 1):
                self.logger.info(f"    {i}. {issue}")
        else:
            self.logger.info("\n  数据验证通过，未发现问题！")

        # 检查验证结果
        all_passed = all(validation_results.values())
        self.logger.info(f"\n验证结果: {'通过' if all_passed else '失败'}")

        return all_passed

    def clean_unnecessary_files(self, cleanup_rules=None):
        """清理无用文件

        Args:
            cleanup_rules: 清理规则配置，可以是字典
                          例如: {'temp_files': True, 'old_versions': True, 'log_files': True}

        Returns:
            int: 清理的文件数量
        """
        self.logger.info("\n" + "=" * 40)
        self.logger.info("开始清理无用文件")
        self.logger.info("=" * 40)

        # 默认清理规则
        default_rules = {
            'temp_files': True,
            'old_versions': True,
            'log_files': True,
            'backup_files': True
        }

        rules = cleanup_rules if cleanup_rules else default_rules

        # 定义要清理的文件模式
        patterns = []

        if rules.get('temp_files'):
            patterns.extend(['*.tmp', '*temp*', '*Temp*'])

        if rules.get('old_versions'):
            patterns.extend(['*old*', '*Old*', '*_old*', '*_v[0-9]*.csv'])

        if rules.get('log_files'):
            patterns.extend(['*.log', '*log*', '*Log*'])

        if rules.get('backup_files'):
            patterns.extend(['*.bak', '*backup*', '*Backup*', '*_backup*'])

        import glob
        import os

        deleted_count = 0

        # 遍历数据目录
        # 使用类已有的目录属性
        data_dirs = [
            self.raw_dir,
            self.processed_dir,
            os.path.join(self.base_dir, 'external')  # 外部数据目录
        ]

        for full_dir in data_dirs:

            if not os.path.exists(full_dir):
                continue

            self.logger.info(f"\n搜索目录: {full_dir}")

            for pattern in patterns:
                files = glob.glob(os.path.join(full_dir, pattern))

                for file_path in files:
                        try:
                            # 跳过重要的原始数据文件
                            if 'TMDB_movie_dataset_v11.csv' in file_path:
                                continue

                            # 跳过当前正在使用的处理后文件
                            if 'cleaned_movie_data.csv' in file_path:
                                continue

                            os.remove(file_path)
                            self.logger.info(f"已删除: {file_path}")
                            deleted_count += 1

                        except Exception as e:
                            self.logger.error(f"删除文件 {file_path} 失败: {str(e)}")

        # 清理根目录下的临时文件
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.logger.info(f"\n搜索根目录: {root_dir}")

        for pattern in patterns:
            files = glob.glob(os.path.join(root_dir, pattern))

            for file_path in files:
                try:
                    os.remove(file_path)
                    self.logger.info(f"已删除: {file_path}")
                    deleted_count += 1

                except Exception as e:
                    self.logger.error(f"删除文件 {file_path} 失败: {str(e)}")

        self.logger.info(f"\n清理完成！共删除 {deleted_count} 个文件")
        return deleted_count

    def run_complete_preprocessing(self, input_file=None, output_file="cleaned_movie_data.csv",
                                  outlier_treatment=None, merge_datasets=None, use_config=True):
        """运行完整的数据预处理流程

        Args:
            input_file: 输入数据文件名
            output_file: 输出数据文件名
            outlier_treatment: 异常值处理方式，'keep'保留, 'remove'删除, 'cap'盖帽
            merge_datasets: 要合并的数据集列表，可以是['credits', 'keywords', 'ratings']
            use_config: 是否使用配置文件中的参数

        Returns:
            处理后的数据集
        """
        self.logger.info("=" * 50)
        self.logger.info("开始完整的数据预处理流程")
        self.logger.info("=" * 50)

        # 检查是否为大型数据集
        is_large_dataset = input_file == "TMDB_movie_dataset_v11.csv"

        # 从配置中获取数据加载参数
        data_loading_config = self.config['data_loading']
        use_dask = data_loading_config['use_dask']
        use_chunks = data_loading_config['use_chunks'] or is_large_dataset
        chunksize = data_loading_config['chunksize']
        optimize_memory = data_loading_config['optimize_memory']

        if is_large_dataset:
            self.logger.info("检测到百万级数据集，使用高性能处理模式...")

            # 加载数据
            data = self.load_data(input_file, use_chunks=use_chunks, chunksize=chunksize, optimize_memory=optimize_memory, use_dask=use_dask)

            if data is None:
                self.logger.error("数据加载失败！")
                return None

            # 检查是否为Dask DataFrame
            if hasattr(data, 'compute') and hasattr(data, 'persist'):
                self.logger.info("使用Dask DataFrame进行并行处理...")

                # 使用Dask并行处理
                # 2. 数据清洗
                data = self.clean_data(data)

                # 3. 处理缺失值
                data = self.handle_missing_values(data)

                # 4. 转换日期时间
                data = self.convert_datetime(data)

                # 5. 提取JSON特征
                data = self.extract_json_features(data, filename=input_file)

                # 6. 计算财务指标
                data = self.calculate_financial_metrics(data)

                # 7. 编码分类特征
                data = self.encode_categorical_features(data)

                # 8. 检测并处理异常值
                detection_columns = self.config['outliers']['detection_columns']
                data, outliers = self.detect_outliers(data, columns=detection_columns, treatment=outlier_treatment)

                # 计算并转换为Pandas DataFrame
                self.logger.info("正在计算并转换为Pandas DataFrame...")
                data = data.compute()

                self.logger.info(f"Dask处理完成，最终数据大小: {data.shape}")

            else:
                # 使用分块处理
                use_chunks = True
                self.logger.info(f"使用Pandas分块处理，分块大小: {chunksize:,}")

                # 重新加载为分块迭代器
                chunks = self.load_data(input_file, use_chunks=True, chunksize=chunksize, optimize_memory=optimize_memory, use_dask=False)

                # 处理每个块
                processed_chunks = []
                chunk_idx = 0

                for chunk in tqdm(chunks, desc="处理数据块"):
                    chunk_idx += 1
                    self.logger.info(f"\n处理第 {chunk_idx} 个数据块，大小: {chunk.shape}")

                    # 2. 数据清洗
                    chunk = self.clean_data(chunk)

                    if chunk.empty:
                        continue

                    # 3. 处理缺失值
                    chunk = self.handle_missing_values(chunk)

                    # 4. 转换日期时间
                    chunk = self.convert_datetime(chunk)

                    # 5. 提取JSON特征
                    chunk = self.extract_json_features(chunk, filename=input_file)

                    # 6. 计算财务指标
                    chunk = self.calculate_financial_metrics(chunk)

                    # 7. 编码分类特征
                    chunk = self.encode_categorical_features(chunk)

                    # 8. 检测并处理异常值
                    detection_columns = self.config['outliers']['detection_columns']
                    chunk, outliers = self.detect_outliers(chunk, columns=detection_columns, treatment=outlier_treatment)

                    processed_chunks.append(chunk)

                # 合并所有处理后的块
                if processed_chunks:
                    data = pd.concat(processed_chunks, ignore_index=True)
                    self.logger.info(f"\n合并所有数据块，最终大小: {data.shape}")
                else:
                    self.logger.error("所有数据块处理后为空！")
                    return None

        else:
            # 常规数据集处理
            # 1. 加载数据
            # 对于小型测试数据集，强制使用Pandas而不是Dask
            data = self.load_data(input_file, use_chunks=use_chunks, chunksize=chunksize, optimize_memory=optimize_memory, use_dask=False)
            if data is None:
                return None

            # 2. 数据清洗
            data = self.clean_data(data)

            # 3. 处理缺失值
            data = self.handle_missing_values(data)

            # 4. 转换日期时间
            data = self.convert_datetime(data)

            # 5. 提取JSON特征
            data = self.extract_json_features(data, filename=input_file)

            # 6. 计算财务指标
            data = self.calculate_financial_metrics(data)

            # 7. 编码分类特征
            data = self.encode_categorical_features(data)

            # 8. 检测并处理异常值
            detection_columns = self.config['outliers']['detection_columns']
            data, outliers = self.detect_outliers(data, columns=detection_columns, treatment=outlier_treatment)

        # 9. 合并相关数据集
        if merge_datasets:
            data = self.merge_related_datasets(data, merge_datasets)

        # 10. 数据验证
        validation_config = self.config['validation']
        if validation_config['enabled']:
            self.validate_data(data, deep_check=validation_config['deep_check'])

        # 11. 保存处理后的数据
        self.save_processed_data(data, output_file)

        self.logger.info("=" * 50)
        self.logger.info("数据预处理流程完成")

        return data


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='电影数据预处理工具')

    # 基本参数
    parser.add_argument('--input', '-i', type=str, help='输入数据文件名')
    parser.add_argument('--output', '-o', type=str, default='cleaned_movie_data.csv', help='输出数据文件名')
    parser.add_argument('--config', '-c', type=str, default=None, help='配置文件路径')

    # 处理参数
    parser.add_argument('--outlier-treatment', '-ot', type=str, choices=['keep', 'remove', 'cap'],
                      help='异常值处理方式')
    parser.add_argument('--merge-datasets', '-md', nargs='+', choices=['credits', 'keywords', 'ratings'],
                      help='要合并的数据集列表')
    parser.add_argument('--base-dir', '-b', type=str, default='./data', help='数据基础目录')

    # 验证参数
    parser.add_argument('--deep-validation', action='store_true', help='启用深度数据验证')

    # 性能参数
    parser.add_argument('--use-chunks', action='store_true', help='强制使用分块处理')
    parser.add_argument('--chunksize', type=int, help='分块大小')

    args = parser.parse_args()

    # 初始化数据预处理对象
    preprocessor = DataPreprocessing(base_dir=args.base_dir, config_file=args.config)

    # 更新配置（如果命令行参数指定了）
    if args.outlier_treatment:
        preprocessor.config['outliers']['treatment'] = args.outlier_treatment

    if args.deep_validation:
        preprocessor.config['validation']['deep_check'] = True

    if args.use_chunks:
        preprocessor.config['data_loading']['use_chunks'] = True

    if args.chunksize:
        preprocessor.config['data_loading']['chunksize'] = args.chunksize

    # 运行预处理流程
    preprocessor.run_complete_preprocessing(
        input_file=args.input,
        output_file=args.output,
        outlier_treatment=args.outlier_treatment,
        merge_datasets=args.merge_datasets
    )


if __name__ == "__main__":
    main()
