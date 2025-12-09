# 异常值检测模块
"""
负责检测和处理数据中的异常值，提供多种异常值检测方法。
"""

import logging
import pandas as pd
import numpy as np


try:
    import importlib.util # 用于检查Dask是否安装
    DASK_AVAILABLE = importlib.util.find_spec("dask.dataframe") is not None  # 检查Dask是否安装
    DASK_AVAILABLE = True  # 检查Dask是否安装
except ImportError:
    logging.info("Dask未安装，将使用Pandas进行处理")
    DASK_AVAILABLE = False


class OutlierDetector:
    """异常值检测器，提供多种异常值检测和处理方法"""

    def __init__(self):
        """初始化异常值检测器"""
        self.logger = logging.getLogger(__name__)

    def detect_outliers(self, data, method='zscore', threshold=1.5, columns=None, treatment=None):
        """检测异常值

        Args:
            data: 输入数据（Pandas或Dask DataFrame）
            method: 检测方法，可选 'zscore' 或 'iqr'
            threshold: 异常值阈值
            columns: 要检测异常值的列列表，如果为None则检测所有数值型列
            treatment: 异常值处理方式，可选 'remove' 或 'cap'

        Returns:
            如果treatment为None，则返回包含异常值标记的数据
            否则返回两个值：处理后的数据和被检测出的异常值
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info(f"正在使用{method}方法检测异常值，阈值: {threshold}...")

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持异常值检测")
            return data

        # 获取要检测的列
        numeric_columns = self._get_numeric_columns(data)
        if columns is not None:
            # 只检测指定的列
            numeric_columns = [col for col in numeric_columns if col in columns]

        # 复制数据以避免修改原始数据
        data_copy = data.copy()

        for col in numeric_columns:
            if method == 'zscore':
                # 使用Z-score方法检测异常值
                self.logger.info(f"正在检测{col}的异常值...")
                data_copy = self._detect_outliers_zscore(data_copy, col, threshold)
            elif method == 'iqr':
                # 使用IQR方法检测异常值
                self.logger.info(f"正在检测{col}的异常值...")
                data_copy = self._detect_outliers_iqr(data_copy, col, threshold)

        # 如果没有指定处理方式，只返回标记了异常值的数据
        if treatment is None:
            return data_copy

        # 收集所有异常值标记列
        outlier_columns = [col for col in data_copy.columns if col.endswith('_outlier')]

        if not outlier_columns:
            # 没有检测到异常值
            return data_copy, pd.DataFrame()

        # 检测是否有任何异常值
        any_outlier = data_copy[outlier_columns].any(axis=1)
        outliers = data_copy[any_outlier].copy()

        # 调试信息
        self.logger.debug(f"Outlier columns: {outlier_columns}")
        self.logger.debug("Any outlier:")
        self.logger.debug(f"{any_outlier}")
        self.logger.debug(f"Outlier count: {any_outlier.sum()}")
        self.logger.debug("Data copy with outliers:")
        self.logger.debug(f"{data_copy[outlier_columns]}")

        if treatment == 'remove':
            # 删除异常值
            data_clean = data_copy[~any_outlier].copy()
            # 删除异常值标记列
            for col in outlier_columns:
                if col in data_clean.columns:
                    del data_clean[col]
            return data_clean, outliers
        elif treatment == 'cap':
            # 盖帽处理异常值
            data_capped = data_copy.copy()
            for col in numeric_columns:
                outlier_col = f'{col}_outlier'
                if outlier_col in data_capped.columns:
                    # 获取正常范围
                    if method == 'zscore':
                        mean = data_capped[col].mean()
                        std = data_capped[col].std()
                        upper_limit = mean + threshold * std
                        lower_limit = mean - threshold * std
                    elif method == 'iqr':
                        Q1 = data_capped[col].quantile(0.25)
                        Q3 = data_capped[col].quantile(0.75)
                        IQR = Q3 - Q1
                        upper_limit = Q3 + threshold * IQR
                        lower_limit = Q1 - threshold * IQR
                    else:
                        upper_limit = data_capped[col].max()
                        lower_limit = data_capped[col].min()

                    # 盖帽处理
                    data_capped.loc[data_capped[outlier_col] == 1, col] = np.clip(
                        data_capped.loc[data_capped[outlier_col] == 1, col],
                        lower_limit,
                        upper_limit
                    )

                    # 删除异常值标记列
                    del data_capped[outlier_col]
            return data_capped, outliers
        else:
            # 未知的处理方式，返回标记了异常值的数据和异常值
            return data_copy, outliers

    def _get_numeric_columns(self, data):
        """获取数值型列

        Args:
            data: 输入数据

        Returns:
            数值型列列表
        """
        is_dask = hasattr(data, 'compute')

        if is_dask:
            # Dask DataFrame
            numeric_cols = [col for col in data.columns if 'float' in str(data[col].dtype) or 'int' in str(data[col].dtype)]
        else:
            # Pandas DataFrame
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # 排除异常值标记列
        numeric_cols = [col for col in numeric_cols if not col.endswith('_outlier')]

        return numeric_cols

    def _detect_outliers_zscore(self, data, column, threshold=3):
        """使用Z-score方法检测异常值

        Args:
            data: 输入数据
            column: 要检测的列名
            threshold: 异常值阈值

        Returns:
            包含异常值标记的数据
        """
        # 计算Z-score
        mean = data[column].mean()
        std = data[column].std()

        # 调试信息
        self.logger.debug(f"\nColumn: {column}")
        self.logger.debug(f"Mean: {mean}")
        self.logger.debug(f"Std: {std}")
        self.logger.debug(f"Threshold: {threshold}")
        self.logger.debug(f"Data values: {data[column].tolist()}")

        # 避免除以0
        if std == 0:
            self.logger.warning(f"{column}的标准差为0，无法使用Z-score方法检测异常值")
            data[f'{column}_outlier'] = 0
            return data

        z_scores = (data[column] - mean) / std

        self.logger.debug(f"Z-scores: {z_scores.tolist()}")
        self.logger.debug(f"Abs Z-scores > threshold: {(abs(z_scores) > threshold).tolist()}")

        # 标记异常值
        data[f'{column}_outlier'] = (abs(z_scores) > threshold).astype(int)

        # 记录异常值数量
        outlier_count = data[f'{column}_outlier'].sum()
        self.logger.info(f"{column}检测到{outlier_count}个异常值，占比: {(outlier_count / len(data)):.2%}")

        return data

    def _detect_outliers_iqr(self, data, column, threshold=1.5):
        """使用IQR方法检测异常值

        Args:
            data: 输入数据
            column: 要检测的列名
            threshold: 异常值阈值

        Returns:
            包含异常值标记的数据
        """
        # 计算四分位数
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # 计算上下界
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # 标记异常值
        data[f'{column}_outlier'] = ((data[column] < lower_bound) | (data[column] > upper_bound)).astype(int)

        # 记录异常值数量
        outlier_count = data[f'{column}_outlier'].sum()
        self.logger.info(f"{column}检测到{outlier_count}个异常值，占比: {(outlier_count / len(data)):.2%}")

        return data

    def remove_outliers(self, data, method='zscore', threshold=3):
        """移除异常值

        Args:
            data: 输入数据
            method: 检测方法，可选 'zscore' 或 'iqr'
            threshold: 异常值阈值

        Returns:
            移除异常值后的数据
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info(f"正在移除异常值，方法: {method}，阈值: {threshold}...")

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持异常值移除")
            return data

        # 首先检测异常值
        data_with_outliers = self.detect_outliers(data, method, threshold)

        # 获取数值型列
        numeric_columns = self._get_numeric_columns(data)

        # 收集所有异常值标记列
        outlier_columns = [f'{col}_outlier' for col in numeric_columns]

        # 筛选没有任何异常值的行
        no_outliers_mask = (data_with_outliers[outlier_columns].sum(axis=1) == 0)

        # 记录移除的行数
        removed_rows = len(data_with_outliers) - no_outliers_mask.sum()
        self.logger.info(f"移除了{removed_rows}行异常值，占比: {(removed_rows / len(data_with_outliers)):.2%}")

        # 返回移除异常值后的数据
        return data_with_outliers[no_outliers_mask]

    def cap_outliers(self, data, method='zscore', threshold=3, upper_only=False, lower_only=False):
        """盖帽法处理异常值

        Args:
            data: 输入数据
            method: 检测方法，可选 'zscore' 或 'iqr'
            threshold: 异常值阈值
            upper_only: 是否只处理上界异常值
            lower_only: 是否只处理下界异常值

        Returns:
            处理后的数据
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info(f"正在使用盖帽法处理异常值，方法: {method}，阈值: {threshold}...")

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持异常值处理")
            return data

        # 获取数值型列
        numeric_columns = self._get_numeric_columns(data)

        for col in numeric_columns:
            self.logger.info(f"正在处理{col}的异常值...")

            if method == 'zscore':
                # 使用Z-score方法计算上下界
                mean = data[col].mean()
                std = data[col].std()

                if std == 0:
                    self.logger.warning(f"{col}的标准差为0，跳过异常值处理")
                    continue

                upper_bound = mean + threshold * std
                lower_bound = mean - threshold * std
            elif method == 'iqr':
                # 使用IQR方法计算上下界
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                upper_bound = Q3 + threshold * IQR
                lower_bound = Q1 - threshold * IQR

            # 处理异常值
            if not lower_only:
                # 处理上界异常值
                original_count = len(data[data[col] > upper_bound])
                data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
                self.logger.info(f"{col}上界异常值处理: {original_count}个值被盖帽")

            if not upper_only:
                # 处理下界异常值
                original_count = len(data[data[col] < lower_bound])
                data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                self.logger.info(f"{col}下界异常值处理: {original_count}个值被盖帽")

        return data

    def winsorize_outliers(self, data, lower_quantile=0.01, upper_quantile=0.99):
        """Winsorize方法处理异常值

        Args:
            data: 输入数据
            lower_quantile: 下分位数
            upper_quantile: 上分位数

        Returns:
            处理后的数据
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info(f"正在使用Winsorize方法处理异常值，分位数范围: [{lower_quantile}, {upper_quantile}]...")

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持Winsorize处理")
            return data

        # 获取数值型列
        numeric_columns = self._get_numeric_columns(data)

        for col in numeric_columns:
            self.logger.info(f"正在处理{col}的异常值...")

            # 计算分位数
            lower_bound = data[col].quantile(lower_quantile)
            upper_bound = data[col].quantile(upper_quantile)

            # 处理异常值
            original_lower_count = len(data[data[col] < lower_bound])
            original_upper_count = len(data[data[col] > upper_bound])

            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

            self.logger.info(f"{col}Winsorize处理: {original_lower_count}个下界值, {original_upper_count}个上界值被处理")

        return data

    def get_outlier_statistics(self, data):
        """获取异常值统计信息

        Args:
            data: 输入数据

        Returns:
            异常值统计信息
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在获取异常值统计信息...")

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持异常值统计")
            return None

        # 获取数值型列
        numeric_columns = self._get_numeric_columns(data)

        # 收集所有异常值标记列
        outlier_columns = [f'{col}_outlier' for col in numeric_columns if f'{col}_outlier' in data.columns]

        if not outlier_columns:
            self.logger.warning("未检测到异常值标记列，请先调用detect_outliers方法")
            return None

        # 计算异常值统计信息
        outlier_stats = {}

        for col in numeric_columns:
            outlier_col = f'{col}_outlier'

            if outlier_col in data.columns:
                outlier_count = data[outlier_col].sum()
                outlier_ratio = outlier_count / len(data)

                outlier_stats[col] = {
                    'outlier_count': int(outlier_count),
                    'outlier_ratio': float(outlier_ratio),
                    'total_rows': len(data)
                }

        return outlier_stats

    def visualize_outliers(self, data, columns=None, method='boxplot', figsize=(12, 8)):
        """可视化异常值

        Args:
            data: 输入数据
            columns: 要可视化的列列表
            method: 可视化方法，可选 'boxplot' 或 'scatter'
            figsize: 图表大小

        Returns:
            可视化图表
        """
        is_dask = hasattr(data, 'compute')

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持可视化")
            return None

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置可视化风格
            sns.set_style('darkgrid')

            # 获取数值型列
            if columns is None:
                columns = self._get_numeric_columns(data)

            self.logger.info(f"正在可视化异常值，方法: {method}...")

            if method == 'boxplot':
                # 箱线图
                plt.figure(figsize=figsize)
                sns.boxplot(data=data[columns])
                plt.xticks(rotation=45)
                plt.title('异常值箱线图')
                plt.tight_layout()
            elif method == 'scatter':
                # 散点图矩阵
                sns.pairplot(data[columns])

            plt.show()

            return plt
        except ImportError as e:
            self.logger.error(f"可视化库导入失败: {e}")
            self.logger.info("请安装matplotlib和seaborn库")
            return None
        except Exception as e:
            self.logger.error(f"可视化失败: {e}")
            return None
