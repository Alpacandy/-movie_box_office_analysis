# 缺失值处理模块
"""
负责处理数据中的缺失值，提供多种缺失值处理策略。
"""

import logging
import numpy as np
import importlib.util
from sklearn.impute import IterativeImputer, KNNImputer


# 检查Dask是否可用
try:
    DASK_AVAILABLE = importlib.util.find_spec('dask.dataframe') is not None
    if DASK_AVAILABLE:
        logging.info("Dask已安装，可以使用Dask DataFrame")
    else:
        logging.info("Dask未安装，将使用Pandas进行处理")
except Exception as e:
    logging.info(f"检查Dask可用性时出错: {e}，将使用Pandas进行处理")
    DASK_AVAILABLE = False


class MissingValueHandler:
    """缺失值处理器，提供多种缺失值处理策略"""

    def __init__(self):
        """初始化缺失值处理器"""
        self.logger = logging.getLogger(__name__)

    def handle_missing_values(self, data):
        """处理缺失值并进行数据质量检查

        Args:
            data: 输入数据（Pandas或Dask DataFrame）

        Returns:
            处理后的数据
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在处理缺失值和数据质量检查...")

        # 获取数据大小信息
        if is_dask:
            data_size = f"Dask DataFrame: {data.npartitions} partitions, {len(data.columns)} columns"
            total_rows = len(data)
        else:
            data_size = f"Pandas DataFrame: {data.shape[0]} rows, {data.shape[1]} columns"
            total_rows = data.shape[0]

        self.logger.info(f"数据大小: {data_size}")

        # 1. 删除高缺失率列（缺失率超过70%）
        self.logger.info("正在检查和删除高缺失率列...")
        missing_threshold = 0.7  # 70%缺失率阈值

        if is_dask:
            # Dask DataFrame - 计算每列缺失率
            missing_counts = data.isnull().sum().compute()
            missing_rates = missing_counts / total_rows
            columns_to_drop = [col for col, rate in missing_rates.items() if rate > missing_threshold]

            if columns_to_drop:
                data = data.drop(columns=columns_to_drop)
                self.logger.info(f"删除高缺失率列: {columns_to_drop}")
        else:
            # Pandas DataFrame - 计算每列缺失率
            missing_rates = data.isnull().mean()
            columns_to_drop = missing_rates[missing_rates > missing_threshold].index.tolist()

            if columns_to_drop:
                data = data.drop(columns=columns_to_drop)
                self.logger.info(f"删除高缺失率列: {columns_to_drop}")

        # 1. 处理数值型特征缺失值
        self.logger.info("正在处理数值型特征缺失值...")
        numeric_columns = self._get_numeric_columns(data)

        if numeric_columns:
            if is_dask:
                # Dask DataFrame - 简单的均值填充
                for col in numeric_columns:
                    data[col] = data[col].fillna(data[col].mean())
            else:
                # Pandas DataFrame - 智能填充
                data = self._handle_numeric_missing_values(data, numeric_columns)

        # 2. 处理分类型特征缺失值
        self.logger.info("正在处理分类型特征缺失值...")
        categorical_columns = self._get_categorical_columns(data)

        if categorical_columns:
            if is_dask:
                # Dask DataFrame - 使用众数填充
                for col in categorical_columns:
                    data[col] = data[col].fillna(data[col].mode().compute()[0])
            else:
                # Pandas DataFrame - 使用众数填充
                for col in categorical_columns:
                    data[col] = data[col].fillna(data[col].mode()[0])

        # 3. 处理文本型特征缺失值
        self.logger.info("正在处理文本型特征缺失值...")
        text_columns = self._get_text_columns(data)

        if text_columns:
            # 使用'Unknown'填充文本缺失值
            for col in text_columns:
                data[col] = data[col].fillna('Unknown')

        # 4. 数据质量检查
        self.logger.info("正在进行数据质量检查...")
        self._perform_data_quality_check(data)

        return data

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

        return numeric_cols

    def _get_categorical_columns(self, data):
        """获取分类型列

        Args:
            data: 输入数据

        Returns:
            分类型列列表
        """
        is_dask = hasattr(data, 'compute')

        if is_dask:
            # Dask DataFrame
            categorical_cols = [col for col in data.columns if 'category' in str(data[col].dtype)]
        else:
            # Pandas DataFrame
            categorical_cols = data.select_dtypes(include=['category', 'object']).columns.tolist()

        # 排除文本型列
        text_cols = self._get_text_columns(data)
        categorical_cols = [col for col in categorical_cols if col not in text_cols]

        return categorical_cols

    def _get_text_columns(self, data):
        """获取文本型列

        Args:
            data: 输入数据

        Returns:
            文本型列列表
        """
        # 定义常见的文本型列
        text_columns = ['title', 'overview', 'tagline', 'original_title', 'imdb_id', 'poster_path', 'backdrop_path']

        # 只返回数据中存在的文本列
        return [col for col in text_columns if col in data.columns]

    def _handle_numeric_missing_values(self, data, numeric_columns):
        """处理数值型特征的缺失值

        Args:
            data: 输入数据
            numeric_columns: 数值型列列表

        Returns:
            处理后的数据
        """
        for col in numeric_columns:
            missing_ratio = data[col].isnull().sum() / len(data)

            if missing_ratio < 0.05:  # 缺失值比例小于5%，使用均值填充
                data[col] = data[col].fillna(data[col].mean())
            elif missing_ratio < 0.3:  # 缺失值比例在5%-30%之间，使用KNN插值
                self.logger.info(f"使用KNN插值处理{col}的缺失值...")
                data = self._knn_imputation(data, [col])
            elif missing_ratio < 0.5:  # 缺失值比例在30%-50%之间，使用迭代插值
                self.logger.info(f"使用迭代插值处理{col}的缺失值...")
                data = self._iterative_imputation(data, [col])
            else:  # 缺失值比例大于50%，考虑删除该列
                self.logger.warning(f"{col}缺失值比例过高 ({missing_ratio:.2%})，考虑删除该列")

        return data

    def _knn_imputation(self, data, columns, n_neighbors=5):
        """使用KNN插值处理缺失值

        Args:
            data: 输入数据
            columns: 要处理的列
            n_neighbors: KNN邻居数量

        Returns:
            处理后的数据
        """
        try:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            data[columns] = imputer.fit_transform(data[columns])
            return data
        except Exception as e:
            self.logger.error(f"KNN插值失败: {e}，回退到均值填充")
            for col in columns:
                data[col] = data[col].fillna(data[col].mean())
            return data

    def _iterative_imputation(self, data, columns, max_iter=10, random_state=42):
        """使用迭代插值处理缺失值

        Args:
            data: 输入数据
            columns: 要处理的列
            max_iter: 最大迭代次数
            random_state: 随机种子

        Returns:
            处理后的数据
        """
        try:
            imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
            data[columns] = imputer.fit_transform(data[columns])
            return data
        except Exception as e:
            self.logger.error(f"迭代插值失败: {e}，回退到均值填充")
            for col in columns:
                data[col] = data[col].fillna(data[col].mean())
            return data

    def _perform_data_quality_check(self, data):
        """执行数据质量检查

        Args:
            data: 输入数据
        """
        is_dask = hasattr(data, 'compute')

        # 检查缺失值情况
        if is_dask:
            missing_data = data.isnull().sum().compute()
        else:
            missing_data = data.isnull().sum()

        total_missing = missing_data.sum()

        if total_missing == 0:
            self.logger.info("✅ 数据质量检查通过：无缺失值")
        else:
            self.logger.warning(f"⚠️  仍有缺失值: {total_missing} 个")

            # 显示缺失值最多的5列
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                top_missing = missing_data.sort_values(ascending=False).head(5)
                self.logger.warning("缺失值最多的5列:")
                for col, count in top_missing.items():
                    self.logger.warning(f"  {col}: {count} missing values")

    def detect_outliers(self, data, method='zscore', threshold=3):
        """检测异常值

        Args:
            data: 输入数据
            method: 检测方法，可选 'zscore' 或 'iqr'
            threshold: 异常值阈值

        Returns:
            包含异常值标记的数据
        """
        is_dask = hasattr(data, 'compute')

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持异常值检测")
            return data

        numeric_columns = self._get_numeric_columns(data)

        for col in numeric_columns:
            if method == 'zscore':
                # 使用Z-score方法检测异常值
                z_scores = (data[col] - data[col].mean()) / data[col].std()
                data[f'{col}_outlier'] = (abs(z_scores) > threshold).astype(int)
            elif method == 'iqr':
                # 使用IQR方法检测异常值
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[f'{col}_outlier'] = ((data[col] < lower_bound) | (data[col] > upper_bound)).astype(int)

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

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持异常值移除")
            return data

        # 先检测异常值
        data_with_outliers = self.detect_outliers(data, method, threshold)

        numeric_columns = self._get_numeric_columns(data)

        # 移除异常值
        for col in numeric_columns:
            outlier_col = f'{col}_outlier'
            if outlier_col in data_with_outliers.columns:
                data = data[data_with_outliers[outlier_col] == 0]

        return data
