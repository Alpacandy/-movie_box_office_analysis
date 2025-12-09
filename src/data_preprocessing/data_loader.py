# 数据加载模块
"""
负责数据加载、检查点管理和内存优化的模块。
"""

import os
import pandas as pd
from src.utils.logging_config import get_logger

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    logger = get_logger('DataLoader')
    logger.info("Dask未安装，将使用Pandas进行处理")
    DASK_AVAILABLE = False


class DataLoader:
    """数据加载器，负责数据的加载、检查点管理和内存优化"""

    def __init__(self, base_dir="./data", config=None):
        """初始化数据加载器

        Args:
            base_dir: 数据基础目录
            config: 配置字典
        """
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.logger = get_logger('DataLoader')
        self.config = config if config else {}

        # 创建必要的目录
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def optimize_dtypes(self, data):
        """优化数据类型以减少内存使用

        Args:
            data: pandas DataFrame或Dask DataFrame

        Returns:
            优化后的数据
        """
        self.logger.info("正在优化数据类型...")

        # 检查数据类型
        is_dask = hasattr(data, 'compute')

        if is_dask:
            # Dask DataFrame优化
            for col in data.columns:
                col_dtype = str(data[col].dtype)

                if 'float' in col_dtype:
                    # 检查是否可以使用float32
                    data[col] = data[col].astype('float32')
                elif 'int' in col_dtype:
                    # 检查是否可以使用int32
                    data[col] = data[col].astype('int32')
                elif col_dtype == 'object':
                    # 检查是否是日期列
                    if 'date' in col.lower() or 'release' in col.lower():
                        # 尝试转换为datetime64
                        try:
                            data[col] = dd.to_datetime(data[col], errors='coerce')
                        except Exception as e:
                            self.logger.warning(f"Dask转换列{col}为日期类型失败: {e}")
        else:
            # Pandas DataFrame优化
            for col in data.columns:
                col_dtype = str(data[col].dtype)

                if 'float' in col_dtype:
                    # 尝试将float64转换为float32
                    try:
                        # 检查是否有超出float32范围的值
                        if not (data[col].min() < -1.0e38 or data[col].max() > 1.0e38):
                            data[col] = data[col].astype('float32')
                    except Exception as e:
                        self.logger.warning(f"转换列{col}为float32失败: {e}")
                elif 'int' in col_dtype:
                    # 根据数据范围选择合适的整数类型
                    data[col] = pd.to_numeric(data[col], downcast='integer')
                elif col_dtype == 'object':
                    # 检查是否可以转换为日期类型
                    if 'date' in col.lower() or 'release' in col.lower():
                        try:
                            data[col] = pd.to_datetime(data[col], errors='coerce')
                            continue  # 如果转换成功，继续处理下一列
                        except Exception as e:
                            self.logger.warning(f"转换列{col}为日期类型失败: {e}")

                    # 检查是否可以转换为布尔类型
                    if data[col].dropna().isin([0, 1, '0', '1', False, True, 'False', 'True']).all():
                        try:
                            data[col] = data[col].astype(bool)
                            continue
                        except Exception as e:
                            self.logger.warning(f"转换列{col}为布尔类型失败: {e}")

                    # 检查是否可以转换为分类类型
                    try:
                        unique_values = len(data[col].unique())
                        total_values = len(data[col])

                        # 如果唯一值占比小于10%，转换为分类类型
                        if unique_values / total_values < 0.1:
                            data[col] = data[col].astype('category')
                    except Exception as e:
                        self.logger.warning(f"转换列{col}为分类类型失败: {e}")
                elif col_dtype == 'datetime64[ns]':
                    # 将datetime64[ns]转换为datetime64[us]以节省内存
                    data[col] = data[col].astype('datetime64[us]')

        self.logger.info("数据类型优化完成")
        return data

    def save_checkpoint(self, data, filename="checkpoint.csv", use_dask=True):
        """保存检查点

        Args:
            data: 要保存的数据
            filename: 检查点文件名
            use_dask: 是否使用Dask保存

        Returns:
            保存是否成功
        """
        self.logger.info(f"正在保存检查点: {filename}")

        is_dask = hasattr(data, 'compute')
        checkpoint_file = os.path.join(self.processed_dir, filename)

        try:
            if is_dask:
                # Dask DataFrame保存
                data.to_csv(checkpoint_file, index=False, single_file=True)
            else:
                # Pandas DataFrame保存
                data.to_csv(checkpoint_file, index=False)

            self.logger.info(f"检查点已保存: {checkpoint_file}")
            return True
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            return False

    def load_checkpoint(self, filename="checkpoint.csv", use_dask=True):
        """加载检查点

        Args:
            filename: 检查点文件名
            use_dask: 是否使用Dask加载

        Returns:
            加载的数据，如果失败则返回None
        """
        self.logger.info(f"正在加载检查点: {filename}")

        checkpoint_file = os.path.join(self.processed_dir, filename)

        if not os.path.exists(checkpoint_file):
            self.logger.info(f"检查点不存在: {checkpoint_file}")
            return None

        try:
            if use_dask and DASK_AVAILABLE:
                # 使用Dask加载
                data = dd.read_csv(checkpoint_file, dtype={'id': 'int64'})
            else:
                # 使用Pandas加载
                data = pd.read_csv(checkpoint_file)

            self.logger.info(f"检查点已加载: {checkpoint_file}")
            return data
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return None

    def clear_checkpoint(self, filename="checkpoint.csv"):
        """清除检查点

        Args:
            filename: 检查点文件名

        Returns:
            清除是否成功
        """
        checkpoint_file = os.path.join(self.processed_dir, filename)

        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                self.logger.info(f"已清除检查点: {checkpoint_file}")
            except Exception as e:
                self.logger.error(f"清除检查点失败: {e}")
                return False
        else:
            self.logger.info(f"检查点不存在: {checkpoint_file}")

        return True

    def load_data(self, filename="tmdb_merged.csv", use_chunks=True, chunksize=200000, optimize_memory=True, use_dask=True):
        """加载原始数据

        Args:
            filename: 数据文件名
            use_chunks: 是否使用分块加载
            chunksize: 分块大小
            optimize_memory: 是否优化内存使用
            use_dask: 是否使用Dask加载

        Returns:
            加载的数据，如果失败则返回None
        """
        self.logger.info(f"正在加载数据: {filename}")

        file_path = os.path.join(self.raw_dir, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"错误: 文件不存在: {file_path}")
            self.logger.info("请检查文件名是否正确，或先运行数据获取脚本")
            return None

        try:
            if use_dask and DASK_AVAILABLE:
                # 使用Dask加载
                self.logger.info("使用Dask加载数据...")
                data = dd.read_csv(file_path, dtype={'id': 'int64'})

                if optimize_memory:
                    data = self.optimize_dtypes(data)
            elif use_chunks:
                # 使用Pandas分块加载
                self.logger.info(f"使用Pandas分块加载数据，块大小: {chunksize}")
                chunks = []

                for chunk in pd.read_csv(file_path, chunksize=chunksize):
                    if optimize_memory:
                        chunk = self.optimize_dtypes(chunk)
                    chunks.append(chunk)

                data = pd.concat(chunks, ignore_index=True)
            else:
                # 使用Pandas一次性加载
                self.logger.info("使用Pandas一次性加载数据...")
                data = pd.read_csv(file_path)

                if optimize_memory:
                    data = self.optimize_dtypes(data)

            self.logger.info(f"数据加载完成，形状: {data.shape}")
            return data
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            self.logger.info("请检查文件格式是否正确")
            return None
