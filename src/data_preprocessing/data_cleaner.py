# 数据清洗模块
"""
负责数据清洗的模块，包括数据过滤、重复值处理、异常值检测等功能。
"""

import logging
import pandas as pd
from src.utils.data_validation import DataValidator


class DataCleaner:
    """数据清洗器，负责数据的清洗和预处理"""

    def __init__(self, config=None):
        """初始化数据清洗器

        Args:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config if config else {}

        # 从配置中获取清洗参数
        self.remove_duplicates = self.config.get('data_cleaning', {}).get('remove_duplicates', True)
        self.min_budget = self.config.get('data_cleaning', {}).get('min_budget', 100)
        self.min_revenue = self.config.get('data_cleaning', {}).get('min_revenue', 100)
        self.max_future_years = self.config.get('data_cleaning', {}).get('max_future_years', 0)

    def clean_data(self, data):
        """数据清洗，支持Pandas和Dask DataFrame

        Args:
            data: 输入数据（Pandas或Dask DataFrame）

        Returns:
            清洗后的数据
        """
        self.logger.info("正在进行数据清洗...")

        # 使用统一的数据验证和清洗模块
        validator = DataValidator(self.config)

        # 执行数据验证和清洗
        cleaned_data, quality_report = validator.validate_and_clean_data(
            data, 
            generate_report=True, 
            report_path="results/reports/data_quality_report.csv"
        )

        # 额外的清洗步骤
        is_dask = hasattr(cleaned_data, 'compute')

        # 移除收入和预算都为0的行
        self.logger.info("正在移除无效的财务数据行...")
        if 'revenue' in cleaned_data.columns and 'budget' in cleaned_data.columns:
            if is_dask:
                cleaned_data = cleaned_data[(cleaned_data['revenue'] > 0) | (cleaned_data['budget'] > 0)]
            else:
                cleaned_data = cleaned_data[(cleaned_data['revenue'] > 0) | (cleaned_data['budget'] > 0)]

        # 处理预算和票房的异常值
        self.logger.info("正在处理预算和票房的异常值...")
        if 'budget' in cleaned_data.columns:
            cleaned_data = cleaned_data[(cleaned_data['budget'].isnull()) | (cleaned_data['budget'] >= self.min_budget)]
        if 'revenue' in cleaned_data.columns:
            cleaned_data = cleaned_data[(cleaned_data['revenue'].isnull()) | (cleaned_data['revenue'] >= self.min_revenue)]

        # 移除运行时间为0或负数的行
        self.logger.info("正在移除无效的运行时间行...")
        if 'runtime' in cleaned_data.columns:
            cleaned_data = cleaned_data[(cleaned_data['runtime'].isnull()) | (cleaned_data['runtime'] > 0)]

        self.logger.info("数据清洗完成")

        return cleaned_data

    def filter_data_by_date(self, data, start_date=None, end_date=None):
        """根据日期范围过滤数据

        Args:
            data: 输入数据
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）

        Returns:
            过滤后的数据
        """
        if 'release_date' not in data.columns:
            self.logger.warning("数据中没有release_date列，跳过日期过滤")
            return data

        self.logger.info("正在根据日期范围过滤数据...")

        # 检查数据类型
        is_dask = hasattr(data, 'compute')

        if not is_dask:
            # 将release_date转换为日期类型
            data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

            # 过滤日期范围
            if start_date:
                data = data[data['release_date'] >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data['release_date'] <= pd.to_datetime(end_date)]
        else:
            # Dask DataFrame处理
            self.logger.warning("Dask DataFrame暂不支持复杂的日期过滤")

        return data

    def filter_data_by_budget(self, data, min_budget=None, max_budget=None):
        """根据预算范围过滤数据

        Args:
            data: 输入数据
            min_budget: 最小预算
            max_budget: 最大预算

        Returns:
            过滤后的数据
        """
        if 'budget' not in data.columns:
            self.logger.warning("数据中没有budget列，跳过预算过滤")
            return data

        self.logger.info("正在根据预算范围过滤数据...")

        if min_budget:
            data = data[data['budget'] >= min_budget]
        if max_budget:
            data = data[data['budget'] <= max_budget]

        return data

    def filter_data_by_revenue(self, data, min_revenue=None, max_revenue=None):
        """根据收入范围过滤数据

        Args:
            data: 输入数据
            min_revenue: 最小收入
            max_revenue: 最大收入

        Returns:
            过滤后的数据
        """
        if 'revenue' not in data.columns:
            self.logger.warning("数据中没有revenue列，跳收入过滤")
            return data

        self.logger.info("正在根据收入范围过滤数据...")

        if min_revenue:
            data = data[data['revenue'] >= min_revenue]
        if max_revenue:
            data = data[data['revenue'] <= max_revenue]

        return data
