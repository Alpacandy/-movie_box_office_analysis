#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据验证和清洗模块
提供数据完整性检查、一致性检查、清洗功能和质量报告生成
"""

import os
import pandas as pd
from datetime import datetime
from src.utils.logging_config import get_logger


class DataValidator:
    """
    数据验证器类，提供统一的数据验证和清洗功能
    """

    def __init__(self, config=None):
        """
        初始化数据验证器

        Args:
            config: 配置字典
        """
        self.logger = get_logger('DataValidator')
        self.config = config if config else {}
        self.quality_report = {}

    def check_data_integrity(self, data):
        """
        检查数据完整性

        Args:
            data: 输入数据（Pandas DataFrame）

        Returns:
            dict: 数据完整性报告
        """
        self.logger.info("正在检查数据完整性...")

        # 检查是否为Dask DataFrame
        is_dask = hasattr(data, 'compute')

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持完整的数据完整性检查，将使用Pandas进行检查")
            data = data.compute()

        # 初始化报告
        integrity_report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': {},
            'duplicate_rows': 0,
            'unique_columns': {}
        }

        # 检查缺失值
        missing_values = data.isnull().sum()
        integrity_report['missing_values'] = missing_values.to_dict()
        integrity_report['total_missing'] = missing_values.sum()

        # 检查重复行
        duplicate_rows = data.duplicated().sum()
        integrity_report['duplicate_rows'] = duplicate_rows

        # 检查列的唯一性
        for col in data.columns:
            integrity_report['unique_columns'][col] = data[col].nunique()

        # 检查ID列
        if 'id' in data.columns:
            integrity_report['unique_ids'] = data['id'].nunique()
            integrity_report['duplicate_ids'] = len(data) - data['id'].nunique()

        self.logger.info(f"数据完整性检查完成：总行数={integrity_report['total_rows']}, 总缺失值={integrity_report['total_missing']}, 重复行={integrity_report['duplicate_rows']}")

        return integrity_report

    def check_data_consistency(self, data):
        """
        检查数据一致性

        Args:
            data: 输入数据（Pandas DataFrame）

        Returns:
            dict: 数据一致性报告
        """
        self.logger.info("正在检查数据一致性...")

        # 检查是否为Dask DataFrame
        is_dask = hasattr(data, 'compute')

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持完整的数据一致性检查，将使用Pandas进行检查")
            data = data.compute()

        # 初始化报告
        consistency_report = {
            'data_types': {},
            'range_checks': {},
            'format_checks': {}
        }

        # 检查数据类型
        consistency_report['data_types'] = data.dtypes.to_dict()

        # 范围检查（针对数值型列）
        numeric_columns = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        for col in numeric_columns:
            min_val = data[col].min()
            max_val = data[col].max()
            consistency_report['range_checks'][col] = {
                'min': min_val,
                'max': max_val,
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std()
            }

        # 格式检查（针对日期和文本列）
        if 'release_date' in data.columns:
            # 检查日期格式
            date_format_errors = data['release_date'].apply(
                lambda x: 0 if pd.to_datetime(x, errors='coerce') is pd.NaT else 1
            ).sum()
            consistency_report['format_checks']['release_date'] = {
                'valid_format_count': date_format_errors,
                'invalid_format_count': len(data) - date_format_errors
            }

        if 'budget' in data.columns and 'revenue' in data.columns:
            # 检查预算和收入的合理性
            zero_budget_zero_revenue = ((data['budget'] == 0) & (data['revenue'] == 0)).sum()
            consistency_report['format_checks']['financial_data'] = {
                'zero_budget_zero_revenue': zero_budget_zero_revenue,
                'negative_budget': (data['budget'] < 0).sum(),
                'negative_revenue': (data['revenue'] < 0).sum()
            }

        self.logger.info("数据一致性检查完成")

        return consistency_report

    def clean_data(self, data, remove_duplicates=True, handle_missing=True, remove_invalid=True):
        """
        统一数据清洗

        Args:
            data: 输入数据（Pandas或Dask DataFrame）
            remove_duplicates: 是否移除重复行
            handle_missing: 是否处理缺失值
            remove_invalid: 是否移除无效数据

        Returns:
            清洗后的数据
        """
        self.logger.info("开始统一数据清洗...")

        # 记录初始数据形状
        initial_shape = data.shape
        self.logger.info(f"清洗前数据形状: {initial_shape}")

        # 1. 移除重复行
        if remove_duplicates:
            self.logger.info("移除重复行...")
            data = data.drop_duplicates()

        # 2. 处理缺失值
        if handle_missing:
            self.logger.info("处理缺失值...")

            # 移除完全为空的行
            data = data.dropna(how='all')

            # 对于关键列（ID、标题），移除缺失值
            if 'id' in data.columns:
                data = data[data['id'].notnull() & (data['id'] > 0)]

            if 'title' in data.columns:
                data = data[data['title'].notnull()]

            if 'release_date' in data.columns:
                data = data[data['release_date'].notnull()]

        # 3. 移除无效数据
        if remove_invalid:
            self.logger.info("移除无效数据...")

            # 移除预算和收入都为0的行（如果同时存在）
            if 'budget' in data.columns and 'revenue' in data.columns:
                min_budget = self.config.get('data_cleaning', {}).get('min_budget', 100)
                min_revenue = self.config.get('data_cleaning', {}).get('min_revenue', 100)

                data = data[~((data['budget'] < min_budget) & (data['revenue'] < min_revenue))]

            # 移除无效的评分
            if 'vote_average' in data.columns:
                data = data[(data['vote_average'] >= 0) & (data['vote_average'] <= 10)]

            # 移除无效的运行时间
            if 'runtime' in data.columns:
                data = data[(data['runtime'] > 0) & (data['runtime'] < 500)]

        # 记录清洗后的数据形状
        final_shape = data.shape
        self.logger.info(f"清洗后数据形状: {final_shape}")
        self.logger.info(f"清洗共移除 {initial_shape[0] - final_shape[0]} 行数据")

        return data

    def generate_quality_report(self, data):
        """
        生成完整的数据质量报告

        Args:
            data: 输入数据（Pandas DataFrame）

        Returns:
            dict: 完整的数据质量报告
        """
        self.logger.info("生成数据质量报告...")

        # 检查是否为Dask DataFrame
        is_dask = hasattr(data, 'compute')

        if is_dask:
            self.logger.warning("Dask DataFrame暂不支持完整的数据质量报告生成，将使用Pandas进行报告生成")
            data = data.compute()

        # 生成报告
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'integrity_report': self.check_data_integrity(data),
            'consistency_report': self.check_data_consistency(data)
        }

        # 计算数据质量分数
        total_rows = quality_report['integrity_report']['total_rows']
        missing_score = 1.0 if total_rows == 0 else 1.0 - (quality_report['integrity_report']['total_missing'] / (total_rows * quality_report['integrity_report']['total_columns']))
        duplicate_score = 1.0 if total_rows == 0 else 1.0 - (quality_report['integrity_report']['duplicate_rows'] / total_rows)

        quality_report['quality_score'] = {
            'missing_score': round(missing_score, 4),
            'duplicate_score': round(duplicate_score, 4),
            'overall_score': round((missing_score + duplicate_score) / 2, 4)
        }

        self.logger.info(f"数据质量报告生成完成，总体质量分数: {quality_report['quality_score']['overall_score']}")

        # 保存到类变量
        self.quality_report = quality_report

        return quality_report

    def save_quality_report(self, report_path="data_quality_report.csv"):
        """
        保存数据质量报告

        Args:
            report_path: 报告保存路径

        Returns:
            bool: 保存是否成功
        """
        if not self.quality_report:
            self.logger.error("没有可用的数据质量报告，请先调用generate_quality_report")
            return False

        try:
            # 保存缺失值报告
            missing_df = pd.DataFrame.from_dict(self.quality_report['integrity_report']['missing_values'], orient='index', columns=['missing_values'])
            missing_df['percentage'] = (missing_df['missing_values'] / self.quality_report['integrity_report']['total_rows']) * 100

            # 保存范围检查报告
            range_df = pd.DataFrame.from_dict(self.quality_report['consistency_report']['range_checks'], orient='index')

            # 保存到CSV
            missing_df.to_csv(os.path.join(os.path.dirname(report_path), "missing_values_report.csv"))
            range_df.to_csv(os.path.join(os.path.dirname(report_path), "range_checks_report.csv"))

            # 保存总体报告
            with open(report_path, 'w') as f:
                f.write("# 数据质量报告\n")
                f.write(f"生成时间: {self.quality_report['timestamp']}\n")
                f.write(f"总体质量分数: {self.quality_report['quality_score']['overall_score']}\n")
                f.write(f"总行数: {self.quality_report['integrity_report']['total_rows']}\n")
                f.write(f"总缺失值: {self.quality_report['integrity_report']['total_missing']}\n")
                f.write(f"重复行: {self.quality_report['integrity_report']['duplicate_rows']}\n")

            self.logger.info(f"数据质量报告已保存到: {report_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存数据质量报告失败: {e}")
            return False

    def validate_and_clean_data(self, data, generate_report=True, report_path=None):
        """
        完整的数据验证和清洗流程

        Args:
            data: 输入数据（Pandas或Dask DataFrame）
            generate_report: 是否生成数据质量报告
            report_path: 报告保存路径

        Returns:
            tuple: (清洗后的数据, 数据质量报告)
        """
        self.logger.info("开始完整的数据验证和清洗流程...")

        # 检查是否为Dask DataFrame
        is_dask = hasattr(data, 'compute')

        # 1. 生成数据质量报告
        quality_report = None
        if generate_report:
            if is_dask:
                sample_data = data.head(10000).compute()  # 使用样本数据生成报告
                quality_report = self.generate_quality_report(sample_data)
            else:
                quality_report = self.generate_quality_report(data)

            # 保存报告
            if report_path:
                self.save_quality_report(report_path)

        # 2. 清洗数据
        cleaned_data = self.clean_data(data)

        self.logger.info("数据验证和清洗流程完成")

        return cleaned_data, quality_report


def main():
    """主函数，用于测试数据验证模块"""
    # 初始化日志
    logger = get_logger('data_validation_main')

    # 示例用法
    logger.info("测试数据验证模块...")

    # 创建测试数据
    test_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5],
        'title': ['电影1', '电影2', None, '电影4', '电影5', '电影5'],
        'budget': [1000000, 2000000, 3000000, 0, 5000000, 5000000],
        'revenue': [5000000, 10000000, 15000000, 0, 25000000, 25000000],
        'release_date': ['2020-01-01', '2020-02-02', '2020-03-03', '2020-04-04', '2020-05-05', '2020-05-05'],
        'vote_average': [8.5, 7.8, 6.5, 9.0, 5.5, 5.5]
    })

    # 初始化验证器
    validator = DataValidator()

    # 验证和清洗数据
    cleaned_data, quality_report = validator.validate_and_clean_data(test_data, generate_report=True)

    # 打印结果
    logger.info("原始数据:")
    logger.info(f"形状: {test_data.shape}")
    logger.info(test_data)

    logger.info("清洗后数据:")
    logger.info(f"形状: {cleaned_data.shape}")
    logger.info(cleaned_data)

    logger.info("数据质量报告:")
    logger.info(quality_report)

    logger.info("数据验证模块测试完成")


if __name__ == "__main__":
    main()
