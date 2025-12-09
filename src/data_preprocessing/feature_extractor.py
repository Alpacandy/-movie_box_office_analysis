# 特征提取模块
"""
负责从数据中提取和生成特征，包括JSON特征提取、财务指标计算、分类特征编码等。
"""

import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    logging.info("Dask未安装，将使用Pandas进行处理")
    DASK_AVAILABLE = False


class FeatureExtractor:
    """特征提取器，负责从数据中提取和生成特征"""

    def __init__(self, config=None):
        """初始化特征提取器

        Args:
            config: 配置字典
        """
        self.logger = logging.getLogger(__name__)
        self.config = config if config else {}

        # 从配置中获取特征提取参数
        self.enable_json_features = self.config.get('feature_extraction', {}).get('extract_json_features', True)
        self.enable_datetime_features = self.config.get('feature_extraction', {}).get('extract_datetime_features', True)
        self.enable_financial_metrics = self.config.get('feature_extraction', {}).get('calculate_financial_metrics', True)

    def extract_json_features(self, data, filename=None):
        """提取JSON格式或逗号分隔格式的特征

        Args:
            data: 输入数据
            filename: 数据文件名，用于判断数据格式

        Returns:
            包含提取特征的数据
        """
        self.logger.info("正在提取JSON格式特征...")
        self.logger.info(f"extract_json_features输入数据类型: {type(data)}")

        # 严格检查是否为DataFrame，如果不是则直接返回
        if not hasattr(data, 'columns'):
            self.logger.error(f"extract_json_features输入数据不是DataFrame: {type(data)}")
            return data

        # 检查是否为Dask DataFrame（安全检查，避免Dask未安装时的错误）
        is_dask_dataframe = False
        if DASK_AVAILABLE and isinstance(data, dd.DataFrame):
            is_dask_dataframe = True
            self.logger.info("输入是Dask DataFrame")
        else:
            self.logger.info("输入是Pandas DataFrame或其他DataFrame类型")

        # 1. 提取类型特征
        self.logger.info("正在提取类型特征...")
        # 最严格的检查：确保data是DataFrame并且具有genres列
        if hasattr(data, 'columns') and 'genres' in data.columns:
            self.logger.info(f"提取类型特征前数据类型: {type(data)}")
            try:
                # 只对Dask DataFrame使用Dask处理，并且确保data是dd.DataFrame实例
                if is_dask_dataframe and DASK_AVAILABLE:
                    self.logger.info("使用Dask处理类型特征")
                    # 再次检查data是否为Dask DataFrame
                    if isinstance(data, dd.DataFrame):
                        # 创建一个样本数据点来推断meta
                        sample_data = data.head(1).compute()
                        self.logger.info(f"样本数据类型: {type(sample_data)}")
                        if hasattr(sample_data, 'columns'):
                            sample_df = self._extract_genres_pandas(sample_data)
                            self.logger.info(f"样本DataFrame类型: {type(sample_df)}")
                            if hasattr(sample_df, 'columns'):
                                data = data.map_partitions(self._extract_genres_dask, meta=sample_df)
                                self.logger.info(f"Dask类型特征处理后数据类型: {type(data)}")
                else:
                    # 对所有非Dask DataFrame使用Pandas处理，确保data是DataFrame
                    if hasattr(data, 'columns'):
                        self.logger.info("使用Pandas处理类型特征")
                        data = self._extract_genres_pandas(data)
                        self.logger.info(f"Pandas类型特征处理后数据类型: {type(data)}")
            except Exception as e:
                self.logger.error(f"处理类型特征时出错: {e}")
                # 如果出错，尝试使用Pandas处理，确保data是DataFrame
                if hasattr(data, 'columns'):
                    self.logger.info("出错后尝试使用Pandas处理类型特征")
                    data = self._extract_genres_pandas(data)

        # 2. 提取制作公司特征
        self.logger.info("正在提取制作公司特征...")
        if hasattr(data, 'columns') and 'production_companies' in data.columns:
            self.logger.info(f"提取制作公司特征前数据类型: {type(data)}")
            try:
                # 只对Dask DataFrame使用Dask处理
                if is_dask_dataframe and DASK_AVAILABLE:
                    self.logger.info("使用Dask处理制作公司特征")
                    sample_data = data.head(1).compute()
                    if hasattr(sample_data, 'columns'):
                        sample_df = self._extract_production_companies_pandas(sample_data)
                        if hasattr(sample_df, 'columns'):
                            data = data.map_partitions(self._extract_production_companies_dask, meta=sample_df)
                else:
                    # 对所有非Dask DataFrame使用Pandas处理
                    self.logger.info("使用Pandas处理制作公司特征")
                    data = self._extract_production_companies_pandas(data)
            except Exception as e:
                self.logger.error(f"处理制作公司特征时出错: {e}")
                # 如果出错，尝试使用Pandas处理
                if hasattr(data, 'columns'):
                    self.logger.info("出错后尝试使用Pandas处理制作公司特征")
                    data = self._extract_production_companies_pandas(data)

        # 3. 提取制作国家特征
        self.logger.info("正在提取制作国家特征...")
        if hasattr(data, 'columns') and 'production_countries' in data.columns:
            self.logger.info(f"提取制作国家特征前数据类型: {type(data)}")
            try:
                # 只对Dask DataFrame使用Dask处理
                if is_dask_dataframe and DASK_AVAILABLE:
                    self.logger.info("使用Dask处理制作国家特征")
                    sample_data = data.head(1).compute()
                    if hasattr(sample_data, 'columns'):
                        sample_df = self._extract_production_countries_pandas(sample_data)
                        if hasattr(sample_df, 'columns'):
                            data = data.map_partitions(self._extract_production_countries_dask, meta=sample_df)
                else:
                    # 对所有非Dask DataFrame使用Pandas处理
                    self.logger.info("使用Pandas处理制作国家特征")
                    data = self._extract_production_countries_pandas(data)
            except Exception as e:
                self.logger.error(f"处理制作国家特征时出错: {e}")
                # 如果出错，尝试使用Pandas处理
                if hasattr(data, 'columns'):
                    self.logger.info("出错后尝试使用Pandas处理制作国家特征")
                    data = self._extract_production_countries_pandas(data)

        # 4. 提取语言特征
        self.logger.info("正在提取语言特征...")
        if hasattr(data, 'columns') and 'spoken_languages' in data.columns:
            self.logger.info(f"提取语言特征前数据类型: {type(data)}")
            try:
                # 只对Dask DataFrame使用Dask处理
                if is_dask_dataframe and DASK_AVAILABLE:
                    self.logger.info("使用Dask处理语言特征")
                    sample_data = data.head(1).compute()
                    if hasattr(sample_data, 'columns'):
                        sample_df = self._extract_spoken_languages_pandas(sample_data)
                        if hasattr(sample_df, 'columns'):
                            data = data.map_partitions(self._extract_spoken_languages_dask, meta=sample_df)
                else:
                    # 对所有非Dask DataFrame使用Pandas处理
                    self.logger.info("使用Pandas处理语言特征")
                    data = self._extract_spoken_languages_pandas(data)
            except Exception as e:
                self.logger.error(f"处理语言特征时出错: {e}")
                # 如果出错，尝试使用Pandas处理
                if hasattr(data, 'columns'):
                    self.logger.info("出错后尝试使用Pandas处理语言特征")
                    data = self._extract_spoken_languages_pandas(data)

        # 5. 提取关键词特征
        self.logger.info("正在提取关键词特征...")
        if hasattr(data, 'columns') and 'keywords' in data.columns:
            self.logger.info(f"提取关键词特征前数据类型: {type(data)}")
            try:
                # 只对Dask DataFrame使用Dask处理
                if is_dask_dataframe and DASK_AVAILABLE:
                    self.logger.info("使用Dask处理关键词特征")
                    # 创建一个样本数据点来推断meta
                    sample_data = data.head(1).compute()
                    self.logger.info(f"样本数据类型: {type(sample_data)}")
                    if hasattr(sample_data, 'columns'):
                        sample_df = self._extract_keywords_pandas(sample_data)
                        self.logger.info(f"样本DataFrame类型: {type(sample_df)}")
                        if hasattr(sample_df, 'columns'):
                            data = data.map_partitions(self._extract_keywords_dask, meta=sample_df)
                            self.logger.info(f"Dask关键词特征处理后数据类型: {type(data)}")
                else:
                    # 对所有非Dask DataFrame使用Pandas处理
                    self.logger.info("使用Pandas处理关键词特征")
                    data = self._extract_keywords_pandas(data)
                    self.logger.info(f"Pandas关键词特征处理后数据类型: {type(data)}")
            except Exception as e:
                self.logger.error(f"处理关键词特征时出错: {e}")
                # 如果出错，尝试使用Pandas处理
                if hasattr(data, 'columns'):
                    self.logger.info("出错后尝试使用Pandas处理关键词特征")
                    data = self._extract_keywords_pandas(data)

        return data

    def _extract_genres_pandas(self, data):
        """从Pandas DataFrame中提取类型特征"""
        # 严格检查输入是否为DataFrame
        if not hasattr(data, 'columns'):
            self.logger.error(f"_extract_genres_pandas输入不是DataFrame: {type(data)}")
            return data

        # 确保genres列存在
        if 'genres' not in data.columns:
            self.logger.error("_extract_genres_pandas数据中没有'genres'列")
            return data

        # 检查genres列的类型
        if data['genres'].dtype == 'object':
            # 检查是否为JSON格式
            try:
                # 安全地获取第一个非空值
                first_valid = data['genres'].dropna().iloc[0] if not data['genres'].dropna().empty else ''
                if first_valid.startswith('['):
                    # JSON格式
                    data['genres_list'] = data['genres'].apply(lambda x: [genre['name'] for genre in json.loads(x)] if x and isinstance(x, str) else [])
                else:
                    # 假设是逗号分隔格式
                    data['genres_list'] = data['genres'].apply(lambda x: x.split(',') if x and isinstance(x, str) else [])
            except Exception as e:
                self.logger.error(f"处理genres列时出错: {e}")
                # 如果出错，创建默认列（使用Pandas Series）
                data['genres_list'] = pd.Series([[]] * len(data), index=data.index)
        else:
            # 如果genres列不是object类型，创建默认列（使用Pandas Series）
            data['genres_list'] = pd.Series([[]] * len(data), index=data.index)

        # 提取主类型
        try:
            data['main_genre'] = data['genres_list'].apply(lambda x: x[0] if x else 'Unknown')
        except Exception as e:
            self.logger.error(f"提取主类型时出错: {e}")
            data['main_genre'] = 'Unknown'

        # 计算类型数量
        try:
            data['genre_count'] = data['genres_list'].apply(len)
        except Exception as e:
            self.logger.error(f"计算类型数量时出错: {e}")
            data['genre_count'] = 0

        return data

    def _extract_genres_dask(self, data):
        """从Dask DataFrame中提取类型特征"""
        return self._extract_genres_pandas(data)

    def _extract_production_companies_pandas(self, data):
        """从Pandas DataFrame中提取制作公司特征"""
        # 严格检查输入是否为DataFrame
        if not hasattr(data, 'columns'):
            self.logger.error(f"_extract_production_companies_pandas输入不是DataFrame: {type(data)}")
            return data

        # 确保production_companies列存在
        if 'production_companies' not in data.columns:
            self.logger.error("_extract_production_companies_pandas数据中没有'production_companies'列")
            return data

        # 检查production_companies列的类型
        if data['production_companies'].dtype == 'object':
            # 检查是否为JSON格式
            try:
                # 安全地获取第一个非空值
                first_valid = data['production_companies'].dropna().iloc[0] if not data['production_companies'].dropna().empty else ''
                if first_valid.startswith('['):
                    # JSON格式
                    data['production_companies_list'] = data['production_companies'].apply(lambda x: [company['name'] for company in json.loads(x)] if x and isinstance(x, str) else [])
                else:
                    # 假设是逗号分隔格式
                    data['production_companies_list'] = data['production_companies'].apply(lambda x: x.split(',') if x and isinstance(x, str) else [])
            except Exception as e:
                self.logger.error(f"处理production_companies列时出错: {e}")
                # 如果出错，创建默认列（使用Pandas Series）
                data['production_companies_list'] = pd.Series([[]] * len(data), index=data.index)
        else:
            # 如果production_companies列不是object类型，创建默认列（使用Pandas Series）
            data['production_companies_list'] = pd.Series([[]] * len(data), index=data.index)

        # 提取主制作公司
        try:
            data['main_production_company'] = data['production_companies_list'].apply(lambda x: x[0] if x else 'Unknown')
        except Exception as e:
            self.logger.error(f"提取主制作公司时出错: {e}")
            data['main_production_company'] = pd.Series(['Unknown'] * len(data), index=data.index)

        # 计算制作公司数量
        try:
            data['production_company_count'] = data['production_companies_list'].apply(len)
        except Exception as e:
            self.logger.error(f"计算制作公司数量时出错: {e}")
            data['production_company_count'] = pd.Series([0] * len(data), index=data.index)

        return data

    def _extract_production_companies_dask(self, data):
        """从Dask DataFrame中提取制作公司特征"""
        return self._extract_production_companies_pandas(data)

    def _extract_production_countries_pandas(self, data):
        """从Pandas DataFrame中提取制作国家特征"""
        # 严格检查输入是否为DataFrame
        if not hasattr(data, 'columns'):
            self.logger.error(f"_extract_production_countries_pandas输入不是DataFrame: {type(data)}")
            return data

        # 确保production_countries列存在
        if 'production_countries' not in data.columns:
            self.logger.error("_extract_production_countries_pandas数据中没有'production_countries'列")
            return data

        # 检查production_countries列的类型
        if data['production_countries'].dtype == 'object':
            # 检查是否为JSON格式
            try:
                # 安全地获取第一个非空值
                first_valid = data['production_countries'].dropna().iloc[0] if not data['production_countries'].dropna().empty else ''
                if first_valid.startswith('['):
                    # JSON格式
                    data['production_countries_list'] = data['production_countries'].apply(lambda x: [country['name'] for country in json.loads(x)] if x and isinstance(x, str) else [])
                else:
                    # 假设是逗号分隔格式
                    data['production_countries_list'] = data['production_countries'].apply(lambda x: x.split(',') if x and isinstance(x, str) else [])
            except Exception as e:
                self.logger.error(f"处理production_countries列时出错: {e}")
                # 如果出错，创建默认列（使用Pandas Series）
                data['production_countries_list'] = pd.Series([[]] * len(data), index=data.index)
        else:
            # 如果production_countries列不是object类型，创建默认列（使用Pandas Series）
            data['production_countries_list'] = pd.Series([[]] * len(data), index=data.index)

        # 提取主制作国家
        try:
            data['main_production_country'] = data['production_countries_list'].apply(lambda x: x[0] if x else 'Unknown')
        except Exception as e:
            self.logger.error(f"提取主制作国家时出错: {e}")
            data['main_production_country'] = pd.Series(['Unknown'] * len(data), index=data.index)

        # 计算制作国家数量
        try:
            data['production_country_count'] = data['production_countries_list'].apply(len)
        except Exception as e:
            self.logger.error(f"计算制作国家数量时出错: {e}")
            data['production_country_count'] = pd.Series([0] * len(data), index=data.index)

        return data

    def _extract_production_countries_dask(self, data):
        """从Dask DataFrame中提取制作国家特征"""
        return self._extract_production_countries_pandas(data)

    def _extract_spoken_languages_pandas(self, data):
        """从Pandas DataFrame中提取语言特征"""
        # 严格检查输入是否为DataFrame
        if not hasattr(data, 'columns'):
            self.logger.error(f"_extract_spoken_languages_pandas输入不是DataFrame: {type(data)}")
            return data

        # 确保spoken_languages列存在
        if 'spoken_languages' not in data.columns:
            self.logger.error("_extract_spoken_languages_pandas数据中没有'spoken_languages'列")
            return data

        # 检查spoken_languages列的类型
        if data['spoken_languages'].dtype == 'object':
            # 检查是否为JSON格式
            try:
                # 安全地获取第一个非空值
                first_valid = data['spoken_languages'].dropna().iloc[0] if not data['spoken_languages'].dropna().empty else ''
                if first_valid.startswith('['):
                    # JSON格式
                    data['spoken_languages_list'] = data['spoken_languages'].apply(lambda x: [lang['name'] for lang in json.loads(x)] if x and isinstance(x, str) else [])
                else:
                    # 假设是逗号分隔格式
                    data['spoken_languages_list'] = data['spoken_languages'].apply(lambda x: x.split(',') if x and isinstance(x, str) else [])
            except Exception as e:
                self.logger.error(f"处理spoken_languages列时出错: {e}")
                # 如果出错，创建默认列（使用Pandas Series）
                data['spoken_languages_list'] = pd.Series([[]] * len(data), index=data.index)
        else:
            # 如果spoken_languages列不是object类型，创建默认列（使用Pandas Series）
            data['spoken_languages_list'] = pd.Series([[]] * len(data), index=data.index)

        # 提取主语言
        try:
            data['main_spoken_language'] = data['spoken_languages_list'].apply(lambda x: x[0] if x else 'Unknown')
        except Exception as e:
            self.logger.error(f"提取主语言时出错: {e}")
            data['main_spoken_language'] = pd.Series(['Unknown'] * len(data), index=data.index)

        # 计算语言数量
        try:
            data['spoken_language_count'] = data['spoken_languages_list'].apply(len)
        except Exception as e:
            self.logger.error(f"计算语言数量时出错: {e}")
            data['spoken_language_count'] = pd.Series([0] * len(data), index=data.index)

        return data

    def _extract_spoken_languages_dask(self, data):
        """从Dask DataFrame中提取语言特征"""
        return self._extract_spoken_languages_pandas(data)

    def _extract_keywords_pandas(self, data):
        """从Pandas DataFrame中提取关键词特征"""
        # 严格检查输入是否为DataFrame
        if not hasattr(data, 'columns'):
            self.logger.error(f"_extract_keywords_pandas输入不是DataFrame: {type(data)}")
            return data

        # 确保keywords列存在
        if 'keywords' not in data.columns:
            self.logger.error("_extract_keywords_pandas数据中没有'keywords'列")
            return data

        # 检查keywords列的类型
        if data['keywords'].dtype == 'object':
            # 检查是否为JSON格式
            try:
                # 安全地获取第一个非空值
                first_valid = data['keywords'].dropna().iloc[0] if not data['keywords'].dropna().empty else ''
                if first_valid.startswith('['):
                    # JSON格式
                    data['keywords_list'] = data['keywords'].apply(lambda x: [keyword['name'] for keyword in json.loads(x)] if x and isinstance(x, str) else [])
                else:
                    # 假设是逗号分隔格式
                    data['keywords_list'] = data['keywords'].apply(lambda x: x.split(',') if x and isinstance(x, str) else [])
            except Exception as e:
                self.logger.error(f"处理keywords列时出错: {e}")
                # 如果出错，创建默认列（使用Pandas Series）
                data['keywords_list'] = pd.Series([[]] * len(data), index=data.index)
        else:
            # 如果keywords列不是object类型，创建默认列（使用Pandas Series）
            data['keywords_list'] = pd.Series([[]] * len(data), index=data.index)

        # 计算关键词数量
        try:
            data['keyword_count'] = data['keywords_list'].apply(len)
        except Exception as e:
            self.logger.error(f"计算关键词数量时出错: {e}")
            data['keyword_count'] = pd.Series([0] * len(data), index=data.index)

        return data

    def _extract_keywords_dask(self, data):
        """从Dask DataFrame中提取关键词特征"""
        return self._extract_keywords_pandas(data)

    def calculate_financial_metrics(self, data):
        """计算财务指标

        Args:
            data: 输入数据

        Returns:
            包含财务指标的数据
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在计算财务指标...")

        # 检查必要的财务列是否存在
        has_financial_data = 'revenue' in data.columns and 'budget' in data.columns

        if has_financial_data:
            # 1. 计算投资回报率
            self.logger.info("正在计算投资回报率...")
            if is_dask:
                # Dask DataFrame
                data = data.map_partitions(self._calculate_roi_dask)
            else:
                # Pandas DataFrame
                data = self._calculate_roi_pandas(data)

            # 2. 计算利润
            self.logger.info("正在计算利润...")
            data['profit'] = data['revenue'] - data['budget']

            # 3. 计算每美元预算产生的收入
            self.logger.info("正在计算每美元预算产生的收入...")
            if is_dask:
                data = data.map_partitions(self._calculate_revenue_per_dollar_dask)
            else:
                data = self._calculate_revenue_per_dollar_pandas(data)

            # 4. 估算观影人数（假设平均票价为10美元）
            self.logger.info("正在估算观影人数...")
            data['estimated_tickets'] = data['revenue'] / 10
        else:
            self.logger.warning("缺少必要的财务数据列(revenue或budget)，跳过财务指标计算")

        # 3. 计算电影年龄（当前年份 - 发布年份）
        self.logger.info("正在计算电影年龄...")
        if 'release_year' in data.columns:
            current_year = pd.Timestamp.now().year
            data['movie_age'] = current_year - data['release_year']

        return data

    def _calculate_roi_pandas(self, data):
        """从Pandas DataFrame中计算投资回报率"""
        # 避免除以0
        data['return_on_investment'] = np.where(
            data['budget'] > 0, 
            (data['revenue'] - data['budget']) / data['budget'], 
            0
        )
        return data

    def _calculate_roi_dask(self, data):
        """从Dask DataFrame中计算投资回报率"""
        return self._calculate_roi_pandas(data)

    def _calculate_revenue_per_dollar_pandas(self, data):
        """从Pandas DataFrame中计算每美元预算产生的收入"""
        # 避免除以0
        data['revenue_per_dollar'] = np.where(
            data['budget'] > 0, 
            data['revenue'] / data['budget'], 
            0
        )
        return data

    def _calculate_revenue_per_dollar_dask(self, data):
        """从Dask DataFrame中计算每美元预算产生的收入"""
        return self._calculate_revenue_per_dollar_pandas(data)

    def encode_categorical_features(self, data):
        """编码分类特征

        Args:
            data: 输入数据

        Returns:
            包含编码特征的数据
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在编码分类特征...")

        # 1. 主要制作公司标识
        self.logger.info("正在创建主要制作公司标识特征...")
        if hasattr(data, 'columns') and 'production_companies_list' in data.columns:
            major_studios = ['Warner Bros. Pictures', 'Universal Pictures', 'Paramount Pictures', 
                           '20th Century Fox', 'Walt Disney Pictures', 'Sony Pictures', 
                           'Columbia Pictures', 'Lionsgate', 'DreamWorks SKG']

            if is_dask:
                # Dask DataFrame
                data['has_major_studio'] = data['production_companies_list'].apply(
                    lambda x: 1 if any(studio in x for studio in major_studios) else 0, 
                    meta=('has_major_studio', int)
                )
            else:
                # Pandas DataFrame
                data['has_major_studio'] = data['production_companies_list'].apply(
                    lambda x: 1 if any(studio in x for studio in major_studios) else 0
                )

        # 2. 编码主类型特征
        self.logger.info("正在编码主类型特征...")
        if 'main_genre' in data.columns:
            if not is_dask:
                # Pandas DataFrame
                from sklearn.preprocessing import LabelEncoder
                le_genre = LabelEncoder()
                data['main_genre_encoded'] = le_genre.fit_transform(data['main_genre'])

            # 创建类型独热编码
            if not any(col.startswith('genre_') for col in data.columns):
                if is_dask:
                    # Dask DataFrame
                    data = data.map_partitions(self._create_genre_dummies_dask)
                else:
                    # Pandas DataFrame
                    data = self._create_genre_dummies_pandas(data)

        # 2. 编码主制作国家特征
        self.logger.info("正在编码主制作国家特征...")
        if 'main_production_country' in data.columns:
            if not is_dask:
                # Pandas DataFrame
                from sklearn.preprocessing import LabelEncoder
                le_country = LabelEncoder()
                data['main_production_country_encoded'] = le_country.fit_transform(data['main_production_country'])

        # 3. 编码主语言特征
        self.logger.info("正在编码主语言特征...")
        if 'main_spoken_language' in data.columns:
            if not is_dask:
                # Pandas DataFrame
                from sklearn.preprocessing import LabelEncoder
                le_language = LabelEncoder()
                data['main_spoken_language_encoded'] = le_language.fit_transform(data['main_spoken_language'])

        return data

    def _create_genre_dummies_pandas(self, data):
        """从Pandas DataFrame中创建类型独热编码"""
        # 获取最常见的10种类型
        top_genres = data['main_genre'].value_counts().head(10).index

        # 为每种类型创建独热编码
        for genre in top_genres:
            data[f'genre_{genre.replace(" ", "_").lower()}'] = (data['main_genre'] == genre).astype(int)

        return data

    def _create_genre_dummies_dask(self, data):
        """从Dask DataFrame中创建类型独热编码"""
        return self._create_genre_dummies_pandas(data)

    def convert_datetime(self, data):
        """转换日期时间特征

        Args:
            data: 输入数据

        Returns:
            包含转换日期特征的数据
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在转换日期时间特征...")

        if hasattr(data, 'columns') and 'release_date' in data.columns:
            if is_dask:
                # Dask DataFrame
                # 转换日期
                data['release_date'] = data['release_date'].apply(pd.to_datetime, errors='coerce', meta=('release_date', 'datetime64[ns]'))

                # 提取年、月、日、星期
                data['release_year'] = data['release_date'].dt.year
                data['release_month'] = data['release_date'].dt.month
                data['release_day'] = data['release_date'].dt.day
                data['release_quarter'] = data['release_date'].dt.quarter
                data['release_dayofweek'] = data['release_date'].dt.weekday  # 0=周一, 6=周日
            else:
                # Pandas DataFrame
                data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

                # 提取年、月、日、星期
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

    def create_additional_features(self, data):
        """创建额外的特征

        Args:
            data: 输入数据

        Returns:
            包含额外特征的数据
        """
        is_dask = hasattr(data, 'compute')
        self.logger.info("正在创建额外的特征...")

        # 1. 创建评分和票数的交互特征
        if 'vote_average' in data.columns and 'vote_count' in data.columns:
            self.logger.info("正在创建评分和票数的交互特征...")
            data['weighted_rating'] = data['vote_average'] * data['vote_count']

        # 2. 创建运行时间相关特征
        if 'runtime' in data.columns:
            self.logger.info("正在创建运行时间相关特征...")
            data['runtime_category'] = pd.cut(
                data['runtime'], 
                bins=[0, 90, 120, 150, np.inf], 
                labels=['Short', 'Medium', 'Long', 'Very Long'],
                include_lowest=True
            )

        return data

    def extract_features(self, data):
        """统一的特征提取方法，整合所有特征提取功能

        Args:
            data: 输入数据

        Returns:
            包含所有提取特征的数据
        """
        self.logger.info("开始统一特征提取...")

        # 1. 提取JSON特征（如果启用）
        if self.enable_json_features:
            data = self.extract_json_features(data)

        # 2. 转换日期时间特征（如果启用）
        if self.enable_datetime_features:
            data = self.convert_datetime(data)

        # 3. 创建额外特征
        data = self.create_additional_features(data)

        self.logger.info("特征提取完成")
        return data
