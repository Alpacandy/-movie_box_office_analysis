import os
import sys
import string
import warnings
# 导入第三方库
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import BertTokenizer, BertModel
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

# 设置matplotlib为非交互式后端，避免线程问题
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，适合非交互式环境

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入本地模块
from src.utils.logging_config import get_logger

# 尝试导入tqdm，如不可用则使用简单的进度显示
try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    # 定义一个简单的替代函数
    def tqdm(iterable, *args, **kwargs):
        return iterable
    tqdm_available = False


warnings.filterwarnings('ignore')

# 初始化日志记录器
logger = get_logger('feature_engineering')

# 全局变量，标记情感分析是否可用
sentiment_available = False


class FeatureEngineering:
    def __init__(self, base_dir=None):
        # 使用项目根目录作为基础路径
        self.base_dir = os.path.join(project_root, "data") if base_dir is None else base_dir
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.logger = get_logger('feature_engineering.FeatureEngineering')

        # 尝试初始化情感分析器，处理下载失败情况
        try:
            # 尝试直接使用，如果已安装就不会报错
            self.sia = SentimentIntensityAnalyzer()
            self.sentiment_available = True
        except LookupError:
            try:
                # 尝试下载
                self.logger.info("正在下载vader_lexicon...")
                nltk.download('vader_lexicon', quiet=True)
                self.sia = SentimentIntensityAnalyzer()
                self.sentiment_available = True
            except Exception as e:
                self.logger.warning(f"无法下载或初始化vader_lexicon: {e}")
                self.logger.warning("情感分析功能将不可用")
                self.sia = None
                self.sentiment_available = False

        # 更新全局变量
        global sentiment_available
        sentiment_available = self.sentiment_available

    def load_data(self, filename="cleaned_movie_data.csv"):
        """加载处理后的数据"""
        file_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"文件不存在: {file_path}")
            self.logger.error("请先运行数据预处理脚本: python data_preprocessing.py")
            return None

        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            self.logger.error(f"错误: 加载数据失败: {e}")
            self.logger.error("请检查文件格式是否正确")
            return None

    def encode_categorical_features(self, data):
        """编码分类特征"""
        self.logger.info("正在编码分类特征...")

        # 1. 导演编码（保留前100位导演，其他标记为'Other'）
        top_directors = data['director'].value_counts().head(100).index
        data['director_encoded'] = data['director'].apply(lambda x: x if x in top_directors else 'Other')

        # 2. 标签编码
        le_director = LabelEncoder()
        data['director_encoded'] = le_director.fit_transform(data['director_encoded'])

        le_genre = LabelEncoder()
        data['main_genre_encoded'] = le_genre.fit_transform(data['main_genre'])

        # 3. 类型独热编码（如果尚未进行）
        if 'genre_action' not in data.columns:
            genre_dummies = pd.get_dummies(data['main_genre'], prefix='genre', drop_first=True)
            data = pd.concat([data, genre_dummies], axis=1)

        return data

    def calculate_director_influence(self, data):
        """计算导演影响力特征"""
        self.logger.info("正在计算导演影响力特征...")

        # 1. 导演平均票房
        director_stats = data.groupby('director').agg({
            'revenue': ['mean', 'median', 'std'],
            'return_on_investment': 'mean',
            'budget': 'mean',
            'vote_average': 'mean',
            'id': 'count'
        }).reset_index()
        director_stats.columns = ['director', 'director_avg_revenue', 'director_median_revenue',
                                  'director_revenue_std', 'director_avg_roi', 'director_avg_budget',
                                  'director_avg_rating', 'director_movie_count']

        # 2. 合并导演统计信息
        data = data.merge(director_stats, on='director', how='left')

        # 3. 导演经验年数（首次和最近作品）
        first_work = data.groupby('director')['release_year'].min().reset_index()
        first_work.columns = ['director', 'director_first_work']
        data = data.merge(first_work, on='director', how='left')
        data['director_experience'] = data['release_year'] - data['director_first_work']

        return data

    def calculate_actor_influence(self, data):
        """计算演员影响力特征"""
        self.logger.info("正在计算演员影响力特征...")

        # 1. 提取演员列表，使用更高效的方式处理JSON字符串
        if not isinstance(data['top_3_actors'].iloc[0], list):
            data['top_3_actors'] = data['top_3_actors'].apply(eval)

        # 2. 计算每个演员的平均票房 - 使用explode替代逐行迭代
        actor_df = data.explode('top_3_actors')
        actor_avg_revenue = actor_df.groupby('top_3_actors')['revenue'].mean().reset_index()
        actor_avg_revenue.columns = ['actor', 'actor_avg_revenue']

        # 3. 计算电影中演员的平均影响力 - 使用向量化操作
        # 创建演员到平均票房的映射字典
        actor_revenue_dict = actor_avg_revenue.set_index('actor')['actor_avg_revenue'].to_dict()

        # 使用向量化的方式计算平均演员影响力
        def get_avg_actor_revenue(actors):
            actor_revenues = [actor_revenue_dict.get(actor, 0) for actor in actors]
            return np.mean(actor_revenues) if actor_revenues else 0

        # 使用pandas的apply函数，比iterrows快得多
        data['avg_actor_revenue'] = data['top_3_actors'].apply(get_avg_actor_revenue)

        # 4. 演员数量特征 - 向量化操作
        data['top_actor_count'] = data['top_3_actors'].apply(len)

        return data

    def preprocess_text(self, text):
        """简化的文本预处理函数"""
        if pd.isna(text):
            return ""

        # 转换为小写
        text = text.lower()

        # 移除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 简单分词（使用空格）
        tokens = text.split()

        return ' '.join(tokens)

    def text_features(self, data):
        """增强的文本特征生成"""
        self.logger.info("正在生成增强文本特征...")

        # 确保overview列存在
        if 'overview' not in data.columns:
            self.logger.warning("警告: overview列不存在，跳过文本特征生成")
            return data

        # 1. 基本文本统计特征
        data['overview_length'] = data['overview'].str.len()
        data['overview_word_count'] = data['overview'].str.split().str.len()
        data['overview_avg_word_length'] = data['overview'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).strip() else 0
        )
        # 使用简单的句子计数（按句号、问号、感叹号分割）
        data['overview_sentence_count'] = data['overview'].apply(
            lambda x: len([sent for sent in str(x).split('.') if sent.strip()]) if str(x).strip() else 0
        )

        # 2. 简单情感分析（仅使用VADER，如果可用）
        try:
            if hasattr(self, 'sentiment_available') and self.sentiment_available:
                def get_vader_sentiment(text):
                    if pd.isna(text):
                        return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
                    return self.sia.polarity_scores(text)

                sentiment_results = data['overview'].apply(get_vader_sentiment)
                data['overview_sentiment_neg'] = sentiment_results.apply(lambda x: x['neg'])
                data['overview_sentiment_neu'] = sentiment_results.apply(lambda x: x['neu'])
                data['overview_sentiment_pos'] = sentiment_results.apply(lambda x: x['pos'])
                data['overview_sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
            else:
                # 初始化VADER情感分析器
                try:
                    self.sia = SentimentIntensityAnalyzer()
                    self.sentiment_available = True
                    # 应用VADER情感分析

                    def get_vader_sentiment(text):
                        if pd.isna(text):
                            return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
                        return self.sia.polarity_scores(text)

                    sentiment_results = data['overview'].apply(get_vader_sentiment)
                    data['overview_sentiment_neg'] = sentiment_results.apply(lambda x: x['neg'])
                    data['overview_sentiment_neu'] = sentiment_results.apply(lambda x: x['neu'])
                    data['overview_sentiment_pos'] = sentiment_results.apply(lambda x: x['pos'])
                    data['overview_sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
                except Exception as e:
                    self.logger.warning(f"警告: 无法初始化情感分析器: {e}")
                    # 添加默认情感特征
                    data['overview_sentiment_neg'] = 0
                    data['overview_sentiment_neu'] = 1
                    data['overview_sentiment_pos'] = 0
                    data['overview_sentiment_compound'] = 0
        except Exception as e:
            self.logger.warning(f"警告: 情感分析失败: {e}")
            # 添加默认值
            data['overview_sentiment_neg'] = 0
            data['overview_sentiment_neu'] = 1
            data['overview_sentiment_pos'] = 0
            data['overview_sentiment_compound'] = 0

        # 3. 简单文本复杂度特征
        data['overview_syllable_count'] = data['overview'].apply(
            lambda x: sum([len([c for c in word.lower()
                                if c in 'aeiouy'])
                          for word in str(x).split()])
            if str(x).strip() else 0
        )

        # 4. 词袋特征
        bow_vectorizer = CountVectorizer(max_features=20, stop_words='english')
        bow_matrix = bow_vectorizer.fit_transform(data['overview'].fillna(''))
        bow_df = pd.DataFrame(
            bow_matrix.toarray(),
            columns=[f'bow_{word}' for word in bow_vectorizer.get_feature_names_out()]
        )

        # 5. TF-IDF特征
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data['overview'].fillna(''))
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{word}' for word in tfidf.get_feature_names_out()]
        )

        # 6. 标题文本特征
        if 'title' in data.columns:
            data['title_length'] = data['title'].str.len()
            data['title_word_count'] = data['title'].str.split().str.len()
            # 标题是否包含数字
            data['title_has_number'] = data['title'].apply(
                lambda x: 1 if any(char.isdigit() for char in str(x)) else 0
            )

        # 合并所有文本特征
        data = pd.concat([data, bow_df, tfidf_df], axis=1)

        return data

    def create_interaction_features(self, data):
        """创建交互特征"""
        self.logger.info("正在创建交互特征...")

        # 1. 预算与类型交互
        data['budget_genre_action'] = (
            data['budget'] * data['genre_action']
            if 'genre_action' in data.columns else 0
        )
        data['budget_genre_adventure'] = (
            data['budget'] * data['genre_adventure']
            if 'genre_adventure' in data.columns else 0
        )
        data['budget_genre_fantasy'] = (
            data['budget'] * data['genre_fantasy']
            if 'genre_fantasy' in data.columns else 0
        )

        # 2. 预算与主要制作公司交互（添加列存在性检查）
        if 'has_major_studio' in data.columns:
            data['budget_major_studio'] = data['budget'] * data['has_major_studio']

        # 3. 导演影响力与预算交互
        if 'director_avg_revenue' in data.columns:
            data['director_revenue_budget'] = data['director_avg_revenue'] * data['budget']

        # 4. 评分与投票数交互
        data['rating_vote_interaction'] = data['vote_average'] * data['vote_count']

        # 5. 年份与类型交互
        if 'main_genre_encoded' in data.columns:
            data['year_genre_difference'] = (
                (data['release_year'] - 2000) * data['main_genre_encoded']
            )

        return data

    def feature_scaling(self, data, features_to_scale=None):
        """特征缩放，保留原始特征"""
        self.logger.info("正在进行特征缩放...")

        if features_to_scale is None:
            # 选择数值型特征进行缩放
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            exclude_cols = ['id', 'revenue', 'profit', 'return_on_investment']
            features_to_scale = [col for col in numeric_cols if col not in exclude_cols]

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[features_to_scale])
        scaled_df = pd.DataFrame(
            scaled_features,
            columns=[f'{col}_scaled' for col in features_to_scale]
        )

        # 保留原始特征，添加缩放后的特征
        data = pd.concat([data, scaled_df], axis=1)

        return data, scaler

    def select_features_with_shap(self, X, y, n_features=20):
        """使用SHAP选择重要特征

        Args:
            X: 特征矩阵
            y: 目标变量
            n_features: 要选择的特征数量

        Returns:
            选择的特征名称列表
        """
        self.logger.info("正在使用SHAP选择特征...")

        try:
            # 检查数据是否足够
            if len(X) < 10 or len(X.columns) < 2:
                self.logger.warning("数据量或特征数量不足，无法进行SHAP分析，返回所有特征")
                return X.columns.tolist()

            # 1. 训练随机森林模型（减少树的数量以提高性能）
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)

            # 2. 创建SHAP解释器
            explainer = shap.Explainer(model)

            # 3. 计算SHAP值
            shap_values = explainer(X)

            # 4. 计算特征重要性（使用SHAP值的绝对值的均值）
            feature_importance = np.abs(shap_values.values).mean(axis=0)

            # 5. 选择最重要的n_features个特征
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            })

            # 6. 排序并选择前n_features个特征
            selected_features = feature_importance_df.sort_values(
                'importance', ascending=False
            ).head(n_features)['feature'].tolist()

            self.logger.info(f"使用SHAP选择的特征: {selected_features}")

            return selected_features
        except Exception as e:
            self.logger.warning(f"SHAP特征选择失败: {e}")
            self.logger.warning("将返回所有输入特征")
            return X.columns.tolist()

    def feature_selection(self, data, target='revenue', k=20, use_shap=False):
        """特征选择，支持传统SelectKBest和基于SHAP的特征选择

        Args:
            data: 输入数据
            target: 目标变量
            k: 要选择的特征数量
            use_shap: 是否使用SHAP进行特征选择

        Returns:
            选择的特征和特征名称列表
        """
        self.logger.info("正在进行特征选择...")

        # 1. 选择数值型特征，并排除会导致数据泄露的特征（如estimated_tickets）
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        # 排除导致数据泄露的特征和缩放后的特征
        leak_features = [col for col in data.columns if 'estimated_tickets' in col]
        scaled_features = [col for col in data.columns if '_scaled' in col]
        exclude_cols = [target, 'profit', 'return_on_investment'] + leak_features + scaled_features
        X = data[numeric_cols].drop(exclude_cols, axis=1, errors='ignore')
        y = data[target]

        # 2. 处理NaN值（使用SimpleImputer填充）
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

        # 3. 根据use_shap参数选择特征选择方法
        if use_shap:
            # 使用SHAP选择特征
            selected_features = self.select_features_with_shap(X_imputed_df, y, n_features=k)
        else:
            # 使用SelectKBest选择前k个特征
            selector = SelectKBest(score_func=f_regression, k=k)
            selector.fit(X_imputed, y)
            selected_features = X.columns[selector.get_support()]
            self.logger.info(f"使用SelectKBest选择的特征: {selected_features.tolist()}")

        # 5. 保留选择的特征和目标变量
        # 确定可用的标题列
        title_column = 'title'
        if title_column not in data.columns:
            # 尝试使用title_x或title_y
            if 'title_x' in data.columns:
                title_column = 'title_x'
            elif 'title_y' in data.columns:
                title_column = 'title_y'
            else:
                # 如果没有标题列，不包含标题
                title_column = None

        # 构建保留的列列表
        if title_column:
            keep_columns = list(selected_features) + [target, 'profit', 'return_on_investment', title_column]
        else:
            keep_columns = list(selected_features) + [target, 'profit', 'return_on_investment']

        selected_data = data[keep_columns]

        return selected_data, selected_features

    def dimensionality_reduction(self, data, features, n_components=10):
        """降维处理"""
        self.logger.info("正在进行降维处理...")

        X = data[features]

        # 使用PCA降维
        pca = PCA(n_components=n_components, random_state=42)
        pca_features = pca.fit_transform(X)

        # 创建PCA特征DataFrame
        pca_df = pd.DataFrame(pca_features, columns=[f'pca_component_{i + 1}' for i in range(n_components)])

        # 合并PCA特征
        data = pd.concat([data, pca_df], axis=1)

        # 记录解释方差比
        self.logger.info(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

        return data, pca

    def calculate_seasonal_features(self, data):
        """计算季节性特征"""
        self.logger.info("正在计算季节性特征...")

        # 1. 季节编码
        data['is_summer'] = data['release_month'].apply(lambda x: 1 if 5 <= x <= 8 else 0)  # 夏季
        data['is_winter'] = data['release_month'].apply(lambda x: 1 if x in [11, 12, 1] else 0)  # 冬季
        data['is_holiday_season'] = data['release_month'].apply(lambda x: 1 if x in [11, 12] else 0)  # 节假日季

        # 2. 月份正弦余弦编码
        data['month_sin'] = np.sin(2 * np.pi * data['release_month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['release_month'] / 12)

        # 3. 季度正弦余弦编码
        data['quarter_sin'] = np.sin(2 * np.pi * data['release_quarter'] / 4)
        data['quarter_cos'] = np.cos(2 * np.pi * data['release_quarter'] / 4)

        return data

    def extract_bert_features(self, data, text_columns=['title', 'overview']):
        """使用BERT提取文本特征

        Args:
            data: 输入数据
            text_columns: 要提取特征的文本列

        Returns:
            包含BERT特征的数据
        """
        self.logger.info("\n正在使用BERT提取文本特征...")

        # 检查是否存在已保存的BERT特征文件
        bert_features_file = os.path.join(self.processed_dir, "bert_features.pkl")
        if os.path.exists(bert_features_file):
            try:
                self.logger.info(f"发现已保存的BERT特征文件: {bert_features_file}")
                saved_features = pd.read_pickle(bert_features_file)

                # 检查数据是否匹配
                if len(saved_features) == len(data) and all(saved_features.index == data.index):
                    self.logger.info("直接加载已保存的BERT特征")
                    return pd.concat([data, saved_features], axis=1)
                else:
                    self.logger.warning("已保存的BERT特征与当前数据不匹配，重新提取")
            except Exception as e:
                self.logger.warning(f"加载已保存的BERT特征失败: {e}")
                self.logger.warning("重新提取BERT特征")

        try:
            # 尝试加载预训练BERT模型和分词器
            tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased',
                timeout=10,  # 增加超时时间
                resume_download=True,  # 允许恢复下载
                local_files_only=True,  # 优先使用本地文件
                max_retries=3  # 允许重试
            )
            model = BertModel.from_pretrained(
                'bert-base-uncased',
                timeout=10,  # 增加超时时间
                resume_download=True,  # 允许恢复下载
                local_files_only=True,  # 优先使用本地文件
                max_retries=3  # 允许重试
            )
        except Exception as e:
            self.logger.warning(f"无法加载BERT模型: {e}")
            self.logger.warning("跳过BERT特征提取，继续执行后续步骤")
            return data

        all_bert_features = pd.DataFrame(index=data.index)

        for column in text_columns:
            if column in data.columns:
                self.logger.info(f"正在处理列: {column}")
                embeddings = []

                # 填充缺失值
                text_data = data[column].fillna('')

                for text in tqdm(text_data, desc=f'Extracting BERT features for {column}'):
                    # 分词和编码
                    inputs = tokenizer(
                        text, return_tensors='pt', truncation=True,
                        padding='max_length', max_length=128
                    )  # 减少max_length以提高性能

                    # 获取BERT嵌入
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # 使用最后一层隐藏状态的均值作为文本嵌入
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

                    embeddings.append(embedding)

                # 将嵌入作为新特征添加到数据中
                embeddings_df = pd.DataFrame(
                    embeddings,
                    columns=[f'{column}_bert_{i}' for i in range(embeddings[0].shape[0])],
                    index=data.index
                )
                all_bert_features = pd.concat([all_bert_features, embeddings_df], axis=1)

        # 保存BERT特征到文件
        try:
            all_bert_features.to_pickle(bert_features_file)
            self.logger.info(f"BERT特征已保存到: {bert_features_file}")
        except Exception as e:
            self.logger.warning(f"保存BERT特征失败: {e}")

        # 合并BERT特征到原始数据
        data = pd.concat([data, all_bert_features], axis=1)
        self.logger.info("BERT特征提取完成")
        self.logger.info(
            f"使用BERT提取了 {len(text_columns)} 个文本特征，包含 "
            f"{all_bert_features.shape[1]} 个BERT嵌入维度。"
        )
        return data

    def keyword_analysis(self, data):
        """关键词分析"""
        self.logger.info("正在进行关键词分析...")

        # 确保keywords_list列存在
        if 'keywords_list' not in data.columns:
            return data

        import ast

        # 1. 关键词数量
        # 使用ast.literal_eval替代eval，更安全高效
        def count_keywords(keywords_str):
            if pd.isna(keywords_str) or str(keywords_str).strip() == '[]':
                return 0
            try:
                keywords = ast.literal_eval(keywords_str)
                return len(keywords)
            except (ValueError, SyntaxError):
                return 0

        # 使用向量化的方式应用函数
        data['keywords_count_actual'] = data['keywords_list'].apply(count_keywords)

        # 2. 关键关键词的存在性
        important_keywords = ['sequel', 'based on novel', 'superhero', 'comic book',
                              'action', 'adventure', 'romance', 'horror', 'thriller']

        # 先将关键词列表转换为集合，提高查找效率
        def get_keywords_set(keywords_str):
            if pd.isna(keywords_str) or str(keywords_str).strip() == '[]':
                return set()
            try:
                keywords = ast.literal_eval(keywords_str)
                return {kw.lower() for kw in keywords}
            except (ValueError, SyntaxError):
                return set()

        # 创建关键词集合列
        data['keywords_set'] = data['keywords_list'].apply(get_keywords_set)

        # 使用向量化操作检查关键词存在性
        for keyword in important_keywords:
            keyword_lower = keyword.lower()
            data[f'has_keyword_{keyword}'] = data['keywords_set'].apply(
                lambda kw_set: 1 if keyword_lower in kw_set else 0
            )

        # 删除临时列
        data.drop('keywords_set', axis=1, inplace=True)

        return data

    def run_complete_feature_engineering(
        self, input_file="cleaned_movie_data.csv", output_file="feature_engineered_data.csv",
        bert_features=False, use_shap=False
    ):
        """运行完整的特征工程流程"""
        self.logger.info("=" * 50)
        self.logger.info("开始完整的特征工程流程")
        self.logger.info("=" * 50)
        self.logger.info(f"BERT特征提取状态: {'启用' if bert_features else '禁用'}")

        # 1. 加载数据
        data = self.load_data(input_file)
        if data is None:
            return None

        # 2. 编码分类特征
        data = self.encode_categorical_features(data)

        # 3. 计算导演影响力特征
        data = self.calculate_director_influence(data)

        # 4. 计算演员影响力特征
        data = self.calculate_actor_influence(data)

        # 5. 深入语言处理
        self.logger.info("正在进行深入语言处理...")

        # 5.1 文本特征
        if 'overview' in data.columns:
            data = self.text_features(data)

        # 5.2 关键词分析
        data = self.keyword_analysis(data)

        # 5.3 BERT文本特征提取
        if bert_features:
            try:
                data = self.extract_bert_features(data, text_columns=['title', 'overview'])
            except Exception as e:
                self.logger.warning(f"警告: BERT特征提取失败: {e}")
                self.logger.warning("跳过BERT特征提取，继续执行后续步骤...")
        else:
            self.logger.info("BERT特征提取已禁用")

        # 6. 创建交互特征
        data = self.create_interaction_features(data)

        # 7. 计算季节性特征
        data = self.calculate_seasonal_features(data)

        # 8. 特征缩放
        data, _ = self.feature_scaling(data)

        # 9. 特征选择
        data, selected_features = self.feature_selection(data, k=25, use_shap=use_shap)

        # 10. 保存特征工程后的数据
        output_path = os.path.join(self.processed_dir, output_file)
        data.to_csv(output_path, index=False)
        self.logger.info("\n特征工程完成！")
        self.logger.info(f"数据已保存到: {output_path}")
        self.logger.info(f"选择的特征数量: {len(selected_features)}")

        return data, selected_features

    def prepare_model_data(self, data, target='revenue'):
        """准备模型训练数据"""
        self.logger.info("正在准备模型训练数据...")

        # 1. 选择特征和目标变量
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        X = data[numeric_cols].drop([target, 'profit', 'return_on_investment'], axis=1, errors='ignore')
        y = data[target]

        # 2. 分离训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=data['release_year'] // 5
        )

        self.logger.info(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

        return X_train, X_test, y_train, y_test


def main():
    """主函数，执行特征工程流程"""
    fe = FeatureEngineering()

    # 动态检测可用的数据集文件
    import os
    processed_dir = os.path.join("./data", "processed")

    # 优先使用的数据集文件名列表
    preferred_files = [
        "cleaned_movie_data_large.csv",  # 百万级数据集
        "cleaned_movie_data.csv",        # 常规数据集
        "tmdb_merged.csv"                # TMDB合并数据集
    ]

    # 选择可用的数据集文件
    selected_file = None
    for filename in preferred_files:
        file_path = os.path.join(processed_dir, filename)
        if os.path.exists(file_path):
            selected_file = filename
            logger.info(f"检测到可用数据集: {selected_file}")
            break

    if selected_file is None:
        logger.error("错误: 未找到可用的数据集文件")
        logger.error("请先运行数据预处理脚本: python data_preprocessing.py")
        return

    # 运行特征工程
    fe.run_complete_feature_engineering(input_file=selected_file)


if __name__ == "__main__":
    main()
