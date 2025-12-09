# 文档字符串和注释可以在导入之前

# 先设置sys.path来导入本地模块（这必须在所有导入之前）
import os
import sys

# 获取项目根目录并添加到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 然后导入所有其他模块
import warnings
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from tqdm import tqdm
from src.utils.logging_config import get_logger


# 设置警告过滤和下载nltk数据
warnings.filterwarnings('ignore')

class TextAnalysis:
    def __init__(self, base_dir="../data"):
        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "processed")
        self.raw_dir = os.path.join(base_dir, "raw")
        os.makedirs(self.processed_dir, exist_ok=True)

        # 初始化日志系统
        self.logger = get_logger('text_analysis.TextAnalysis')

        # 初始化情感分析器，处理下载失败情况
        try:
            # 尝试直接使用，如果已安装就不会报错
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            try:
                # 尝试下载
                self.logger.info("正在下载vader_lexicon...")
                nltk.download('vader_lexicon', quiet=True)
                self.sia = SentimentIntensityAnalyzer()
            except Exception as e:
                self.logger.warning(f"无法下载或初始化vader_lexicon: {e}")
                self.logger.warning("情感分析功能将不可用")
                self.sia = None

        # 初始化Transformer模型
        self.embedding_model = None
        self.topic_model = None

    def load_data(self, filename="cleaned_movie_data.csv"):
        """加载处理后的数据"""
        file_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"错误: 文件不存在: {file_path}")
            self.logger.error("请先运行数据预处理脚本: python data_preprocessing.py")
            return None

        self.logger.info(f"正在加载数据: {filename}")
        data = pd.read_csv(file_path)
        self.logger.info(f"数据形状: {data.shape}")
        return data

    def initialize_models(self, embedding_model_name="all-MiniLM-L6-v2"):
        """初始化Transformer模型"""
        self.logger.info(f"\n正在初始化Transformer模型: {embedding_model_name}")

        # 初始化SentenceTransformer
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
            self.logger.info(f"成功加载嵌入模型: {embedding_model_name}")
        except Exception as e:
            self.logger.error(f"错误: 加载嵌入模型失败: {e}")
            self.logger.info("尝试使用默认模型...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # 初始化BERTopic
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics="auto",
            min_topic_size=5,
            verbose=True
        )
        self.logger.info("成功初始化BERTopic模型")

    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""

        # 转换为字符串
        text = str(text)

        # 去除多余空格
        text = " ".join(text.split())

        return text

    def generate_sentiment_scores(self, texts):
        """生成情感分数"""
        self.logger.info("\n正在生成情感分数...")

        sentiment_scores = []
        for text in tqdm(texts, desc="计算情感分数"):
            score = self.sia.polarity_scores(text)
            sentiment_scores.append(score)

        # 转换为DataFrame
        sentiment_df = pd.DataFrame(sentiment_scores)
        sentiment_df.columns = [f"sentiment_{col}" for col in sentiment_df.columns]

        return sentiment_df

    def generate_text_embeddings(self, texts):
        """生成文本嵌入"""
        self.logger.info("\n正在生成文本嵌入...")

        # 确保模型已初始化
        if not self.embedding_model:
            self.initialize_models()

        # 生成嵌入
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, batch_size=32)

        # 转换为DataFrame
        embedding_df = pd.DataFrame(embeddings, columns=[f"embedding_{i}" for i in range(embeddings.shape[1])])

        return embedding_df

    def perform_topic_modeling(self, texts):
        """执行主题建模"""
        self.logger.info("\n正在执行主题建模...")

        # 确保模型已初始化
        if not self.topic_model:
            self.initialize_models()

        # 执行主题建模
        topics, probabilities = self.topic_model.fit_transform(texts)

        # 获取主题信息
        topic_info = self.topic_model.get_topic_info()
        self.logger.info(f"\n主题建模完成，共识别出{len(topic_info)-1}个主题")

        # 保存主题信息
        topic_info_path = os.path.join(self.processed_dir, "topic_info.csv")
        topic_info.to_csv(topic_info_path, index=False)
        self.logger.info(f"主题信息已保存到: {topic_info_path}")

        # 可视化主题
        fig = self.topic_model.visualize_topics()
        fig.write_html(os.path.join(self.processed_dir, "topic_visualization.html"))
        self.logger.info(f"主题可视化已保存到: {os.path.join(self.processed_dir, 'topic_visualization.html')}")

        return topics, probabilities

    def extract_keywords(self, texts, num_keywords=5):
        """提取关键词"""
        self.logger.info(f"\n正在提取关键词，每个文本提取{num_keywords}个关键词...")

        keywords_list = []

        for text in tqdm(texts, desc="提取关键词"):
            # 使用BERTopic的extract_keywords方法
            keywords = self.topic_model._extract_embedding_based_keywords(
                text, 
                embeddings=self.embedding_model.encode(text), 
                top_n=num_keywords
            )

            # 只保留关键词文本
            keywords = [keyword[0] for keyword in keywords]
            keywords_list.append(keywords)

        return keywords_list

    def run_text_analysis(self, data, text_column="overview"):
        """运行完整的文本分析流程"""
        self.logger.info("=" * 50)
        self.logger.info("开始文本分析流程")
        self.logger.info("=" * 50)

        # 1. 检查文本列是否存在
        if text_column not in data.columns:
            self.logger.error(f"错误: 数据中不存在列: {text_column}")
            return None

        # 2. 预处理文本
        self.logger.info(f"\n正在预处理{text_column}列的文本...")
        data["processed_text"] = data[text_column].apply(self.preprocess_text)

        # 3. 过滤空文本
        initial_count = len(data)
        data = data[data["processed_text"] != ""]
        filtered_count = len(data)
        self.logger.info(f"过滤了{initial_count - filtered_count}条空文本，剩余{filtered_count}条文本用于分析")

        if len(data) == 0:
            self.logger.error("错误: 没有可用的文本数据")
            return None

        # 4. 初始化模型
        self.initialize_models()

        # 5. 生成情感分数
        sentiment_df = self.generate_sentiment_scores(data["processed_text"])

        # 6. 生成文本嵌入
        embedding_df = self.generate_text_embeddings(data["processed_text"])

        # 7. 执行主题建模
        topics, probabilities = self.perform_topic_modeling(data["processed_text"])

        # 8. 提取关键词
        keywords = self.extract_keywords(data["processed_text"], num_keywords=5)
        data["keywords"] = keywords

        # 9. 合并结果
        text_analysis_results = pd.concat([
            data.reset_index(drop=True),
            sentiment_df,
            embedding_df,
            pd.DataFrame({"topic_id": topics, "topic_probability": probabilities})
        ], axis=1)

        # 10. 保存结果
        results_path = os.path.join(self.processed_dir, "text_analysis_results.csv")
        text_analysis_results.to_csv(results_path, index=False)
        self.logger.info(f"\n文本分析结果已保存到: {results_path}")

        self.logger.info("\n" + "=" * 50)
        self.logger.info("文本分析流程完成！")
        self.logger.info("=" * 50)

        return text_analysis_results

    def visualize_text_results(self, data):
        """可视化文本分析结果"""
        self.logger.info("\n正在生成文本分析可视化...")

        # 1. 主题分布
        self.logger.info("\n1. 主题分布")
        topic_counts = data["topic_id"].value_counts().sort_index()
        topic_counts = topic_counts[topic_counts.index != -1]  # 排除异常主题

        # 保存主题分布
        topic_dist_path = os.path.join(self.processed_dir, "topic_distribution.csv")
        topic_counts.to_csv(topic_dist_path)
        self.logger.info(f"主题分布已保存到: {topic_dist_path}")

        # 2. 情感分数分布
        self.logger.info("\n2. 情感分数分布")
        sentiment_cols = [col for col in data.columns if col.startswith("sentiment_")]
        sentiment_stats = data[sentiment_cols].describe()

        # 保存情感统计
        sentiment_stats_path = os.path.join(self.processed_dir, "sentiment_stats.csv")
        sentiment_stats.to_csv(sentiment_stats_path)
        self.logger.info(f"情感统计已保存到: {sentiment_stats_path}")

        self.logger.info("文本分析可视化完成！")

    def main(self):
        """主函数"""
        # 1. 加载数据
        data = self.load_data()
        if data is None:
            return

        # 2. 运行文本分析
        results = self.run_text_analysis(data, text_column="overview")
        if results is None:
            return

        # 3. 可视化结果
        self.visualize_text_results(results)

def main():
    """主函数"""
    ta = TextAnalysis()
    ta.main()

if __name__ == "__main__":
    main()