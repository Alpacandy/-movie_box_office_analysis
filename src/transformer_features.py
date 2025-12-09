#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer特征提取模块

该模块使用预训练的BERT模型提取电影文本特征，包括标题和概述的语义表示，
用于增强电影票房预测模型的性能。
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import pickle

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('transformer_features')

try:
    # 尝试导入transformers库，处理导入失败情况
    from transformers import BertTokenizer, BertModel, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
except ImportError as e:
    logger.warning(f"无法导入transformers库: {e}")
    logger.info("请安装transformers库: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

class TransformerFeatureExtractor:
    """使用预训练BERT模型提取文本特征"""

    def __init__(self, base_dir="./data", model_name="bert-base-uncased"):
        """初始化Transformer特征提取器

        Args:
            base_dir: 数据基础目录
            model_name: 预训练模型名称
        """
        self.base_dir = base_dir
        self.model_name = model_name
        self.processed_dir = os.path.join(base_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        # 检查transformers库是否可用
        if not TRANSFORMERS_AVAILABLE:
            self.tokenizer = None
            self.model = None
            logger.error("transformers库不可用，无法初始化BERT模型")
            return

        # 加载预训练模型和分词器
        try:
            logger.info(f"正在加载预训练模型: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info("预训练模型加载完成")

            # 初始化情感分析pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if device.type == 'cuda' else -1
            )
            logger.info("情感分析模型加载完成")
        except Exception as e:
            logger.error(f"加载预训练模型失败: {e}")
            self.tokenizer = None
            self.model = None
            self.sentiment_analyzer = None

    def load_data(self, filename="cleaned_movie_data.csv"):
        """加载处理后的数据"""
        file_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None

        try:
            data = pd.read_csv(file_path)
            logger.info(f"成功加载数据，形状: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return None

    def extract_bert_features(self, text_list, batch_size=32):
        """提取BERT特征

        Args:
            text_list: 文本列表
            batch_size: 批量处理大小

        Returns:
            numpy数组: BERT特征矩阵
        """
        if not TRANSFORMERS_AVAILABLE or self.tokenizer is None or self.model is None:
            logger.error("BERT模型不可用，无法提取特征")
            return None

        features = []

        # 批量处理文本
        for i in tqdm(range(0, len(text_list), batch_size), desc="提取BERT特征"):
            batch_texts = text_list[i:i+batch_size]

            # 分词和编码
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(device)

            # 前向传播，获取CLS嵌入
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                # 获取CLS嵌入作为文本特征
                batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.append(batch_features)

        # 合并所有批次的特征
        return np.vstack(features)

    def extract_sentiment_features(self, text_list, batch_size=32):
        """提取情感特征

        Args:
            text_list: 文本列表
            batch_size: 批量处理大小

        Returns:
            pandas DataFrame: 包含情感得分的DataFrame
        """
        if not TRANSFORMERS_AVAILABLE or self.sentiment_analyzer is None:
            logger.error("情感分析模型不可用，无法提取情感特征")
            return None

        sentiment_results = []

        # 批量处理文本
        for i in tqdm(range(0, len(text_list), batch_size), desc="提取情感特征"):
            batch_texts = text_list[i:i+batch_size]

            # 进行情感分析
            results = self.sentiment_analyzer(batch_texts)
            sentiment_results.extend(results)

        # 转换为DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        sentiment_df.columns = ['sentiment_label', 'sentiment_score']

        # 转换标签为数值
        sentiment_df['sentiment_numeric'] = sentiment_df['sentiment_label'].map({'POSITIVE': 1, 'NEGATIVE': 0})

        return sentiment_df

    def process_text_columns(self, data, text_columns=['overview', 'title']):
        """处理文本列，提取BERT特征

        Args:
            data: 输入数据
            text_columns: 要处理的文本列列表

        Returns:
            pandas DataFrame: 包含BERT特征的数据
        """
        if not TRANSFORMERS_AVAILABLE or self.tokenizer is None or self.model is None:
            logger.error("BERT模型不可用，无法处理文本列")
            return data

        processed_data = data.copy()

        for col in text_columns:
            if col not in processed_data.columns:
                logger.warning(f"列 {col} 不存在于数据中，跳过")
                continue

            logger.info(f"正在处理列: {col}")

            # 处理缺失值
            text_list = processed_data[col].fillna("Unknown").tolist()

            # 提取BERT特征
            bert_features = self.extract_bert_features(text_list)
            if bert_features is None:
                logger.error(f"无法提取 {col} 的BERT特征")
                continue

            # 提取情感特征
            sentiment_df = self.extract_sentiment_features(text_list)
            if sentiment_df is not None:
                # 添加情感特征
                processed_data[f"{col}_sentiment_label"] = sentiment_df['sentiment_label']
                processed_data[f"{col}_sentiment_score"] = sentiment_df['sentiment_score']
                processed_data[f"{col}_sentiment_numeric"] = sentiment_df['sentiment_numeric']

            # 降维BERT特征（使用PCA）
            from sklearn.decomposition import PCA
            logger.info(f"正在对 {col} 的BERT特征进行PCA降维")

            # 根据特征数量选择PCA组件数
            n_components = min(128, bert_features.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            reduced_features = pca.fit_transform(bert_features)

            logger.info(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

            # 添加降维后的BERT特征到数据中
            for i in range(reduced_features.shape[1]):
                processed_data[f"{col}_bert_{i+1}"] = reduced_features[:, i]

            # 保存PCA模型
            pca_file = os.path.join(self.processed_dir, f"pca_{col}.pkl")
            with open(pca_file, 'wb') as f:
                pickle.dump(pca, f)
            logger.info(f"PCA模型已保存到: {pca_file}")

        return processed_data

    def add_text_complexity_features(self, data, text_columns=['overview', 'title']):
        """添加文本复杂度特征

        Args:
            data: 输入数据
            text_columns: 要处理的文本列列表

        Returns:
            pandas DataFrame: 包含文本复杂度特征的数据
        """
        processed_data = data.copy()

        for col in text_columns:
            if col not in processed_data.columns:
                logger.warning(f"列 {col} 不存在于数据中，跳过")
                continue

            logger.info(f"正在添加 {col} 的文本复杂度特征")

            # 处理缺失值
            text_list = processed_data[col].fillna("Unknown")

            # 计算文本长度
            processed_data[f"{col}_length"] = text_list.str.len()

            # 计算单词数量
            processed_data[f"{col}_word_count"] = text_list.str.split().str.len()

            # 计算平均单词长度
            processed_data[f"{col}_avg_word_length"] = text_list.apply(
                lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).strip() else 0
            )

            # 计算句子数量
            processed_data[f"{col}_sentence_count"] = text_list.str.count('\.|\?|!') + 1

        return processed_data

    def run_complete_feature_extraction(self, input_file="cleaned_movie_data.csv", output_file="transformer_enhanced_data.csv"):
        """运行完整的特征提取流程

        Args:
            input_file: 输入数据文件名
            output_file: 输出数据文件名

        Returns:
            pandas DataFrame: 包含所有特征的数据
        """
        logger.info("开始Transformer特征提取流程")

        # 1. 加载数据
        data = self.load_data(input_file)
        if data is None:
            logger.error("数据加载失败，流程终止")
            return None

        # 2. 添加文本复杂度特征
        data = self.add_text_complexity_features(data)

        # 3. 提取Transformer特征
        data = self.process_text_columns(data)

        # 4. 保存处理后的数据
        output_path = os.path.join(self.processed_dir, output_file)
        try:
            data.to_csv(output_path, index=False)
            logger.info(f"处理后的数据已保存到: {output_path}")
            logger.info(f"增强后的数据形状: {data.shape}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return None

        return data

    def batch_process(self, data, batch_size=1000, text_columns=['overview', 'title']):
        """批量处理大规模数据

        Args:
            data: 输入数据
            batch_size: 批量大小
            text_columns: 要处理的文本列列表

        Returns:
            pandas DataFrame: 包含所有特征的数据
        """
        logger.info(f"开始批量处理，批量大小: {batch_size}")

        processed_batches = []
        total_batches = (len(data) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(data), batch_size), total=total_batches, desc="批量处理"):
            batch = data.iloc[i:i+batch_size].copy()

            # 添加文本复杂度特征
            batch = self.add_text_complexity_features(batch, text_columns)

            # 提取Transformer特征
            batch = self.process_text_columns(batch, text_columns)

            processed_batches.append(batch)

        # 合并所有批次
        processed_data = pd.concat(processed_batches, ignore_index=True)
        logger.info(f"批量处理完成，最终数据形状: {processed_data.shape}")

        return processed_data

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Transformer特征提取工具')
    parser.add_argument('--input', '-i', type=str, default='cleaned_movie_data.csv', help='输入数据文件名')
    parser.add_argument('--output', '-o', type=str, default='transformer_enhanced_data.csv', help='输出数据文件名')
    parser.add_argument('--model', '-m', type=str, default='bert-base-uncased', help='预训练模型名称')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='批量处理大小')
    parser.add_argument('--base-dir', '-d', type=str, default='./data', help='数据基础目录')
    parser.add_argument('--large-data', action='store_true', help='大规模数据模式，使用批量处理')

    args = parser.parse_args()

    # 初始化特征提取器
    extractor = TransformerFeatureExtractor(
        base_dir=args.base_dir,
        model_name=args.model
    )

    # 检查transformers库是否可用
    if not TRANSFORMERS_AVAILABLE:
        logger.error("transformers库不可用，无法运行特征提取")
        logger.info("请安装transformers库: pip install transformers torch")
        return

    # 加载数据
    data = extractor.load_data(args.input)
    if data is None:
        return

    # 运行特征提取
    if args.large_data or len(data) > 10000:
        # 大规模数据模式，使用批量处理
        logger.info("启用大规模数据处理模式")
        processed_data = extractor.batch_process(data, batch_size=args.batch_size)

        # 保存数据
        output_path = os.path.join(extractor.processed_dir, args.output)
        processed_data.to_csv(output_path, index=False)
        logger.info(f"批量处理后的数据已保存到: {output_path}")
    else:
        # 常规模式
        extractor.run_complete_feature_extraction(args.input, args.output)

if __name__ == "__main__":
    main()
