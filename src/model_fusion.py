#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型融合模块

该模块实现了不同机器学习模型的融合功能，通过加权平均的方式结合多个模型的预测结果，
以提高电影票房预测的准确性。
"""

import os
import sys

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入标准库和第三方库
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import pickle
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor

# 导入本地模块
from src.utils.logging_config import get_logger
# 导入统一配置（备用）

# 初始化日志记录器
global_logger = get_logger('model_fusion')

class ModelFusion:
    """
    模型融合类，用于融合多种机器学习和深度学习模型的预测结果
    """

    def __init__(self, base_dir=None, results_dir=None):
        """
        初始化模型融合类

        Args:
            base_dir (str): 数据目录基础路径
            results_dir (str): 结果目录路径
        """
        # 使用绝对路径，避免相对路径问题
        if base_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        if results_dir is None:
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "processed")
        self.models_dir = os.path.join(results_dir, "models")
        self.reports_dir = os.path.join(results_dir, "reports")
        self.cache_dir = os.path.join(results_dir, "cache")

        # 创建必要的目录
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.X_train = None
        self.y_train = None
        self.features = None
        self.models = {}
        self.scalers = {}
        self.cache_enabled = True  # 默认启用缓存
        self.cache_ttl = 3600  # 缓存过期时间（秒）

        # 初始化日志记录器
        self.logger = global_logger

        # 模型优先级配置
        self.model_priorities = {
            "transformer_network": 1.5,  # 最高优先级
            "xgboost": 1.2,              # 高优先级
            "lightgbm": 1.2,             # 高优先级
            "gradient_boosting": 1.0,    # 中优先级
            "random_forest": 1.0,        # 中优先级
            "dense_network": 0.9,        # 中低优先级
            "cnn_network": 0.8,          # 中低优先级
            "lstm_network": 0.8,         # 中低优先级
            "gru_network": 0.8,          # 中低优先级
            "ridge_regression": 0.5      # 低优先级
        }

        # 模型权重配置
        self.model_weights = {}
        self.update_model_weights()

    def update_model_weights(self):
        """
        根据模型优先级更新模型权重
        """
        # 只对已加载的模型更新权重
        available_models = list(self.models.keys())
        if not available_models:
            return

        # 根据优先级计算权重
        total_priority = sum(self.model_priorities.get(model, 1.0) for model in available_models)
        self.model_weights = {
            model: self.model_priorities.get(model, 1.0) / total_priority 
            for model in available_models
        }

        self.logger.info(f"模型权重已更新: {self.model_weights}")

    def load_data(self):
        """
        加载特征工程后的数据
        """
        self.logger.info("正在加载特征工程后的数据...")

        # 加载特征工程后的数据
        feature_engineered_file = os.path.join(self.processed_dir, "feature_engineered_data.csv")

        if not os.path.exists(feature_engineered_file):
            self.logger.error(f"错误: 文件不存在: {feature_engineered_file}")
            self.logger.error("请先运行特征工程脚本: python feature_engineering.py")
            self.logger.error("特征工程脚本依赖于数据预处理脚本的输出")
            return None, None

        try:
            data = pd.read_csv(feature_engineered_file)

            # 与modeling.py保持一致的特征选择
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            features = [col for col in numeric_cols if col not in ['revenue', 'profit', 'return_on_investment']]

            # 移除可能存在的'title_x'列（合并数据时产生）
            if 'title_x' in features:
                features.remove('title_x')

            # 确保特征顺序与modeling.py中使用的顺序一致
            # 先对特征进行排序，确保每次运行顺序一致
            features.sort()

            self.X_train = data[features]
            self.y_train = data['revenue']
            self.features = features

            self.logger.info(f"数据加载完成。特征数量: {len(features)}, 样本数量: {len(data)}")
            return self.X_train, self.y_train
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            self.logger.error(f"错误: 加载数据失败: {e}")
            self.logger.error("请检查文件格式是否正确")
            return None, None

    def _generate_cache_key(self, data):
        """
        生成缓存键

        Args:
            data: 用于生成缓存键的数据

        Returns:
            str: 缓存键
        """
        # 使用数据的哈希值作为缓存键
        data_str = str(data)
        hash_obj = hashlib.md5(data_str.encode())
        return hash_obj.hexdigest()

    def _save_cache(self, key, data, description=""):
        """
        保存数据到缓存

        Args:
            key (str): 缓存键
            data: 要缓存的数据
            description (str): 缓存描述
        """
        if not self.cache_enabled:
            return

        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        cache_data = {
            "data": data,
            "description": description,
            "timestamp": time.time(),
            "created_at": datetime.now().isoformat()
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        self.logger.debug(f"✓ 缓存已保存: {description}")

    def _load_cache(self, key, description=""):
        """
        从缓存加载数据

        Args:
            key (str): 缓存键
            description (str): 缓存描述

        Returns:
            缓存的数据或None（如果缓存不存在或已过期）
        """
        if not self.cache_enabled:
            return None

        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")

        if not os.path.exists(cache_path):
            return None

        # 检查缓存是否过期
        cache_mtime = os.path.getmtime(cache_path)
        if time.time() - cache_mtime > self.cache_ttl:
            os.remove(cache_path)  # 删除过期缓存
            return None

        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)

        self.logger.debug(f"✓ 缓存已加载: {description}")
        return cache_data["data"]

    def _check_and_train_models(self):
        """
        检查模型是否存在，如果不存在则自动训练
        """
        self.logger.info("\n检查模型是否存在...")

        # 检查模型目录是否存在
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            self.logger.info(f"创建模型目录: {self.models_dir}")

        # 检查传统机器学习模型
        traditional_models = [
            "gradient_boosting", "random_forest", "xgboost", 
            "lightgbm", "ridge_regression"
        ]

        # 检查深度学习模型
        deep_learning_models = [
            "dense_network", "cnn_network", "lstm_network", 
            "gru_network", "transformer_network"
        ]

        # 检查是否需要训练传统模型
        need_train_traditional = False
        missing_models = []
        for model_name in traditional_models:
            model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
            model_path = os.path.join(self.models_dir, model_filename)
            if not os.path.exists(model_path):
                self.logger.info(f"未找到模型: {model_name}")
                need_train_traditional = True
                missing_models.append(model_name)

        # 训练传统模型
        if need_train_traditional:
            self.logger.info("\n需要训练传统机器学习模型...")
            import subprocess
            import sys

            # 构建命令
            command = [
                sys.executable, 
                "src/modeling.py",
                "--filename", "feature_engineered_data.csv",
                "--model-names" 
            ] + missing_models

            # 如果主程序禁用了缓存，也将--no-cache参数传递给modeling.py
            if not self.cache_enabled:
                command.append("--no-cache")

            self.logger.info(f"运行命令: {' '.join(command)}")

            # 运行建模脚本，设置编码为utf-8以处理中文输出
            result = subprocess.run(command, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), capture_output=True, text=True, encoding='utf-8')

            self.logger.info("模型训练输出:")
            self.logger.info(result.stdout)
            if result.stderr:
                self.logger.error("模型训练错误:")
                self.logger.error(result.stderr)

            self.logger.info(f"模型训练完成，返回码: {result.returncode}")
            
            # 检查模型是否真的训练成功
            for model_name in missing_models:
                model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
                model_path = os.path.join(self.models_dir, model_filename)
                if not os.path.exists(model_path):
                    self.logger.warning(f"⚠️  {model_name} 模型训练完成后仍未找到模型文件")

        # 检查深度学习模型是否需要训练
        need_train_deep = False
        for model_name in deep_learning_models:
            model_filename = f"{model_name.lower().replace(' ', '_')}_model.h5"
            model_path = os.path.join(self.models_dir, model_filename)
            if not os.path.exists(model_path):
                self.logger.info(f"未找到深度学习模型: {model_name}")
                need_train_deep = True
                break

        # 训练深度学习模型
        if need_train_deep:
            self.logger.info("\n需要训练深度学习模型...")
            import subprocess
            import sys

            # 构建命令
            command = [
                sys.executable, 
                "src/deep_learning.py",
                "--filename", "feature_engineered_data.csv"
            ]

            # 如果主程序禁用了缓存，也将--no-cache参数传递给deep_learning.py
            if not self.cache_enabled:
                command.append("--no-cache")

            self.logger.info(f"运行命令: {' '.join(command)}")

            # 运行深度学习脚本，设置编码为utf-8以处理中文输出
            result = subprocess.run(command, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), capture_output=True, text=True, encoding='utf-8')

            self.logger.info("深度学习模型训练输出:")
            self.logger.info(result.stdout)
            if result.stderr:
                self.logger.error("深度学习模型训练错误:")
                self.logger.error(result.stderr)

            self.logger.info(f"深度学习模型训练完成，返回码: {result.returncode}")

        return True

    def load_models(self):
        """
        加载预训练的模型，如果模型不存在则自动训练
        """
        self.logger.info("正在加载预训练模型...")

        # 首先检查并训练模型
        self._check_and_train_models()

        # 传统机器学习模型列表
        traditional_models = [
            "gradient_boosting", "random_forest", "xgboost", 
            "lightgbm", "ridge_regression"
        ]

        # 深度学习模型列表
        deep_learning_models = [
            "dense_network", "cnn_network", "lstm_network", 
            "gru_network", "transformer_network"
        ]

        # 加载传统机器学习模型
        for model_name in traditional_models:
            try:
                model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
                self.models[model_name] = joblib.load(
                    os.path.join(self.models_dir, model_filename)
                )
                self.logger.info(f"✓ 已加载 {model_name} 模型")
            except FileNotFoundError:
                self.logger.warning(f"✗ {model_name} 模型文件不存在")
            except Exception as e:
                self.logger.error(f"✗ 加载 {model_name} 模型失败: {e}")

        # 加载深度学习模型
        for model_name in deep_learning_models:
            try:
                model_filename = f"{model_name.lower().replace(' ', '_')}_model.h5"
                self.models[model_name] = load_model(
                    os.path.join(self.models_dir, model_filename)
                )
                self.logger.info(f"✓ 已加载 {model_name} 模型")
            except FileNotFoundError:
                self.logger.warning(f"✗ {model_name} 模型文件不存在")
            except Exception as e:
                self.logger.error(f"✗ 加载 {model_name} 模型失败: {e}")

        # 加载深度学习所需的scalers
        try:
            self.scalers["x_scaler"] = joblib.load(
                os.path.join(self.models_dir, "deep_learning_scaler_x.joblib")
            )
            self.scalers["y_scaler"] = joblib.load(
                os.path.join(self.models_dir, "deep_learning_scaler_y.joblib")
            )
            self.logger.info("✓ 已加载数据缩放器")
        except FileNotFoundError:
            self.logger.warning("✗ 数据缩放器文件不存在")

        # 如果没有加载到任何模型，抛出异常
        if not self.models:
            raise Exception("没有加载到任何模型，请确保模型文件存在于 results/models/ 目录")

        # 更新模型权重
        self.update_model_weights()

    def _predict_single_model(self, model_name, X_np, X_scaled_2d=None, X_scaled_3d=None):
        """
        使用单个模型进行预测（用于并行处理）

        Args:
            model_name: 模型名称
            X_np: numpy格式的输入数据（传统模型）
            X_scaled_2d: 2D缩放后的数据（Dense, Transformer）
            X_scaled_3d: 3D缩放后的数据（CNN, LSTM, GRU）

        Returns:
            tuple: (模型名称, 预测结果)
        """
        try:
            if model_name in ["gradient_boosting", "random_forest", "xgboost", "lightgbm", "ridge_regression"]:
                prediction = self.models[model_name].predict(X_np)
            elif model_name in ["dense_network", "transformer_network"] and X_scaled_2d is not None:
                y_pred_scaled = self.models[model_name].predict(X_scaled_2d)
                prediction = self.scalers["y_scaler"].inverse_transform(y_pred_scaled).flatten()
            elif model_name in ["cnn_network", "lstm_network", "gru_network"] and X_scaled_3d is not None:
                y_pred_scaled = self.models[model_name].predict(X_scaled_3d)
                prediction = self.scalers["y_scaler"].inverse_transform(y_pred_scaled).flatten()
            else:
                return (model_name, None)

            self.logger.debug(f"✓ {model_name} 模型预测完成")
            return (model_name, prediction)
        except Exception as e:
            self.logger.error(f"✗ {model_name} 模型预测失败: {e}")
            return (model_name, None)

    def predict_individual_models(self):
        """
        使用各个模型进行预测

        Returns:
            dict: 各个模型的预测结果
        """
        self.logger.info("正在使用各个模型进行预测...")

        # 生成缓存键
        cache_key = self._generate_cache_key({
            "features": self.features,
            "sample_count": len(self.X_train),
            "models": list(self.models.keys())
        })

        # 尝试从缓存加载预测结果
        cached_predictions = self._load_cache(cache_key, "模型预测结果")
        if cached_predictions is not None:
            return cached_predictions

        # 并行预测实现
        predictions = {}
        models_to_predict = []

        # 将DataFrame转换为numpy数组，用于传统机器学习模型
        X_np = self.X_train.values
        X_scaled_2d = None
        X_scaled_3d = None

        # 传统机器学习模型列表
        traditional_models = ["gradient_boosting", "random_forest", "xgboost", "lightgbm", "ridge_regression"]

        # 深度学习模型准备
        deep_learning_models = []
        if "x_scaler" in self.scalers and "y_scaler" in self.scalers:
            # 对输入数据进行缩放
            X_scaled = self.scalers["x_scaler"].transform(self.X_train)

            # 重塑数据以适应不同的深度学习模型
            X_scaled_2d = X_scaled
            X_scaled_3d = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

            # 2D模型（Dense, Transformer）
            models_2d = ["dense_network", "transformer_network"]
            # 3D模型（CNN, LSTM, GRU）
            models_3d = ["cnn_network", "lstm_network", "gru_network"]

            deep_learning_models = models_2d + models_3d

        # 收集所有要预测的模型
        for model_name in traditional_models + deep_learning_models:
            if model_name in self.models:
                models_to_predict.append(model_name)

        # 使用线程池并行进行预测
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() + 4)) as executor:
            # 提交任务
            future_to_model = {}
            for model_name in models_to_predict:
                future = executor.submit(
                    self._predict_single_model,
                    model_name=model_name,
                    X_np=X_np,
                    X_scaled_2d=X_scaled_2d,
                    X_scaled_3d=X_scaled_3d
                )
                future_to_model[future] = model_name

            # 获取结果
            for future in future_to_model:
                model_name = future_to_model[future]
                result = future.result()
                if result[1] is not None:
                    pred = result[1]
                    
                    # 检查并处理NaN和inf值
                    if np.isnan(pred).any() or np.isinf(pred).any():
                        self.logger.warning(f"模型 {model_name} 的预测结果中包含NaN或inf值")
                        
                        # 替换inf值
                        pred = np.where(np.isinf(pred), np.finfo(pred.dtype).max, pred)
                        # 替换NaN值为该模型预测的平均值
                        pred = np.where(np.isnan(pred), np.nanmean(pred), pred)
                    
                    predictions[model_name] = pred

        # 保存预测结果到缓存
        self._save_cache(cache_key, predictions, "模型预测结果")

        return predictions

    def weighted_average_fusion(self, predictions, weights=None):
        """
        使用加权平均法融合模型预测结果

        Args:
            predictions (dict): 各个模型的预测结果
            weights (dict): 模型权重

        Returns:
            np.array: 融合后的预测结果
        """
        print("正在进行模型融合...")

        # 如果没有提供权重，则使用平均权重
        if weights is None:
            weights = {model: 1.0 / len(predictions) for model in predictions}

        # 验证权重和预测是否匹配
        for model in predictions:
            if model not in weights:
                raise ValueError(f"预测结果中存在模型 {model}，但权重字典中没有对应的权重")

        # 计算加权平均
        fused_predictions = np.zeros(len(next(iter(predictions.values()))))

        for model, pred in predictions.items():
            fused_predictions += pred * weights[model]

        print(f"模型融合完成。使用的权重: {weights}")
        return fused_predictions

    def performance_based_weighted_fusion(self, predictions, y_true):
        """
        基于模型性能的加权平均融合

        Args:
            predictions (dict): 各个模型的预测结果
            y_true (np.array): 真实值

        Returns:
            np.array: 融合后的预测结果
        """
        print("正在进行基于性能的模型融合...")

        # 计算每个模型的性能分数（使用R2 Score）
        performance_scores = {}
        for model_name, pred in predictions.items():
            r2 = r2_score(y_true, pred)
            # 将R2 Score转换为正数权重
            performance_scores[model_name] = max(0, r2)

        # 计算权重总和
        total_score = sum(performance_scores.values())

        # 如果所有模型性能都很差，使用平均权重
        if total_score == 0:
            weights = {model: 1.0 / len(predictions) for model in predictions}
        else:
            # 计算基于性能的权重
            weights = {model: score / total_score for model, score in performance_scores.items()}

        # 使用加权平均融合
        return self.weighted_average_fusion(predictions, weights)

    def stacking_fusion(self, predictions, y_true):
        """
        使用堆叠法融合模型预测结果

        Args:
            predictions (dict): 各个模型的预测结果
            y_true (np.array): 真实值

        Returns:
            np.array: 融合后的预测结果
        """
        print("正在进行堆叠模型融合...")

        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split

        # 将预测结果转换为堆叠特征
        model_predictions = list(predictions.values())
        
        # 检查并处理NaN和inf值
        clean_predictions = []
        for i, pred in enumerate(model_predictions):
            # 替换inf值
            pred = np.where(np.isinf(pred), np.finfo(pred.dtype).max, pred)
            # 替换NaN值为该模型预测的平均值
            pred = np.where(np.isnan(pred), np.nanmean(pred), pred)
            clean_predictions.append(pred)
        
        # 将预测结果转换为堆叠特征
        stacked_features = np.column_stack(clean_predictions)
        
        # 检查堆叠特征中是否还有NaN或inf值
        if np.isnan(stacked_features).any() or np.isinf(stacked_features).any():
            self.logger.warning("堆叠特征中仍存在NaN或inf值，将使用中位数进行替换")
            # 使用中位数替换所有NaN和inf值
            median_values = np.nanmedian(stacked_features, axis=0)
            for i in range(stacked_features.shape[1]):
                col = stacked_features[:, i]
                col = np.where(np.isinf(col), median_values[i], col)
                col = np.where(np.isnan(col), median_values[i], col)
                stacked_features[:, i] = col

        # 划分训练集和验证集
        X_stack_train, X_stack_val, y_stack_train, y_stack_val = train_test_split(
            stacked_features, y_true, test_size=0.3, random_state=42
        )

        # 使用Ridge回归作为元模型
        meta_model = Ridge(random_state=42, alpha=1.0)
        meta_model.fit(X_stack_train, y_stack_train)

        # 预测
        stacked_predictions = meta_model.predict(stacked_features)

        # 评估元模型性能
        r2 = r2_score(y_stack_val, meta_model.predict(X_stack_val))
        print(f"堆叠模型融合完成。元模型R2 Score: {r2:.4f}")

        return stacked_predictions

    def evaluate_fusion(self, y_true, y_pred, model_name="融合模型"):
        """
        评估融合模型的性能

        Args:
            y_true (array): 真实值
            y_pred (array): 预测值
            model_name (str): 模型名称

        Returns:
            dict: 评估指标
        """
        self.logger.info(f"正在评估 {model_name} 的性能...")

        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 计算MAPE（平均绝对百分比误差）
        # 避免除以零的情况
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        metrics = {
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2 Score": r2
        }

        self.logger.info(f"{model_name} 性能评估完成:")
        self.logger.info(f"  RMSE: {rmse:.4f}")
        self.logger.info(f"  MAE: {mae:.4f}")
        self.logger.info(f"  MAPE: {mape:.4f}%")
        self.logger.info(f"  R2 Score: {r2:.4f}")

        return metrics

    def plot_comparison(self, individual_metrics, fusion_metrics):
        """
        绘制单个模型与融合模型的性能对比图

        Args:
            individual_metrics (list): 单个模型的评估指标
            fusion_metrics (dict): 融合模型的评估指标
        """
        self.logger.info("正在绘制模型性能对比图...")

        # 合并所有指标
        all_metrics = individual_metrics + [fusion_metrics]
        metrics_df = pd.DataFrame(all_metrics)

        # 绘制RMSE对比图
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Model", y="RMSE", data=metrics_df)
        plt.title("不同模型的RMSE对比")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "model_rmse_comparison.png"), dpi=300)
        self.logger.info("✓ RMSE对比图已保存")

        # 绘制R2 Score对比图
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Model", y="R2 Score", data=metrics_df)
        plt.title("不同模型的R2 Score对比")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "model_r2_comparison.png"), dpi=300)
        self.logger.info("✓ R2 Score对比图已保存")

        # 保存所有指标到CSV文件
        metrics_df.to_csv(os.path.join(self.reports_dir, "model_comparison_metrics.csv"), index=False)
        self.logger.info("✓ 模型对比指标已保存到CSV文件")

    def run_model_fusion(self, fusion_strategy="all"):
        """
        运行完整的模型融合流程

        Args:
            fusion_strategy (str): 融合策略
                - "average": 简单平均融合
                - "weighted": 加权平均融合
                - "performance": 基于性能的加权融合
                - "stacking": 堆叠融合
                - "all": 所有策略都运行

        Returns:
            dict: 融合模型的评估指标
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("开始完整的模型融合流程")
        self.logger.info("="*50)

        try:
            # 加载数据
            self.load_data()

            # 加载模型
            self.load_models()

            # 各个模型进行预测
            predictions = self.predict_individual_models()

            # 计算单个模型的评估指标
            individual_metrics = self._evaluate_individual_models(predictions)

            # 定义要运行的融合策略
            strategies = self._get_fusion_strategies(fusion_strategy)

            # 运行所有指定的融合策略
            all_fusion_metrics = self._run_all_fusion_strategies(strategies, predictions)

            # 绘制对比图，只使用最后一个融合策略的结果
            if all_fusion_metrics:
                self.plot_comparison(individual_metrics, all_fusion_metrics[-1])

            print("\n" + "="*50)
            print("模型融合流程完成！")
            print("="*50)

            return all_fusion_metrics[-1] if all_fusion_metrics else None

        except Exception as e:
            print(f"模型融合过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _evaluate_individual_models(self, predictions):
        """
        评估所有单个模型的性能

        Args:
            predictions (dict): 各个模型的预测结果

        Returns:
            list: 各个模型的评估指标
        """
        individual_metrics = []
        for model_name, pred in predictions.items():
            metrics = self.evaluate_fusion(self.y_train, pred, model_name)
            individual_metrics.append(metrics)
        return individual_metrics

    def _get_fusion_strategies(self, fusion_strategy):
        """
        根据输入参数获取要运行的融合策略

        Args:
            fusion_strategy (str): 融合策略
                - "average": 简单平均融合
                - "weighted": 加权平均融合
                - "performance": 基于性能的加权融合
                - "stacking": 堆叠融合
                - "all": 所有策略都运行

        Returns:
            list: 要运行的融合策略列表
        """
        if fusion_strategy == "all":
            return ["average", "weighted", "performance", "stacking"]
        else:
            return [fusion_strategy]

    def _run_all_fusion_strategies(self, strategies, predictions):
        """
        运行所有指定的融合策略

        Args:
            strategies (list): 要运行的融合策略列表
            predictions (dict): 各个模型的预测结果

        Returns:
            list: 所有融合策略的评估指标
        """
        all_fusion_metrics = []
        for strategy in strategies:
            print("\n" + "-"*40)
            print(f"使用 {strategy} 策略进行模型融合")
            print("-"*40)

            # 生成融合缓存键
            fusion_cache_key = self._generate_cache_key({
                "strategy": strategy,
                "predictions": self._generate_cache_key(predictions),
                "y_true": self._generate_cache_key(self.y_train.tolist())
            })

            # 尝试从缓存加载融合结果
            cached_fusion_result = self._load_cache(fusion_cache_key, f"{strategy} 融合结果")

            if cached_fusion_result is not None:
                fused_pred, fusion_metrics = cached_fusion_result
            else:
                # 根据策略进行融合
                fused_pred = self._perform_fusion(strategy, predictions)

                # 评估融合模型
                fusion_metrics = self.evaluate_fusion(self.y_train, fused_pred, f"{strategy} 融合模型")

                # 保存融合结果到缓存
                self._save_cache(fusion_cache_key, (fused_pred, fusion_metrics), f"{strategy} 融合结果")

            all_fusion_metrics.append(fusion_metrics)
        return all_fusion_metrics

    def _perform_fusion(self, strategy, predictions):
        """
        根据指定策略执行模型融合

        Args:
            strategy (str): 融合策略
            predictions (dict): 各个模型的预测结果

        Returns:
            np.array: 融合后的预测结果
        """
        if strategy == "average":
            # 简单平均融合
            weights = {model: 1.0 / len(predictions) for model in predictions}
            fused_pred = self.weighted_average_fusion(predictions, weights)
        elif strategy == "weighted":
            # 使用基于模型优先级的动态权重
            weights = self.model_weights.copy()
            # 过滤掉不在预测结果中的模型
            weights = {model: weight for model, weight in weights.items() if model in predictions}
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                weights = {model: 1.0 / len(predictions) for model in predictions}
            fused_pred = self.weighted_average_fusion(predictions, weights)
        elif strategy == "performance":
            # 基于性能的加权融合
            fused_pred = self.performance_based_weighted_fusion(predictions, self.y_train)
        elif strategy == "stacking":
            # 堆叠融合
            fused_pred = self.stacking_fusion(predictions, self.y_train)
        else:
            raise ValueError(f"不支持的融合策略: {strategy}")

        return fused_pred

def main():
    """
    主函数
    """
    import argparse
    from src.utils.logging_config import get_logger

    # 初始化日志记录器
    logger = get_logger('model_fusion_main')

    parser = argparse.ArgumentParser(description="模型融合脚本")
    parser.add_argument("--strategy", type=str, default="all", 
                      choices=["average", "weighted", "performance", "stacking", "all"],
                      help="融合策略")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument("--cache-ttl", type=int, default=3600, 
                      help="缓存过期时间（秒）")

    args = parser.parse_args()

    try:
        # 创建模型融合实例（使用默认的绝对路径）
        model_fusion = ModelFusion()

        # 配置缓存
        model_fusion.cache_enabled = not args.no_cache
        model_fusion.cache_ttl = args.cache_ttl

        logger.info(f"缓存设置: {'启用' if model_fusion.cache_enabled else '禁用'}")
        if model_fusion.cache_enabled:
            logger.info(f"缓存过期时间: {model_fusion.cache_ttl} 秒")

        # 运行模型融合流程
        model_fusion.run_model_fusion(fusion_strategy=args.strategy)

        logger.info("\n模型融合流程成功完成！")
        logger.info("查看结果:")
        logger.info("- 模型对比图表: results/reports/ 目录下")
        logger.info("- 模型对比指标: results/reports/model_comparison_metrics.csv")
        if model_fusion.cache_enabled:
            logger.info("- 缓存文件: results/cache/ 目录下")

    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
