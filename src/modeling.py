import os
import sys
# 导入标准库
import traceback
import joblib
import warnings
import time
import hashlib
import pickle
from datetime import datetime

# 导入第三方库
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import optuna
import shap
# 添加项目根目录到sys.path，这必须在其他导入之前
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"当前文件路径: {os.path.abspath(__file__)}")
print(f"项目根目录: {project_root}")
sys.path.insert(0, project_root)  # 将项目根目录添加到sys.path的开头，确保优先使用
print(f"当前sys.path: {sys.path}")
# 导入本地模块
from src.utils.logging_config import get_logger
from src.utils.config_manager import global_config




# 模型解释库
try:
    from lime import lime_tabular
    MODEL_INTERPRETABILITY_ENABLED = True
except ImportError:
    MODEL_INTERPRETABILITY_ENABLED = False


# 设置matplotlib为非交互式后端，避免线程问题
matplotlib.use('Agg')  # 使用Agg后端，适合非交互式环境

# 初始化日志记录器
logger = get_logger('modeling')

# 检查LIME库是否安装
if not MODEL_INTERPRETABILITY_ENABLED:
    logger.warning("LIME库未安装，部分模型解释功能将不可用")
    logger.warning("请运行: pip install lime")

# 检查LIME库是否安装
if 'lime' not in sys.modules:
    logger.warning("LIME库未安装，部分模型解释功能将不可用")
    logger.warning("请运行: pip install lime")
    MODEL_INTERPRETABILITY_ENABLED = False

warnings.filterwarnings('ignore')

# 设置可视化风格
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
# 设置英文显示以避免乱码
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TraditionalModeling:
    def __init__(self, base_dir=None, results_dir=None, large_data_mode=False, chunk_size=100000, cache_enabled=True, cache_ttl=3600):
        # 参数验证
        if chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")

        self.logger = get_logger('modeling.TraditionalModeling')

        if not isinstance(cache_enabled, bool) or not isinstance(large_data_mode, bool):
            raise ValueError("cache_enabled和large_data_mode必须是布尔值")

        # 使用项目根目录作为基础路径
        self.base_dir = os.path.join(project_root, "data") if base_dir is None else base_dir
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.results_dir = os.path.join(project_root, "results") if results_dir is None else results_dir
        self.models_dir = os.path.join(self.results_dir, "models")
        self.charts_dir = os.path.join(self.results_dir, "charts")
        self.cache_dir = os.path.join(self.results_dir, "cache")
        self.large_data_mode = large_data_mode  # 大规模数据模式开关
        self.chunk_size = chunk_size  # 分块大小
        self.cache_enabled = cache_enabled  # 缓存开关
        self.cache_ttl = cache_ttl  # 缓存过期时间（秒）

        try:
            # 确保目录存在并检查写入权限
            for dir_path in [self.models_dir, self.charts_dir, self.cache_dir]:
                os.makedirs(dir_path, exist_ok=True)
                if not os.access(dir_path, os.W_OK):
                    self.logger.warning(f"目录 {dir_path} 不可写")
        except Exception as e:
            self.logger.warning(f"创建目录结构失败: {e}")

    def _generate_cache_key(self, data):
        """
        生成缓存键，优化生成逻辑以提高性能

        Args:
            data: 用于生成缓存键的数据

        Returns:
            str: 缓存键
        """
        # 使用更高效的哈希方式生成缓存键
        import json
        if isinstance(data, dict):
            # 对字典进行排序以确保一致性
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        # 使用SHA256哈希算法，生成更安全的缓存键
        hash_obj = hashlib.sha256(data_str.encode())
        return hash_obj.hexdigest()

    def _save_cache(self, key, data, description=""):
        """
        保存数据到缓存，优化缓存保存逻辑

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

        # 使用更高效的pickle协议
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"✓ Cache saved: {description}")

    def _load_cache(self, key, description=""):
        """
        从缓存加载数据，增强错误处理

        Args:
            key (str): 缓存键
            description (str): 缓存描述

        Returns:
            缓存的数据或None（如果缓存不存在、已过期或损坏）
        """
        if not self.cache_enabled:
            return None

        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")

        if not os.path.exists(cache_path):
            return None

        # 检查缓存是否过期
        cache_mtime = os.path.getmtime(cache_path)
        if time.time() - cache_mtime > self.cache_ttl:
            try:
                os.remove(cache_path)  # 删除过期缓存
            except OSError:
                pass
            return None

        try:
            # 检查文件大小是否为0
            if os.path.getsize(cache_path) == 0:
                self.logger.warning(f"警告: 缓存文件 {cache_path} 为空，将删除")
                os.remove(cache_path)
                return None

            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # 验证缓存数据格式
            if not isinstance(cache_data, dict) or "data" not in cache_data:
                self.logger.warning(f"警告: 缓存文件 {cache_path} 格式无效，将删除")
                os.remove(cache_path)
                return None

            self.logger.info(f"✓ Cache loaded: {description}")
            return cache_data["data"]
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            self.logger.warning(f"警告: 缓存文件 {cache_path} 损坏: {e}，将删除")
            try:
                os.remove(cache_path)
            except OSError:
                pass
            return None
        except Exception as e:
            self.logger.warning(f"警告: 加载缓存失败: {e}")
            return None

    def load_data(self, filename="feature_engineered_data.csv"):
        """加载特征工程后的数据，支持分块读取和缓存，优化数据加载效率"""
        file_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(file_path):
            self.logger.error(f"错误: 文件不存在: {file_path}")
            self.logger.error("请先运行特征工程脚本: python feature_engineering.py")
            self.logger.error("特征工程脚本依赖于数据预处理脚本的输出")
            return None

        self.logger.info(f"Loading data: {filename}")
        self.logger.info(f"Large data mode: {'Enabled' if self.large_data_mode else 'Disabled'}")

        # 生成缓存键
        cache_key = self._generate_cache_key({
            "filename": filename,
            "large_data_mode": self.large_data_mode,
            "chunk_size": self.chunk_size,
            "file_modified": os.path.getmtime(file_path)
        })

        # 尝试从缓存加载数据
        cached_data = self._load_cache(cache_key, f"Data loading: {filename}")
        if cached_data is not None:
            self.logger.info(f"Data shape: {cached_data.shape}")
            self.logger.info(f"Memory usage: {cached_data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            return cached_data

        try:
            # 优化数据加载参数
            read_params = {
                'low_memory': False,  # 提高读取大文件的速度
                'infer_datetime_format': True,  # 加速日期解析
                'parse_dates': True,  # 自动解析日期列
            }

            if self.large_data_mode:
                # 大规模数据模式：分块读取，优化分块策略
                chunks = []

                # 不预先计算总块数，减少一次文件扫描
                with tqdm(desc="加载数据块") as pbar:
                    for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, **read_params):
                        chunks.append(chunk)
                        pbar.update(len(chunk))

                data = pd.concat(chunks, ignore_index=True)
            else:
                # 普通模式：一次性读取，优化内存使用
                data = pd.read_csv(file_path, **read_params)

                # 优化内存使用：转换数据类型
                for col in data.columns:
                    if data[col].dtype == 'int64':
                        # 将int64转换为更小的整数类型
                        data[col] = pd.to_numeric(data[col], downcast='integer')
                    elif data[col].dtype == 'float64':
                        # 将float64转换为更小的浮点数类型
                        data[col] = pd.to_numeric(data[col], downcast='float')

            self.logger.info(f"Data shape: {data.shape}")
            self.logger.info(f"Memory usage: {data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

            # 保存到缓存
            self._save_cache(cache_key, data, f"Data loading: {filename}")

            return data
        except Exception as e:
            self.logger.error(f"Error: Failed to load data: {e}")
            self.logger.error("Please check if the file format is correct")
            return None

    def prepare_data(self, data, target='revenue', test_size=0.2, random_state=42):
        """
        准备训练数据和测试数据，支持大规模数据模式下的样本采样优化

        Args:
            data (pd.DataFrame): 输入数据
            target (str): 目标列名
            test_size (float): 测试集比例
            random_state (int): 随机种子

        Returns:
            tuple: (X_train, X_test, y_train, y_test, features) 或 None
        """
        if data is None or len(data) == 0:
            self.logger.error("Error: Input data is empty")
            return None

        # 生成缓存键
        cache_key = self._generate_cache_key({
            "method": "prepare_data",
            "data_shape": data.shape,
            "target": target,
            "test_size": test_size,
            "random_state": random_state,
            "data_hash": hashlib.sha1(data.head(1000).to_string().encode()).hexdigest()
        })

        # 尝试从缓存加载数据
        cached_data = self._load_cache(cache_key, "Data preparation")
        if cached_data is not None:
            X_train, X_test, y_train, y_test, features = cached_data
            self.logger.info(f"Number of features: {len(features)}")
            self.logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            return X_train, X_test, y_train, y_test, features

        self.logger.info("Preparing training and testing data...")

        try:
            # 优化内存使用：减少不必要的列副本
            data = data.copy(deep=False)

            # 如果是大规模数据模式，进行数据采样优化
            if self.large_data_mode:
                # 根据数据大小动态确定采样比例
                max_sample_size = 150000  # 最大采样15万条数据
                if len(data) > max_sample_size:
                    # 使用分层采样，保留目标变量的分布
                    from sklearn.model_selection import StratifiedShuffleSplit

                    # 对目标变量进行分箱，用于分层采样
                    y_bins = pd.cut(data[target], bins=10, labels=False)

                    # 使用分层采样保持数据分布
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=max_sample_size/len(data), random_state=random_state)
                    for train_index, _ in sss.split(data, y_bins):
                        data = data.iloc[train_index]

                    self.logger.info(f"大规模数据模式: 分层采样 {len(data)} 条数据用于训练")

            # 选择特征和目标变量
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            features = [col for col in numeric_cols if col not in [target, 'profit', 'return_on_investment']]

            # 优化内存使用：避免不必要的复制
            X = data[features].copy(deep=False)
            y = data[target].copy()

            # 优化数据类型，减少内存使用
            for col in X.columns:
                if X[col].dtype == 'int64':
                    X[col] = pd.to_numeric(X[col], downcast='integer')
                elif X[col].dtype == 'float64':
                    X[col] = pd.to_numeric(X[col], downcast='float')

            # 划分训练集和测试集
            # 检查release_year列是否存在
            stratify_param = None
            if 'release_year' in data.columns:
                stratify_param = data['release_year']//5
            elif 'release_month' in data.columns:
                stratify_param = data['release_month']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
            )

            # 保存到缓存
            self._save_cache(cache_key, (X_train, X_test, y_train, y_test, features), "数据准备")

            self.logger.info(f"特征数量: {len(features)}")
            self.logger.info(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

            return X_train, X_test, y_train, y_test, features
        except Exception as e:
            self.logger.error(f"错误: 数据准备失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def define_models(self):
        """定义机器学习模型，在大规模数据模式下只保留最有效的模型，增强错误处理"""
        self.logger.info("正在定义模型...")

        try:
            # 从配置管理器获取模型参数
            random_state = global_config.get('traditional_modeling.random_state', 42)
            xgb_params = global_config.get('traditional_modeling.model_params.xgboost', {})
            lgbm_params = global_config.get('traditional_modeling.model_params.lightgbm', {})
            gbm_params = global_config.get('traditional_modeling.model_params.gradient_boosting', {})
            rf_params = global_config.get('traditional_modeling.model_params.random_forest', {})

            # 添加随机种子到所有模型参数
            xgb_params['random_state'] = random_state
            lgbm_params['random_state'] = random_state
            gbm_params['random_state'] = random_state
            rf_params['random_state'] = random_state

            # 基础模型配置 - 使用配置参数，让配置参数优先
            # 设置默认参数，只有在配置参数中不存在时才使用
            xgb_defaults = {'n_jobs': -1, 'verbose': 0}
            for key, value in xgb_defaults.items():
                if key not in xgb_params:
                    xgb_params[key] = value
            base_xgb = XGBRegressor(**xgb_params)

            lgbm_defaults = {'n_jobs': -1, 'verbose': -1}
            for key, value in lgbm_defaults.items():
                if key not in lgbm_params:
                    lgbm_params[key] = value
            base_lgbm = LGBMRegressor(**lgbm_params)

            gbm_defaults = {'n_iter_no_change': 15, 'tol': 0.001}
            for key, value in gbm_defaults.items():
                if key not in gbm_params:
                    gbm_params[key] = value
            base_gbm = GradientBoostingRegressor(**gbm_params)

            rf_defaults = {'n_jobs': -1, 'bootstrap': True, 'oob_score': True}
            for key, value in rf_defaults.items():
                if key not in rf_params:
                    rf_params[key] = value
            base_rf = RandomForestRegressor(**rf_params)

            # 获取要使用的模型列表
            use_models = global_config.get('traditional_modeling.models', {})

            if self.large_data_mode:
                # 大规模数据模式：只使用高效的梯度提升模型
                models = {
                    'XGBoost': base_xgb,
                    'LightGBM': base_lgbm,
                    'Ridge Regression': Ridge(random_state=random_state, alpha=1.0)
                }
            else:
                # 普通模式：根据配置使用多种模型
                models = {}
                if use_models.get('linear_regression', True):
                    models['Linear Regression'] = LinearRegression()
                if use_models.get('ridge_regression', True):
                    models['Ridge Regression'] = Ridge(random_state=random_state, alpha=1.0)
                if use_models.get('random_forest', True):
                    models['Random Forest'] = base_rf
                if use_models.get('gradient_boosting', True):
                    models['Gradient Boosting'] = base_gbm
                if use_models.get('xgboost', True):
                    models['XGBoost'] = base_xgb
                if use_models.get('lightgbm', True):
                    models['LightGBM'] = base_lgbm

            self.logger.info(f"定义的模型数量: {len(models)}")
            return models
        except ImportError as e:
            self.logger.error(f"缺少模型依赖库: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"定义模型失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, models):
        """训练和评估多种模型，添加进度监控和内存优化"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("模型训练和评估")
        self.logger.info("=" * 50)

        results = []
        trained_models = {}

        for name, model in tqdm(models.items(), desc="训练和评估模型"):
            try:
                self.logger.info(f"\n正在训练: {name}")

                # 创建管道并训练模型
                pipeline, training_time = self._train_model(name, model, X_train, y_train)

                # 预测和评估
                y_pred = pipeline.predict(X_test)
                metrics = self._evaluate_predictions(y_test, y_pred)
                metrics['Training Time (s)'] = training_time

                # 保存结果
                results.append({
                    'Model': name,
                    **metrics
                })

                trained_models[name] = pipeline

                # 输出评估结果
                self._print_evaluation_results(name, metrics, training_time)

                # 特征重要性分析
                self.get_feature_importance(name, pipeline, X_train)

                # 模型解释分析
                self.explain_model(name, pipeline, X_train, X_test, y_train, y_test)

                # 保存训练好的模型
                self.save_model(pipeline, name, metrics=metrics)
            except Exception as e:
                self.logger.error(f"训练 {name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 结果排序
        if results:
            results_df = pd.DataFrame(results)
            # 确保'R2 Score'列存在
            if 'R2 Score' in results_df.columns:
                results_df = results_df.sort_values('R2 Score', ascending=False)
            return results_df, trained_models
        else:
            return pd.DataFrame(), trained_models

    def _train_model(self, model_name, model, X_train, y_train):
        """
        训练单个模型并返回训练好的管道和训练时间

        Args:
            model_name (str): 模型名称
            model: 模型对象
            X_train: 训练特征
            y_train: 训练目标

        Returns:
            tuple: (训练好的管道, 训练时间)
        """
        start_time = time.time()

        # 创建管道，添加SimpleImputer处理NaN值
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # 根据模型类型选择训练方式
        if model_name in ["XGBoost", "LightGBM"]:
            # 这些模型支持早期停止
            try:
                # 尝试使用通用参数格式
                pipeline.fit(X_train, y_train, model__eval_set=[(X_train, y_train)],
                             model__early_stopping_rounds=100, model__verbose=0)
            except TypeError:
                # 如果参数格式不支持，使用基础训练
                pipeline.fit(X_train, y_train)
        else:
            pipeline.fit(X_train, y_train)

        training_time = time.time() - start_time
        return pipeline, training_time

    def _evaluate_predictions(self, y_test, y_pred):
        """
        评估模型预测结果

        Args:
            y_test: 真实值
            y_pred: 预测值

        Returns:
            dict: 评估指标
        """
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2 Score': r2
        }

    def _print_evaluation_results(self, model_name, metrics, training_time):
        """
        打印模型评估结果

        Args:
            model_name (str): 模型名称
            metrics (dict): 评估指标
            training_time (float): 训练时间
        """
        self.logger.info(f"{model_name} 评估结果:")
        self.logger.info(f"  训练时间: {training_time:.2f} 秒")
        self.logger.info(f"  RMSE: {metrics['RMSE']:.2f}")
        self.logger.info(f"  MAE: {metrics['MAE']:.2f}")
        self.logger.info(f"  MAPE: {metrics['MAPE']:.4f}")
        self.logger.info(f"  R2 Score: {metrics['R2 Score']:.4f}")

    def get_feature_importance(self, model_name, pipeline, X_train):
        """
        获取并分析模型的特征重要性

        Args:
            model_name (str): 模型名称
            pipeline (Pipeline): 训练好的管道模型
            X_train (pd.DataFrame): 训练数据特征
        """
        try:
            # 获取管道中的模型
            model = pipeline.named_steps['model']
            feature_names = X_train.columns.tolist()

            self.logger.info(f"\n{model_name} Feature Importance Analysis:")

            if hasattr(model, 'feature_importances_'):
                # 树模型的特征重要性
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_importance = feature_importance.sort_values('Importance', ascending=False)

                # 打印前10个重要特征
                self.logger.info("\nTop 10 important features:")
                top_10 = feature_importance.head(10)
                for index, row in top_10.iterrows():
                    self.logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")

                # 可视化前20个重要特征
                self._plot_feature_importance(model_name, feature_importance.head(20))

                return feature_importance

            elif hasattr(model, 'coef_'):
                # 线性模型的特征重要性
                coefs = model.coef_
                if len(coefs.shape) > 1:
                    # 多输出模型
                    coefs = np.mean(np.abs(coefs), axis=0)
                feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefs)})
                feature_importance = feature_importance.sort_values('Importance', ascending=False)

                # 打印前10个重要特征
                self.logger.info("\n前10个重要特征:")
                top_10 = feature_importance.head(10)
                for index, row in top_10.iterrows():
                    self.logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")

                # 可视化前20个重要特征
                self._plot_feature_importance(model_name, feature_importance.head(20))

                return feature_importance

            else:
                self.logger.info("\n该模型不支持特征重要性分析")
                return None

        except Exception as e:
            print(f"错误: 获取特征重要性失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_feature_importance(self, model_name, feature_importance):
        """
        可视化特征重要性

        Args:
            model_name (str): 模型名称
            feature_importance (pd.DataFrame): 特征重要性数据
        """
        try:
            # 删除旧图片
            explain_dir = os.path.join(self.results_dir, "model_explanations")
            old_files = [f for f in os.listdir(self.results_dir) if f.endswith('_feature_importance.png')]
            for old_file in old_files:
                os.remove(os.path.join(self.results_dir, old_file))

            # 删除旧的SHAP图片
            if os.path.exists(explain_dir):
                shap_files = [f for f in os.listdir(explain_dir) if f.endswith('.png')]
                for shap_file in shap_files:
                    os.remove(os.path.join(explain_dir, shap_file))

            plt.figure(figsize=(15, 10))
            plt.barh(range(len(feature_importance)), feature_importance['Importance'], align='center')
            plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
            plt.gca().invert_yaxis()
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'{model_name} - Feature Importance (Top {len(feature_importance)})')
            plt.tight_layout()

            # 保存图片
            os.makedirs(self.results_dir, exist_ok=True)
            filename = os.path.join(self.results_dir, f'{model_name}_feature_importance.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\nFeature importance plot saved to: {filename}")

        except Exception as e:
            print(f"错误: 绘制特征重要性失败: {e}")
            import traceback
            traceback.print_exc()

    def explain_model(self, model_name, pipeline, X_train, X_test, y_train, y_test):
        """
        使用SHAP和LIME进行模型解释

        Args:
            model_name (str): 模型名称
            pipeline (Pipeline): 训练好的管道模型
            X_train (pd.DataFrame): 训练数据特征
            X_test (pd.DataFrame): 测试数据特征
            y_train (pd.Series): 训练数据目标
            y_test (pd.Series): 测试数据目标
        """
        try:
            self.logger.info(f"\n{model_name} Model Explanation Analysis:")

            # 创建保存目录
            explain_dir = os.path.join(self.results_dir, "model_explanations")
            os.makedirs(explain_dir, exist_ok=True)

            # 获取管道中的模型
            model = pipeline.named_steps['model']
            feature_names = X_train.columns.tolist()

            # 小样本用于解释分析
            sample_size = min(100, len(X_test))
            X_sample = X_test.head(sample_size)

            # 1. SHAP分析
            print("\n1. Starting SHAP analysis...")
            try:
                # 初始化SHAP解释器
                explainer = shap.Explainer(model, X_train)

                # 计算SHAP值
                shap_values = explainer(X_sample)

                # 保存SHAP值
                shap_file = os.path.join(explain_dir, f"{model_name}_shap_values.npy")
                np.save(shap_file, shap_values.values)
                print(f"SHAP values saved to: {shap_file}")

                # 全局特征重要性 - 使用bar plot，更加清晰
                plt.figure(figsize=(14, 10))
                shap.summary_plot(shap_values.values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
                plt.tight_layout()
                shap_global_file = os.path.join(explain_dir, f"{model_name}_shap_global.png")
                plt.savefig(shap_global_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Global SHAP plot saved to: {shap_global_file}")

                # 使用summary_plot的另一种形式
                plt.figure(figsize=(14, 10))
                shap.summary_plot(shap_values.values, X_sample, feature_names=feature_names, show=False)
                plt.tight_layout()
                shap_summary_file = os.path.join(explain_dir, f"{model_name}_shap_summary.png")
                plt.savefig(shap_summary_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"SHAP summary plot saved to: {shap_summary_file}")

                # 局部解释（前1个样本，避免生成过多图片）
                for i in range(min(1, len(X_sample))):
                    # 使用force plot替代waterfall plot，更加直观
                    plt.figure(figsize=(16, 10))
                    shap.plots.force(shap_values[i], show=False)
                    plt.tight_layout()
                    shap_local_file = os.path.join(explain_dir, f"{model_name}_shap_local_{i}.png")
                    plt.savefig(shap_local_file, dpi=300, bbox_inches='tight')
                    plt.close()
                print("Local SHAP explanation plots saved")

            except Exception as e:
                print(f"Error: SHAP analysis failed: {e}")
                traceback.print_exc()

            # 2. LIME分析
            print("\n2. 开始LIME分析...")
            if MODEL_INTERPRETABILITY_ENABLED:
                try:
                    # 初始化LIME解释器
                    explainer = lime_tabular.LimeTabularExplainer(
                        training_data=X_train.values,
                        feature_names=feature_names,
                        mode='regression',
                        verbose=False
                    )

                    # 预测函数
                    predict_fn = pipeline.predict

                    # 生成前3个样本的解释
                    for i in range(min(3, len(X_sample))):
                        sample = X_sample.iloc[i].values
                        exp = explainer.explain_instance(
                            data_row=sample,
                            predict_fn=predict_fn,
                            num_features=10
                        )

                        # 保存LIME解释
                        exp.as_pyplot_figure()
                        plt.tight_layout()
                        lime_file = os.path.join(explain_dir, f"{model_name}_lime_{i}.png")
                        plt.savefig(lime_file, dpi=300, bbox_inches='tight')
                        plt.close()

                        # 保存LIME数据 - 安全处理，避免索引越界
                        lime_list = exp.as_list()
                        max_features = min(10, len(lime_list))
                        lime_data = pd.DataFrame({
                            'feature': [lime_list[j][0] for j in range(max_features)],
                            'weight': [lime_list[j][1] for j in range(max_features)]
                        })
                        lime_data_file = os.path.join(explain_dir, f"{model_name}_lime_{i}.csv")
                        lime_data.to_csv(lime_data_file, index=False)

                    print("LIME解释图已保存")

                except Exception as e:
                    print(f"错误: LIME分析失败: {e}")
                    traceback.print_exc()
            else:
                print("LIME分析: 跳过，依赖库未安装")

            self.logger.info(f"\n{model_name} 模型解释完成")
            return True

        except Exception as e:
            print(f"错误: 模型解释失败: {e}")
            traceback.print_exc()
            return False

    def hyperparameter_tuning(self, X_train, y_train, model, param_grid, model_name, cv=3, n_iter=50):
        """超参数调优，在大规模数据模式下减少迭代次数和交叉验证折数"""
        self.logger.info(f"\n正在进行{model_name}超参数调优...")
        print(f"参数: 迭代次数={n_iter}, 交叉验证折数={cv}")

        # 减少计算量：对所有数据集都进行采样处理
        max_sample_size = 50000  # 最大样本数
        if len(X_train) > max_sample_size:
            sample_size = min(max_sample_size, len(X_train) // 2)
            print(f"数据集较大 ({len(X_train)} 样本)，使用{sample_size}个样本进行调优")
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_train_sample = X_train.iloc[idx]
            y_train_sample = y_train.iloc[idx]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
            print(f"使用全部 {len(X_train)} 个样本进行调优")

        # 创建管道，添加SimpleImputer处理NaN值
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # 使用随机搜索进行调优
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=2
        )

        start_time = time.time()
        random_search.fit(X_train_sample, y_train_sample)
        self.logger.info(f"调优耗时: {time.time() - start_time:.2f} 秒")

        self.logger.info(f"最佳参数: {random_search.best_params_}")
        self.logger.info(f"最佳交叉验证分数: {random_search.best_score_:.4f}")

        return random_search.best_estimator_

    def optimize_hyperparameters(self, X_train, y_train, model_name, cv=5, n_trials=100):
        """使用Optuna进行超参数优化

        Args:
            X_train: 训练特征
            y_train: 训练目标
            model_name: 模型名称
            cv: 交叉验证折数
            n_trials: 调优迭代次数

        Returns:
            调优后的最佳模型
        """
        self.logger.info(f"\n正在使用Optuna进行{model_name}超参数优化...")

        # 大规模数据模式下减少计算量
        if self.large_data_mode:
            n_trials = min(n_trials, 50)  # 最多50次迭代
            cv = min(cv, 5)  # 最多5折交叉验证

            # 如果数据集仍然很大，使用更小的样本进行调优
            if len(X_train) > 100000:
                sample_size = min(100000, len(X_train) // 2)
                print(f"数据集较大，使用{sample_size}个样本进行调优")
                idx = np.random.choice(len(X_train), sample_size, replace=False)
                X_train_sample = X_train.iloc[idx]
                y_train_sample = y_train.iloc[idx]
            else:
                X_train_sample = X_train
                y_train_sample = y_train
        else:
            X_train_sample = X_train
            y_train_sample = y_train

        # 定义Optuna目标函数
        def objective(trial):
            """Optuna目标函数"""
            # 根据模型名称定义参数空间
            if model_name == 'Random Forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=200),
                    'max_depth': trial.suggest_int('max_depth', 5, 100, step=5),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 100, step=5),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50, step=2),
                    'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 1.0, step=0.05)
                }
                model = RandomForestRegressor(**params, random_state=42)
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 3000, step=200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 30, step=2),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 50, step=2),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
                    'gamma': trial.suggest_float('gamma', 0.0, 20.0, step=0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 100.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 100.0, log=True),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0, log=True)
                }
                model = XGBRegressor(**params, random_state=42, objective='reg:squarederror')
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 3000, step=200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', -1, 30, step=2),
                    'num_leaves': trial.suggest_int('num_leaves', 16, 1024, step=32),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 1000, step=20),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 100.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 100.0, log=True),
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                    'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 10.0, step=0.1),
                    'max_bin': trial.suggest_int('max_bin', 255, 1000, step=50)
                }
                model = LGBMRegressor(**params, random_state=42)
            elif model_name == 'Gradient Boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 30, step=2),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 100, step=5),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50, step=2),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
                    'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
                    'alpha': trial.suggest_float('alpha', 0.0, 1.0, step=0.05)
                }
                model = GradientBoostingRegressor(**params, random_state=42)
            elif model_name == 'Ridge Regression':
                params = {
                    'alpha': trial.suggest_float('alpha', 0.001, 10000.0, log=True),
                    'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
                }
                model = Ridge(**params, random_state=42)
            else:
                raise ValueError(f"不支持的模型名称: {model_name}")

            # 创建管道
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            # 交叉验证
            scores = cross_val_score(pipeline, X_train_sample, y_train_sample, cv=cv, scoring='r2', n_jobs=-1)
            return scores.mean()

        # 创建并运行Optuna study
        study = optuna.create_study(direction='maximize', random_state=42)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)

        # 获取最佳参数
        best_params = study.best_params
        self.logger.info("Optuna优化完成！")
        self.logger.info(f"最佳参数: {best_params}")
        self.logger.info(f"最佳交叉验证分数: {study.best_value:.4f}")

        # 训练最佳模型
        best_model = self._create_model_from_params(best_params, model_name)

        # 创建管道
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', best_model)
        ])

        # 训练完整模型
        self.logger.info(f"正在使用最佳参数训练{model_name}模型...")
        pipeline.fit(X_train, y_train)

        return pipeline

    def _create_model_from_params(self, params, model_name):
        """根据参数创建模型实例

        Args:
            params: 模型参数
            model_name: 模型名称

        Returns:
            模型实例
        """
        if model_name == 'Random Forest':
            return RandomForestRegressor(**params, random_state=42)
        elif model_name == 'XGBoost':
            return XGBRegressor(**params, random_state=42, objective='reg:squarederror')
        elif model_name == 'LightGBM':
            return LGBMRegressor(**params, random_state=42)
        elif model_name == 'Gradient Boosting':
            return GradientBoostingRegressor(**params, random_state=42)
        elif model_name == 'Ridge Regression':
            return Ridge(**params, random_state=42)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")

    def create_stacking_ensemble(self, X_train, X_test, y_train, y_test, base_models_dict):
        """创建Stacking集成模型

        Args:
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练目标
            y_test: 测试目标
            base_models_dict: 基础模型字典，格式为 {model_name: model_instance}

        Returns:
            集成模型和评估指标
        """
        print("\n正在创建Stacking集成模型...")

        # 定义元模型
        meta_model = LinearRegression()

        # 创建Stacking集成模型
        stacking_regressor = StackingRegressor(
            estimators=list(base_models_dict.items()),
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )

        # 训练集成模型
        stacking_regressor.fit(X_train, y_train)

        # 评估模型
        metrics, y_test_inv, y_pred_inv = self.evaluate_model(stacking_regressor, X_test, y_test)

        # 显示评估结果
        print("\nStacking集成模型评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # 绘制实际值与预测值对比图
        self.plot_actual_vs_predicted(y_test_inv, y_pred_inv, "Stacking Ensemble")

        # 保存集成模型
        self.save_model(stacking_regressor, "Stacking Ensemble", metrics, is_best=True)

        return stacking_regressor, metrics

    def get_param_grids(self):
        """获取不同模型的参数网格，在大规模数据模式下简化参数网格"""
        param_grids = {
            'Random Forest': {
                'model__n_estimators': [100, 200, 300, 500],
                'model__max_depth': [None, 10, 20, 30, 50],
                'model__min_samples_split': [2, 5, 10, 20],
                'model__min_samples_leaf': [1, 2, 5, 10],
                'model__max_features': ['auto', 'sqrt', 'log2', None],
                'model__bootstrap': [True, False],
                'model__oob_score': [True, False]
            },
            'XGBoost': {
                'model__n_estimators': [200, 300, 500, 700, 1000],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'model__max_depth': [3, 5, 7, 9, 12],
                'model__min_child_weight': [1, 3, 5, 7, 10],
                'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'model__gamma': [0, 0.1, 0.2, 0.3, 0.5],
                'model__reg_alpha': [0, 0.1, 0.5, 1.0, 5.0, 10.0],
                'model__reg_lambda': [0, 0.1, 0.5, 1.0, 5.0, 10.0],
                'model__scale_pos_weight': [1, 2, 3, 5]
            },
            'LightGBM': {
                'model__n_estimators': [200, 300, 500, 700, 1000],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'model__max_depth': [-1, 5, 7, 9, 12, 15],
                'model__num_leaves': [31, 63, 127, 255, 511],
                'model__min_data_in_leaf': [10, 20, 50, 100, 200],
                'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'model__reg_alpha': [0, 0.1, 0.5, 1.0, 5.0],
                'model__reg_lambda': [0, 0.1, 0.5, 1.0, 5.0],
                'model__boosting_type': ['gbdt', 'dart', 'goss']
            },
            'Gradient Boosting': {
                'model__n_estimators': [200, 300, 500, 700],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'model__max_depth': [3, 5, 7, 9, 12],
                'model__min_samples_split': [2, 5, 10, 20],
                'model__min_samples_leaf': [1, 2, 5, 10],
                'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'model__max_features': ['auto', 'sqrt', 'log2', None]
            },
            'Ridge Regression': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            }
        }

        return param_grids

    def compare_models(self, results_df):
        """比较不同模型的性能"""
        print("\n" + "=" * 50)
        print("Model Performance Comparison")
        print("=" * 50)

        # 显示排序后的结果
        print(results_df.to_string(index=False))

        # 删除旧的比较图
        old_comparison_files = [f for f in os.listdir(self.charts_dir) if f.startswith('model_comparison_')]
        for old_file in old_comparison_files:
            old_file_path = os.path.join(self.charts_dir, old_file)
            try:
                os.remove(old_file_path)
                self.logger.info(f"Removed old comparison plot: {old_file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old comparison plot: {e}")

        # 可视化比较结果
        # 1. R2 Score比较
        plt.figure(figsize=(14, 8))
        sns.barplot(x='R2 Score', y='Model', data=results_df, palette='viridis')
        plt.title('Model Performance Comparison (R2 Score)', fontsize=16)
        plt.xlabel('R2 Score', fontsize=14)
        plt.ylabel('Model', fontsize=14)
        plt.xlim(0, 1)
        plt.savefig(os.path.join(self.charts_dir, 'model_comparison_r2.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("R2 Score comparison plot saved")

        # 2. RMSE比较
        plt.figure(figsize=(14, 8))
        results_df_sorted = results_df.sort_values('RMSE')
        sns.barplot(x='RMSE', y='Model', data=results_df_sorted, palette='plasma')
        plt.title('Model Performance Comparison (RMSE)', fontsize=16)
        plt.xlabel('RMSE', fontsize=14)
        plt.ylabel('Model', fontsize=14)
        plt.savefig(os.path.join(self.charts_dir, 'model_comparison_rmse.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("RMSE comparison plot saved")

        return results_df

    def feature_importance(self, model, features, model_name):
        """分析特征重要性"""
        self.logger.info(f"\nAnalyzing feature importance for {model_name}...")

        # 获取模型的特征重要性
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            importances = model.named_steps['model'].feature_importances_
        elif hasattr(model.named_steps['model'], 'coef_'):
            importances = np.abs(model.named_steps['model'].coef_)
        else:
            self.logger.info(f"{model_name} does not support feature importance analysis")
            return None

        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        })
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        # 可视化特征重要性
        plt.figure(figsize=(14, 10))
        top_features = feature_importance_df.head(20)
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='coolwarm')
        plt.title(f'{model_name} Feature Importance', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        # 确保每次都生成新图片，不使用缓存
        plt.clf()
        plt.figure(figsize=(14, 10))
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='coolwarm')
        plt.title(f'{model_name} Feature Importance', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.savefig(os.path.join(self.charts_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"{model_name} feature importance plot saved")

    def analyze_model_with_shap(self, model, X_test, features, model_name):
        """使用SHAP分析模型

        Args:
            model: 训练好的模型
            X_test: 测试数据
            features: 特征名称列表
            model_name: 模型名称

        Returns:
            SHAP值和特征重要性
        """
        self.logger.info(f"\nUsing SHAP to analyze {model_name} model...")

        # 获取管道中的基础模型
        base_model = model.named_steps['model']

        # 为不同类型的模型创建SHAP解释器
        try:
            if hasattr(base_model, 'predict_proba'):
                # 分类模型
                explainer = shap.TreeExplainer(base_model)
            else:
                # 回归模型
                explainer = shap.Explainer(base_model)
        except Exception as e:
            self.logger.error(f"Failed to create SHAP explainer: {e}")
            self.logger.warning(f"{model_name} may not support SHAP analysis")
            return None

        # 获取特征的原始值（已经过预处理）
        # 注意：SHAP需要的是模型输入之前的数据
        # 我们需要从管道中提取imputer和scaler，然后处理数据
        X_processed = model.named_steps['imputer'].transform(X_test)
        X_processed = model.named_steps['scaler'].transform(X_processed)

        # 计算SHAP值
        try:
            shap_values = explainer(X_processed)
        except Exception as e:
            self.logger.error(f"Failed to calculate SHAP values: {e}")
            return None

        # 可视化SHAP摘要图
        self.logger.info("Drawing SHAP summary plot...")
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_processed, feature_names=features, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance for {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, f'{model_name.lower().replace(" ", "_")}_shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"{model_name} SHAP summary plot saved")

        # 绘制SHAP依赖图（选择最重要的特征）
        self.logger.info("Drawing SHAP dependence plots...")
        # 获取前5个最重要的特征
        if hasattr(base_model, 'feature_importances_'):
            top_features = np.argsort(base_model.feature_importances_)[-5:][::-1]
        elif hasattr(base_model, 'coef_'):
            top_features = np.argsort(np.abs(base_model.coef_))[-5:][::-1]
        else:
            # 默认使用前5个特征
            top_features = range(min(5, len(features)))

        # 为每个重要特征绘制依赖图
        for i in top_features:
            feature_name = features[i]
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(i, shap_values.values, X_processed, feature_names=features, show=False)
            plt.title(f'SHAP Dependence Plot for {feature_name} ({model_name})', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, f'{model_name.lower().replace(" ", "_")}_shap_dependence_{feature_name.lower().replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"{model_name} SHAP dependence plot saved for feature: {feature_name}")

        # 绘制SHAP力图（示例）
        self.logger.info("Drawing SHAP force plot...")
        # 使用第一个样本绘制力图
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value[0] if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
            shap_values.values[0],
            X_test.iloc[0],
            feature_names=features
        )

        # 保存力图为HTML
        force_plot_path = os.path.join(self.charts_dir, f'{model_name.lower().replace(" ", "_")}_shap_force_plot.html')
        shap.save_html(force_plot_path, force_plot)
        self.logger.info(f"{model_name} SHAP force plot saved")

        return shap_values

    def save_model(self, model, model_name, metrics=None, is_best=False):
        """保存训练好的模型，只保留最优模型

        Args:
            model: 训练好的模型
            model_name: 模型名称
            metrics: 模型性能指标字典，用于比较模型优劣
            is_best: 是否为最佳模型

        Returns:
            str: 保存的模型路径
        """
        # 1. 保存当前模型
        model_path = os.path.join(self.models_dir, f'{model_name.lower().replace(" ", "_")}_model.joblib')
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved to: {model_path}")

        # 2. 如果是最佳模型，额外保存为best模型
        if is_best:
            best_model_path = os.path.join(self.models_dir, f'{model_name.lower().replace(" ", "_")}_best.joblib')
            joblib.dump(model, best_model_path)
            self.logger.info(f"Best {model_name} model saved to: {best_model_path}")

            # 3. 清理旧的非最佳模型文件
            self._clean_old_models(model_name)

        return model_path

    def _clean_old_models(self, model_name):
        """清理旧的非最佳模型文件

        Args:
            model_name: 模型名称
        """
        import glob

        # 构建模型名称前缀
        model_prefix = model_name.lower().replace(" ", "_")

        # 查找所有该模型的文件，排除最佳模型
        model_files = glob.glob(os.path.join(self.models_dir, f'{model_prefix}_*.joblib'))
        best_model_path = os.path.join(self.models_dir, f'{model_prefix}_best.joblib')

        # 删除非最佳模型文件
        for file_path in model_files:
            if file_path != best_model_path:
                try:
                    os.remove(file_path)
                    self.logger.info(f"Removed old model file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove old model file: {e}")

    def load_model(self, model_name):
        """加载训练好的模型"""
        model_path = os.path.join(self.models_dir, f'{model_name.lower().replace(" ", "_")}_model.joblib')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            self.logger.info(f"Model loaded from: {model_path}")
            return model
        else:
            self.logger.error(f"Model file not found: {model_path}")
            return None

    def evaluate_model(self, model, X_test, y_test):
        """评估单个模型"""
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2 Score': r2
        }

        return metrics, y_test, y_pred

    def plot_actual_vs_predicted(self, y_test, y_pred, model_name):
        """绘制实际值与预测值的对比图"""
        # 确保每次都生成新图片，不使用缓存
        plt.clf()
        plt.figure(figsize=(12, 12))
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Revenue', fontsize=14)
        plt.ylabel('Predicted Revenue', fontsize=14)
        plt.title(f'Actual vs Predicted Revenue ({model_name})', fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        # 添加评估指标文本
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR2 Score: {r2:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'))

        plt.savefig(os.path.join(self.charts_dir, f'{model_name.lower().replace(" ", "_")}_actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"{model_name} actual vs predicted plot saved")

    def run_complete_modeling(self, filename="feature_engineered_data.csv", target='revenue'):
        """运行完整的建模流程"""
        self.logger.info("=" * 60)
        self.logger.info("开始完整的机器学习建模流程")
        self.logger.info("=" * 60)

        # 1. 加载数据
        data = self.load_data(filename)
        if data is None:
            return None

        # 2. 准备数据
        X_train, X_test, y_train, y_test, features = self.prepare_data(data, target)

        # 3. 定义模型
        models = self.define_models()

        # 4. 训练和评估模型
        results_df, trained_models = self.train_and_evaluate(X_train, X_test, y_train, y_test, models)

        # 5. 比较模型
        self.compare_models(results_df)

        # 6. 超参数调优（对表现最好的几个模型）
        param_grids = self.get_param_grids()
        top_models = results_df.head(2)['Model'].tolist()  # 只对表现最好的2个模型进行调优

        tuned_models = {}
        for i, model_name in enumerate(top_models):
            if model_name in param_grids:
                self.logger.info("\n" + "=" * 50)
                self.logger.info(f"对{model_name}进行超参数调优 ({i+1}/{len(top_models)})")
                self.logger.info("=" * 50)

                best_model = self.hyperparameter_tuning(
                    X_train, y_train, 
                    models[model_name], 
                    param_grids[model_name], 
                    model_name,
                    n_iter=20,  # 进一步减少迭代次数
                    cv=3
                )

                # 评估调优后的模型
                metrics, _, y_pred = self.evaluate_model(best_model, X_test, y_test)
                self.logger.info(f"\n调优后{model_name}评估结果:")
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric}: {value:.4f}")

                # 绘制实际值与预测值对比图
                self.plot_actual_vs_predicted(y_test, y_pred, model_name)

                # 分析特征重要性
                self.feature_importance(best_model, features, model_name)

                # 保存调优后的模型，标记为最佳模型
                self.save_model(best_model, model_name, metrics=metrics, is_best=True)

                tuned_models[model_name] = best_model

        self.logger.info("\n" + "=" * 60)
        self.logger.info("机器学习建模流程完成！")
        self.logger.info("=" * 60)

        return results_df, tuned_models

def main():
    """主函数，执行完整的建模流程"""
    import argparse

    parser = argparse.ArgumentParser(description='机器学习建模脚本')
    parser.add_argument('--large-data', action='store_true', help='启用大规模数据模式')
    parser.add_argument('--chunk-size', type=int, default=100000, help='分块读取的大小')
    parser.add_argument('--filename', type=str, default='feature_engineered_data.csv', help='特征工程后的数据文件名')
    parser.add_argument('--target', type=str, default='revenue', help='目标变量名称')
    parser.add_argument('--no-cache', action='store_true', help='禁用缓存')
    parser.add_argument('--cache-ttl', type=int, default=3600, help='缓存过期时间（秒）')

    # 新增参数：控制模型训练
    parser.add_argument('--n-iter', type=int, default=20, help='超参数调优的迭代次数')
    parser.add_argument('--cv', type=int, default=3, help='交叉验证折数')
    parser.add_argument('--top-models', type=int, default=2, help='要进行超参数调优的最佳模型数量')
    parser.add_argument('--model-names', type=str, nargs='+', help='要训练的模型名称列表，默认训练所有模型')

    args = parser.parse_args()

    # 创建建模实例
    modeling = TraditionalModeling(
        large_data_mode=args.large_data,
        chunk_size=args.chunk_size,
        cache_enabled=not args.no_cache,
        cache_ttl=args.cache_ttl
    )

    # 运行完整建模流程
    modeling.run_complete_modeling(filename=args.filename, target=args.target)

if __name__ == "__main__":
    main()