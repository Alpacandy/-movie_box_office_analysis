import os
import sys

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入标准库和第三方库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import joblib
from sklearn.model_selection import train_test_split
import warnings

# 导入本地模块
from src.utils.logging_config import get_logger

# 获取当前脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录
project_root = os.path.abspath(os.path.join(script_dir, ".."))

warnings.filterwarnings('ignore')

# 初始化日志记录器
logger = get_logger('model_interpretability')

# 设置可视化风格
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class ModelInterpretability:
    def __init__(self, base_dir=None, results_dir=None):
        self.base_dir = os.path.join(project_root, "data") if base_dir is None else base_dir
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.results_dir = os.path.join(project_root, "results") if results_dir is None else results_dir
        self.models_dir = os.path.join(self.results_dir, "models")
        self.charts_dir = os.path.join(self.results_dir, "charts")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)

        self.model = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None

    def load_data(self, filename="feature_engineered_data.csv"):
        """加载特征工程后的数据"""
        file_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            logger.error("请先运行特征工程脚本: python feature_engineering.py")
            return False

        logger.info(f"正在加载数据: {filename}")
        self.data = pd.read_csv(file_path)
        logger.info(f"数据形状: {self.data.shape}")
        return True

    def load_model(self, model_name="random_forest"):
        """加载训练好的模型"""
        model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            logger.error("请先运行建模脚本: python modeling.py")
            return False

        logger.info(f"正在加载模型: {model_name}")
        self.model = joblib.load(model_path)
        logger.info(f"成功加载{model_name}模型")
        return True

    def prepare_data(self, target='revenue'):
        """准备训练集和测试集"""
        if self.data is None:
            logger.error("请先加载数据")
            return False

        # 选择特征和目标变量
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.features = [col for col in numeric_cols if col not in [target, 'profit', 'return_on_investment']]

        X = self.data[self.features]
        y = self.data[target]

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"训练集样本数: {len(self.X_train)}, 测试集样本数: {len(self.X_test)}")
        logger.info(f"特征数量: {len(self.features)}")
        return True

    def shap_analysis(self, sample_size=100):
        """使用SHAP进行模型解释"""
        if self.model is None or self.X_test is None:
            logger.error("请先加载模型和数据")
            return False

        logger.info("\n开始SHAP分析...")

        # 使用小样本进行分析，提高性能
        X_sample = self.X_test.head(sample_size)

        # 初始化SHAP解释器
        if hasattr(self.model, 'named_steps'):  # 处理Pipeline模型
            model = self.model.named_steps['model']
        else:
            model = self.model

        try:
            # 初始化SHAP解释器
            if hasattr(model, 'predict_proba'):  # 分类模型
                explainer = shap.Explainer(model, self.X_train)
            else:  # 回归模型
                explainer = shap.Explainer(model, self.X_train)

            # 计算SHAP值
            shap_values = explainer(X_sample)

            # 保存SHAP值
            shap_values_path = os.path.join(self.processed_dir, "shap_values.npy")
            np.save(shap_values_path, shap_values.values)
            logger.info(f"SHAP值已保存到: {shap_values_path}")

            # 1. 全局特征重要性
            logger.info("\n1. 生成全局特征重要性图...")
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values.values, X_sample, feature_names=self.features, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, "shap_global_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("全局特征重要性图已保存")

            # 2. 局部解释（前5个样本）
            logger.info("\n2. 生成局部解释图...")
            for i in range(min(5, len(X_sample))):
                plt.figure(figsize=(14, 10))
                shap.plots.waterfall(shap_values[i], show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.charts_dir, f"shap_local_{i}.png"), dpi=300, bbox_inches='tight')
                plt.close()
            logger.info("局部解释图已保存")

            # 3. 依赖图（选择前2个最重要的特征）
            logger.info("\n3. 生成特征依赖图...")
            top_features = np.abs(shap_values.values).mean(axis=0).argsort()[-2:]
            for i in top_features:
                plt.figure(figsize=(14, 8))
                shap.dependence_plot(shap_values.feature_names[i], shap_values.values, X_sample, show=False)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.charts_dir, f"shap_dependence_{shap_values.feature_names[i]}.png"),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
            logger.info("特征依赖图已保存")

            logger.info("\nSHAP分析完成！")
            return True

        except (TypeError, ValueError, AttributeError) as e:
            logger.error(f"SHAP分析失败: {e}")
            logger.error("请确保模型支持SHAP分析")
            return False

    def lime_analysis(self, sample_idx=0, num_features=10):
        """使用LIME进行模型解释"""
        if self.model is None or self.X_test is None:
            logger.error("请先加载模型和数据")
            return False

        logger.info("\n开始LIME分析...")

        # 初始化LIME解释器
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.features,
            mode='regression',
            verbose=False
        )

        # 选择一个样本进行解释
        sample = self.X_test.iloc[sample_idx].values

        # 预测函数
        def predict_fn_proba(x):
            return self.model.predict_proba(x)[:, 1]

        def predict_fn(x):
            return self.model.predict(x)

        if hasattr(self.model, 'predict_proba'):
            predict_fn_final = predict_fn_proba
        else:
            predict_fn_final = predict_fn

        # 生成解释
        exp = explainer.explain_instance(
            data_row=sample,
            predict_fn=predict_fn_final,
            num_features=num_features
        )

        # 保存LIME解释
        lime_path = os.path.join(self.charts_dir, f"lime_explanation_{sample_idx}.png")
        exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(lime_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存LIME解释数据
        lime_data = pd.DataFrame({
            'feature': [exp.as_list()[i][0] for i in range(num_features)],
            'weight': [exp.as_list()[i][1] for i in range(num_features)]
        })
        lime_data_path = os.path.join(self.processed_dir, f"lime_explanation_{sample_idx}.csv")
        lime_data.to_csv(lime_data_path, index=False)

        logger.info(f"LIME解释已保存到: {lime_path}")
        logger.info(f"LIME解释数据已保存到: {lime_data_path}")
        logger.info("\nLIME分析完成！")
        return True

    def run_complete_interpretability(self, model_name="random_forest"):
        """运行完整的模型解释流程"""
        logger.info("=" * 60)
        logger.info("开始模型解释性分析")
        logger.info("=" * 60)

        # 1. 加载数据
        if not self.load_data():
            return False

        # 2. 准备数据
        if not self.prepare_data():
            return False

        # 3. 加载模型
        if not self.load_model(model_name):
            return False

        # 4. SHAP分析
        if not self.shap_analysis():
            logger.warning("跳过SHAP分析，继续LIME分析...")

        # 5. LIME分析
        if not self.lime_analysis():
            logger.warning("跳过LIME分析")

        logger.info("\n" + "=" * 60)
        logger.info("模型解释性分析完成！")
        logger.info("=" * 60)
        logger.info("解释结果已保存到results/charts目录")
        return True

    def plot_feature_importance(self, top_n=20):
        """绘制特征重要性图"""
        if self.model is None:
            logger.error("请先加载模型")
            return False

        logger.info(f"\n绘制前{top_n}个重要特征...")

        # 获取特征重要性
        if hasattr(self.model, 'named_steps'):  # 处理Pipeline模型
            model = self.model.named_steps['model']
        else:
            model = self.model

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            logger.error("模型不支持特征重要性分析")
            return False

        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': importances
        })

        # 按重要性排序
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)

        # 绘制特征重要性图
        plt.figure(figsize=(14, 10))
        sns.barplot(y='feature', x='importance', data=feature_importance, palette='coolwarm')
        plt.title(f'Top {top_n} Feature Importance', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("特征重要性图已保存")
        return True


def main():
    """主函数"""
    interpreter = ModelInterpretability()
    interpreter.run_complete_interpretability()


if __name__ == "__main__":
    main()
