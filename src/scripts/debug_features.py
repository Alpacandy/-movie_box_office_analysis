import os
import pandas as pd
import joblib
import logging

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_features')

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录
project_root = os.path.abspath(os.path.join(script_dir, '../..'))

# 设置路径
base_dir = os.path.join(project_root, "data")
processed_dir = os.path.join(base_dir, "processed")
models_dir = os.path.join(project_root, "results/models")

# 加载数据
data = pd.read_csv(os.path.join(processed_dir, "feature_engineered_data.csv"))

# 与modeling.py保持一致的特征选择
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
target = 'revenue'
features = [col for col in numeric_cols if col not in [target, 'profit', 'return_on_investment']]

# 移除可能存在的'title_x'列
if 'title_x' in features:
    features.remove('title_x')

logger.info("特征列表（按原始顺序）:")
for i, feature in enumerate(features):
    logger.info(f"{i+1}. {feature}")

logger.info(f"\n特征数量: {len(features)}")

# 加载模型并检查其使用的特征
logger.info("\n加载模型...")
try:
    model = joblib.load(os.path.join(models_dir, "gradient_boosting_model.joblib"))
    logger.info("模型加载成功")

    # 检查模型管道中的步骤
    if hasattr(model, 'steps'):
        logger.info(f"\n模型管道步骤: {[name for name, _ in model.steps]}")

        # 检查是否有特征名称信息
        # 注意：并非所有模型都保存了特征名称
        logger.info("\n注意：sklearn模型通常不会保存完整的特征名称列表")
        logger.info("需要确保特征顺序与训练时完全一致")

    # 尝试使用前几个样本进行预测，看看是否成功
    logger.info("\n尝试使用前几个样本进行预测...")
    X_test = data[features].head(5)
    try:
        predictions = model.predict(X_test)
        logger.info("✓ 预测成功！特征顺序正确")
        logger.info(f"预测结果: {predictions}")
    except Exception as e:
        logger.error(f"✗ 预测失败: {e}")
        logger.error("特征顺序可能不正确")

        # 尝试不同的方法来解决这个问题
        logger.info("\n尝试将DataFrame转换为numpy数组...")
        try:
            X_test_np = X_test.values
            predictions = model.predict(X_test_np)
            logger.info("✓ 预测成功！使用numpy数组绕过了特征名称检查")
            logger.info(f"预测结果: {predictions}")
        except Exception as e2:
            logger.error(f"✗ 预测仍然失败: {e2}")

except Exception as e:
    logger.error(f"模型加载失败: {e}")