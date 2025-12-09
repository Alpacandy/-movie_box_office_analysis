#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的特征重要性测试脚本，用于调试测试失败问题
"""

import os
import sys
import traceback

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))


from src.modeling import TraditionalModeling  # noqa: E402


def simple_test_feature_importance():
    """简化的特征重要性测试"""
    print("\n=== 开始简化特征重要性测试 ===")

    try:
        # 创建实例
        print("1. 创建TraditionalModeling实例...")
        modeling = TraditionalModeling()

        # 加载数据
        print("2. 加载数据...")
        data = modeling.load_data()
        if data is None:
            raise ValueError("数据加载失败")
        print(f"   数据形状: {data.shape}")

        # 准备数据
        print("3. 准备数据...")
        result = modeling.prepare_data(data, target='revenue', test_size=0.2, random_state=42)
        if result is None:
            raise ValueError("数据准备失败")

        # 解包数据
        X_train, X_test, y_train, y_test, features = result
        print(f"   特征数量: {len(features)}")
        print(f"   训练集: {X_train.shape}, 测试集: {X_test.shape}")

        # 定义模型
        print("4. 定义模型...")
        models = modeling.define_models()
        print(f"   定义的模型数量: {len(models)}")

        # 选择XGBoost模型
        if 'XGBoost' not in models:
            raise ValueError("未找到XGBoost模型")

        print("5. 训练XGBoost模型...")
        results = modeling.train_and_evaluate(X_train, X_test, y_train, y_test, {'XGBoost': models['XGBoost']})

        # 检查结果
        print("6. 验证结果...")
        print(f"   结果类型: {type(results)}")
        print(f"   结果内容: {results}")

        if not results or len(results) == 0:
            raise ValueError("训练结果为空")

        print("\n✅ 特征重要性测试成功！")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = simple_test_feature_importance()
    sys.exit(0 if success else 1)
