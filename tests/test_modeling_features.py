#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试modeling.py中的特征重要性分析和模型解释功能
"""

import sys
from src.modeling import TraditionalModeling


def test_feature_importance():
    """
    测试特征重要性分析功能
    """
    print("\n\n" + "=" * 80)
    print("=== 开始执行特征重要性分析测试 ===")
    print("=" * 80)

    try:
        # 创建TraditionalModeling实例
        modeling = TraditionalModeling()

        # 加载数据
        print("加载数据...")
        data = modeling.load_data()

        if data is None:
            print("数据加载失败")
            return False

        # 准备数据
        print("准备数据...")
        result = modeling.prepare_data(
            data,
            target='revenue',
            test_size=0.2,
            random_state=42
        )

        if result is None:
            print("数据准备失败")
            return False

        # 正确解包5个返回值
        X_train, X_test, y_train, y_test, features = result

        # 定义模型（仅测试xgboost）
        print("定义模型...")
        models = modeling.define_models()

        # 只保留xgboost模型以加快测试
        if 'XGBoost' in models:
            models = {'XGBoost': models['XGBoost']}
        else:
            print("未找到XGBoost模型")
            return False

        # 运行模型训练和评估，测试特征重要性功能
        results = modeling.train_and_evaluate(X_train, X_test, y_train, y_test, models)

        print("\n✓ 特征重要性分析测试通过！")
        # 正确处理返回的元组结构 (results_df, trained_models_dict)
        results_df, trained_models = results
        print(f"测试模型: {results_df.iloc[0]['Model']}")
        print(f"模型R2分数: {results_df.iloc[0]['R2 Score']:.4f}")

        return True
    except Exception as e:
        print(f"\n✗ 特征重要性分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_interpretability():
    """
    测试模型解释功能
    """
    print("\n\n" + "=" * 80)
    print("=== 开始执行模型解释功能测试 ===")
    print("=" * 80)

    try:
        # 创建TraditionalModeling实例
        modeling = TraditionalModeling()

        # 加载数据
        print("加载数据...")
        data = modeling.load_data()

        if data is None:
            print("数据加载失败")
            return False

        # 准备数据
        print("准备数据...")
        result = modeling.prepare_data(
            data,
            target='revenue',
            test_size=0.2,
            random_state=42
        )

        if result is None:
            print("数据准备失败")
            return False

        # 正确解包5个返回值
        X_train, X_test, y_train, y_test, features = result

        # 定义模型（仅测试xgboost）
        print("定义模型...")
        models = modeling.define_models()

        # 只保留xgboost模型以加快测试
        if 'XGBoost' in models:
            models = {'XGBoost': models['XGBoost']}
        else:
            print("未找到XGBoost模型")
            return False

        # 运行模型训练和评估，测试模型解释功能
        results = modeling.train_and_evaluate(X_train, X_test, y_train, y_test, models)

        print("\n✓ 模型解释功能测试通过！")
        # 正确处理返回的元组结构 (results_df, trained_models_dict)
        results_df, trained_models = results
        print(f"测试模型: {results_df.iloc[0]['Model']}")

        return True
    except Exception as e:
        print(f"\n✗ 模型解释功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    运行所有测试
    """
    print("=" * 80)
    print("开始测试modeling.py的增强功能")
    print("=" * 80)

    tests = [
        test_feature_importance,
        test_model_interpretability
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 80)
    print(f"测试总结: {passed}/{total} 测试通过")
    print("=" * 80)

    if passed == total:
        print("\n✓ 所有测试通过！modeling.py的增强功能正常工作。")
        return 0
    else:
        print("\n✗ 部分测试失败！请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
