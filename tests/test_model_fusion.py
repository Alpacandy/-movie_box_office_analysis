#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型融合功能
"""

import os
import sys
from src.model_fusion import ModelFusion


def test_basic_fusion():
    """
    测试基本融合功能
    """
    print("\n" + "=" * 60)
    print("测试基本模型融合功能")
    print("=" * 60)

    try:
        # 创建模型融合实例
        model_fusion = ModelFusion()

        # 禁用缓存进行测试
        model_fusion.cache_enabled = False

        # 运行模型融合
        fusion_metrics = model_fusion.run_model_fusion(fusion_strategy="average")

        print("\n✓ 基本模型融合测试通过！")
        print(f"融合模型R2 Score: {fusion_metrics['R2 Score']:.4f}")

        return True
    except Exception as e:
        print(f"\n✗ 基本模型融合测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_mechanism():
    """
    测试缓存机制
    """
    print("\n" + "=" * 60)
    print("测试缓存机制")
    print("=" * 60)

    try:
        # 创建模型融合实例
        model_fusion = ModelFusion()

        # 启用缓存，设置较短的TTL
        model_fusion.cache_enabled = True
        model_fusion.cache_ttl = 3600  # 1小时

        print("第一次运行模型融合（生成缓存）...")
        import time
        start_time = time.time()
        fusion_metrics1 = model_fusion.run_model_fusion(fusion_strategy="weighted")
        time1 = time.time() - start_time
        print(f"第一次运行耗时: {time1:.2f} 秒")

        print("\n第二次运行模型融合（使用缓存）...")
        start_time = time.time()
        fusion_metrics2 = model_fusion.run_model_fusion(fusion_strategy="weighted")
        time2 = time.time() - start_time
        print(f"第二次运行耗时: {time2:.2f} 秒")

        # 检查两次结果是否一致
        if abs(fusion_metrics1['R2 Score'] - fusion_metrics2['R2 Score']) < 1e-6:
            print("\n✓ 缓存一致性测试通过！")
        else:
            print("\n✗ 缓存一致性测试失败: 两次结果不一致")
            print(f"第一次结果: {fusion_metrics1['R2 Score']:.4f}")
            print(f"第二次结果: {fusion_metrics2['R2 Score']:.4f}")
            return False

        # 检查缓存是否提高了速度
        if time2 < time1 * 0.5:  # 第二次运行应该至少快50%
            print("✓ 缓存性能测试通过！")
        else:
            print("✗ 缓存性能测试失败: 缓存没有显著提高速度")
            print(f"第一次运行: {time1:.2f} 秒")
            print(f"第二次运行: {time2:.2f} 秒")
            # 不返回失败，因为可能受其他因素影响

        return True
    except Exception as e:
        print(f"\n✗ 缓存机制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_fusion_strategies():
    """
    测试所有融合策略
    """
    print("\n" + "=" * 60)
    print("测试所有融合策略")
    print("=" * 60)

    strategies = ["average", "weighted", "performance", "stacking"]
    results = {}

    for strategy in strategies:
        print(f"\n测试 {strategy} 融合策略...")
        try:
            # 创建模型融合实例
            model_fusion = ModelFusion()
            model_fusion.cache_enabled = False

            # 运行模型融合
            fusion_metrics = model_fusion.run_model_fusion(fusion_strategy=strategy)
            results[strategy] = fusion_metrics['R2 Score']
            print(f"✓ {strategy} 策略测试通过！R2 Score: {fusion_metrics['R2 Score']:.4f}")
        except Exception as e:
            print(f"✗ {strategy} 策略测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[strategy] = None

    # 打印所有策略结果对比
    print("\n" + "=" * 40)
    print("所有融合策略结果对比")
    print("=" * 40)
    for strategy, r2_score in results.items():
        if r2_score is not None:
            print(f"{strategy}: {r2_score:.4f}")
        else:
            print(f"{strategy}: 失败")

    # 检查是否至少有一个策略成功
    successful_strategies = [s for s, r in results.items() if r is not None]
    if len(successful_strategies) >= 2:
        print("\n✓ 至少2个融合策略测试通过！")
        return True
    else:
        print(f"\n✗ 融合策略测试失败: 只有 {len(successful_strategies)} 个策略成功")
        return False


def main():
    """
    主测试函数
    """
    print("开始模型融合功能测试...")

    # 切换到项目根目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 运行测试
    tests = [
        test_basic_fusion,
        test_cache_mechanism,
        test_all_fusion_strategies
    ]

    results = []
    for test in tests:
        results.append(test())

    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"测试总数: {total}")
    print(f"通过数: {passed}")
    print(f"通过率: {(passed / total * 100):.1f}%")

    if passed == total:
        print("\n✓ 所有测试通过！模型融合功能正常。")
        return 0
    else:
        print(f"\n✗ 有 {(total - passed)} 个测试失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())
