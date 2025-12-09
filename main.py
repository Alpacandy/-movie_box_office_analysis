#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电影票房数据分析项目主入口文件
提供统一的项目调用接口和工作流管理
"""

import os
import sys
import argparse
import traceback
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载统一日志配置
from src.utils.logging_config import setup_logging, get_logger  # noqa: E402
setup_logging()
logger = get_logger('movie_analysis_main')

# 加载统一配置管理
from src.utils.config_manager import ConfigManager  # noqa: E402

# 导入项目模块
from src.data_preprocessing import DataPreprocessing  # noqa: E402
from src.feature_engineering import FeatureEngineering  # noqa: E402
from src.modeling import TraditionalModeling  # noqa: E402
from src.deep_learning import DeepLearningModeling  # noqa: E402
from src.model_fusion import ModelFusion  # noqa: E402
from src.visualization import MovieVisualization  # noqa: E402


class MovieAnalysisPipeline:
    """电影票房数据分析项目的统一工作流管理类"""

    def __init__(self, config_file=None):
        """初始化工作流管理器

        Args:
            config_file: 配置文件路径
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.results_dir = os.path.join(self.base_dir, "results")
        self.src_dir = os.path.join(self.base_dir, "src")

        # 加载配置
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config

        # 确保必要目录存在
        for dir_path in [self.data_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)

        logger.info("电影票房数据分析项目初始化完成")
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"结果目录: {self.results_dir}")

    def load_config(self, config_file=None):
        """加载配置文件（兼容旧代码）

        Args:
            config_file: 配置文件路径

        Returns:
            dict: 配置字典
        """
        # 使用统一配置管理器加载配置
        if config_file:
            return ConfigManager(config_file).config
        return ConfigManager().config

    def run_data_preprocessing(self):
        """运行数据预处理流程"""
        logger.info("=" * 50)
        logger.info("开始数据预处理流程")
        logger.info("=" * 50)

        try:

            # 创建数据预处理实例
            preprocessor = DataPreprocessing(
                base_dir=self.data_dir,
                config_file=os.path.join(self.base_dir, "config", "default_config.yaml")
            )

            # 运行完整的数据预处理流程
            preprocessor.run_complete_preprocessing()

            logger.info("数据预处理流程完成")
            return True
        except (IOError, ValueError, TypeError) as e:
            logger.error(f"数据预处理失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def run_feature_engineering(self):
        """运行特征工程流程"""
        logger.info("=" * 50)
        logger.info("开始特征工程流程")
        logger.info("=" * 50)

        try:

            # 创建特征工程实例
            engineer = FeatureEngineering(base_dir=self.data_dir)

            # 运行完整的特征工程流程
            data, selected_features = engineer.run_complete_feature_engineering(
                bert_features=self.config['feature_engineering']['bert_features'],
                use_shap=self.config['feature_engineering']['use_shap']
            )
            if data is None:
                logger.error("特征工程失败: 无法生成特征工程数据")
                return False

            logger.info("特征工程流程完成")
            logger.info(f"选择的特征数量: {len(selected_features)}")
            return True
        except (IOError, ValueError, TypeError, KeyError) as e:
            logger.error(f"特征工程失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def run_modeling(self):
        """运行传统建模流程"""
        logger.info("=" * 50)
        logger.info("开始传统建模流程")
        logger.info("=" * 50)

        try:

            # 创建建模实例
            modeler = TraditionalModeling(
                base_dir=self.data_dir,
                results_dir=self.results_dir
            )

            # 加载数据
            data = modeler.load_data()
            if data is None:
                logger.error("建模失败: 无法加载数据")
                return False

            # 准备数据
            X_train, X_test, y_train, y_test, _ = modeler.prepare_data(
                data,
                target='revenue',  # 默认目标变量
                test_size=self.config['traditional_modeling']['test_size'],
                random_state=self.config['traditional_modeling']['random_state']
            )

            # 定义模型
            models = modeler.define_models()

            # 训练和评估模型
            modeler.train_and_evaluate(X_train, X_test, y_train, y_test, models)

            logger.info("传统建模流程完成")
            return True
        except (IOError, ValueError, TypeError, KeyError) as e:
            logger.error(f"传统建模失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def run_deep_learning(self):
        """运行深度学习建模流程"""
        logger.info("=" * 50)
        logger.info("开始深度学习建模流程")
        logger.info("=" * 50)

        try:

            # 创建深度学习建模实例
            deep_modeler = DeepLearningModeling(
                base_dir=self.data_dir,
                results_dir=self.results_dir
            )

            # 加载数据
            data = deep_modeler.load_data()
            if data is None:
                logger.error("深度学习建模失败: 无法加载数据")
                return False

            # 准备数据
            X_train, X_test, y_train, y_test, scaler_y = deep_modeler.prepare_data(
                data,
                target='revenue',  # 默认目标变量
                test_size=self.config['deep_learning']['test_size'],
                random_state=self.config['deep_learning']['random_state']
            )

            # 运行不同的深度学习模型
            results = {}
            results['dense'] = deep_modeler.run_dense_network(X_train, X_test, y_train, y_test, scaler_y)
            results['cnn'] = deep_modeler.run_cnn_network(X_train, X_test, y_train, y_test, scaler_y)
            results['lstm'] = deep_modeler.run_rnn_network(X_train, X_test, y_train, y_test, scaler_y, rnn_type='lstm')

            # 比较模型
            deep_modeler.compare_deep_models(results)

            logger.info("深度学习建模流程完成")
            return True
        except Exception as e:
            logger.error(f"深度学习建模失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def run_model_fusion(self):
        """
        运行模型融合流程
        """
        logger.info("=" * 50)
        logger.info("开始模型融合流程")
        logger.info("=" * 50)

        try:

            # 创建模型融合实例
            fusion_modeler = ModelFusion(
                base_dir=self.data_dir,
                results_dir=self.results_dir
            )

            # 运行模型融合流程
            fusion_modeler.run_model_fusion()

            logger.info("模型融合流程完成")
            return True
        except (IOError, ValueError, TypeError, KeyError) as e:
            logger.error(f"模型融合失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def run_visualization(self):
        """运行数据可视化流程"""
        logger.info("=" * 50)
        logger.info("开始数据可视化流程")
        logger.info("=" * 50)

        try:

            # 创建可视化实例
            visualizer = MovieVisualization(
                base_dir=self.data_dir,
                results_dir=self.results_dir
            )

            # 运行完整的可视化流程
            visualizer.run_complete_visualization()

            logger.info("数据可视化流程完成")
            return True
        except Exception as e:
            logger.error(f"数据可视化失败: {e}")
            logger.error(traceback.format_exc())
            return False

    def run_full_pipeline(self):
        """运行完整的分析流程"""
        logger.info("=" * 70)
        logger.info("开始完整的电影票房数据分析流程")
        logger.info("=" * 70)

        # 记录开始时间
        start_time = datetime.now()

        # 运行各流程
        success = True

        success &= self.run_data_preprocessing()
        success &= self.run_feature_engineering()
        success &= self.run_modeling()

        # 如果启用了深度学习，运行深度学习流程
        if self.config['deep_learning'].get('enable', True):
            success &= self.run_deep_learning()

        # 运行模型融合流程
        success &= self.run_model_fusion()

        # 运行可视化流程
        success &= self.run_visualization()

        # 计算总耗时
        total_time = datetime.now() - start_time

        if success:
            logger.info("=" * 70)
            logger.info(f"完整分析流程已成功完成！总耗时: {total_time}")
            logger.info("=" * 70)
        else:
            logger.error("=" * 70)
            logger.error(f"分析流程中存在失败的步骤，总耗时: {total_time}")
            logger.error("=" * 70)

        return success

    def run_component(self, component):
        """运行单个组件

        Args:
            component: 组件名称
                      ('preprocess', 'features', 'modeling',
                       'deep', 'visualize')

        Returns:
            bool: 是否成功
        """
        component_map = {
            'preprocess': self.run_data_preprocessing,
            'features': self.run_feature_engineering,
            'modeling': self.run_modeling,
            'deep': self.run_deep_learning,
            'fusion': self.run_model_fusion,
            'visualize': self.run_visualization
        }

        if component in component_map:
            return component_map[component]()
        else:
            logger.error(f"未知组件: {component}")
            logger.error(f"可用组件: {list(component_map.keys())}")
            return False


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='电影票房数据分析项目')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--component', help='运行单个组件: preprocess, features, modeling, deep, visualize')
    parser.add_argument('--full', action='store_true', help='运行完整流程')
    args = parser.parse_args()

    # 创建项目实例
    project = MovieAnalysisPipeline(args.config)

    # 运行指定的流程
    if args.component:
        project.run_component(args.component)
    elif args.full:
        project.run_full_pipeline()
    else:
        # 默认运行数据预处理和特征工程
        project.run_data_preprocessing()
        project.run_feature_engineering()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
