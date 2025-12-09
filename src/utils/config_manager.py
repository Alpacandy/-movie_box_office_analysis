import os
import yaml
from .logging_config import get_logger

# 获取全局日志记录器
logger = get_logger(__name__)

class ConfigManager:
    """
    配置管理类，用于加载和管理项目配置
    """
    def __init__(self, config_path=None):
        """
        初始化配置管理器

        Args:
            config_path (str): 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'config.yaml'
            )

        self.config_path = config_path
        self.config = None
        self.load_config()

    def load_config(self):
        """
        加载配置文件
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
        except FileNotFoundError:
            logger.error(f"配置文件未找到: {self.config_path}")
            # 不抛出异常，而是使用默认空配置
            self.config = {}
            logger.info("将使用默认配置")
        except yaml.YAMLError as e:
            logger.error(f"配置文件解析错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置文件时发生错误: {e}")
            raise

    def get(self, key_path, default=None):
        """
        获取配置值

        Args:
            key_path (str): 配置键路径，使用点号分隔，例如 "data_loading.filename"
            default: 默认值，如果配置不存在则返回

        Returns:
            配置值或默认值
        """
        if not self.config:
            logger.warning("配置未加载，使用默认值")
            return default

        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            logger.warning(f"配置键不存在: {key_path}")
            return default
        except TypeError:
            logger.warning(f"配置路径错误: {key_path}")
            return default

    def set(self, key_path, value):
        """
        设置配置值

        Args:
            key_path (str): 配置键路径，使用点号分隔
            value: 配置值
        """
        if not self.config:
            logger.error("配置未加载，无法设置值")
            return False

        keys = key_path.split('.')
        config_ref = self.config

        try:
            # 遍历到最后一个键之前
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]

            # 设置值
            config_ref[keys[-1]] = value
            logger.info(f"配置值设置成功: {key_path} = {value}")
            return True
        except Exception as e:
            logger.error(f"设置配置值时发生错误: {e}")
            return False

    def save_config(self, output_path=None):
        """
        保存配置到文件

        Args:
            output_path (str): 输出路径，如果为None则覆盖原文件

        Returns:
            bool: 是否保存成功
        """
        if not self.config:
            logger.error("配置未加载，无法保存")
            return False

        if output_path is None:
            output_path = self.config_path

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置保存成功: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存配置时发生错误: {e}")
            return False

# 创建全局配置管理器实例
global_config = ConfigManager()
