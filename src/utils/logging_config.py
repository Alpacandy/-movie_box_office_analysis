#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一日志配置模块
为项目提供一致的日志记录功能
"""

import os
import logging
import logging.handlers
from datetime import datetime

# 获取脚本所在目录
src_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录
project_root = os.path.abspath(os.path.join(src_dir, "../.."))

def setup_logging(
    log_level=logging.INFO,
    log_dir=os.path.join(project_root, "logs"),
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5,
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    配置统一的日志记录

    Args:
        log_level: 日志级别
        log_dir: 日志文件存储目录
        max_bytes: 单个日志文件最大大小
        backup_count: 日志文件备份数量
        log_format: 日志格式

    Returns:
        logging.Logger: 根日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 创建格式化器
    formatter = logging.Formatter(log_format)

    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 移除已存在的处理器，避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 创建文件处理器 (按大小切割)
    log_file = os.path.join(log_dir, f"movie_analysis_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # 文件日志记录更详细的信息
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name):
    """
    获取指定名称的日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        logging.Logger: 日志记录器
    """
    return logging.getLogger(name)
