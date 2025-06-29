#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志管理模块
提供统一的日志记录功能
"""

import logging
import sys
from typing import Optional
from config import LogConfig

class Logger:
    """日志管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._logger = None
            self._initialized = True
    
    def setup(self, config: LogConfig) -> None:
        """设置日志配置"""
        self._logger = logging.getLogger('JobMatcher')
        self._logger.setLevel(getattr(logging, config.level.upper()))
        
        # 清除现有处理器
        self._logger.handlers.clear()
        
        formatter = logging.Formatter(config.format)
        
        # 控制台处理器
        if config.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
        
        # 文件处理器
        if config.file:
            file_handler = logging.FileHandler(config.file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """获取日志记录器"""
        if self._logger is None:
            raise RuntimeError("Logger not initialized. Call setup() first.")
        return self._logger
    
    def info(self, message: str) -> None:
        """记录信息日志"""
        self.get_logger().info(message)
    
    def warning(self, message: str) -> None:
        """记录警告日志"""
        self.get_logger().warning(message)
    
    def error(self, message: str) -> None:
        """记录错误日志"""
        self.get_logger().error(message)
    
    def debug(self, message: str) -> None:
        """记录调试日志"""
        self.get_logger().debug(message)
    
    def exception(self, message: str) -> None:
        """记录异常日志"""
        self.get_logger().exception(message)

# 全局日志实例
logger = Logger()

# 便捷函数
def setup_logger(config: LogConfig) -> None:
    """设置日志配置"""
    logger.setup(config)

def get_logger() -> logging.Logger:
    """获取日志记录器"""
    return logger.get_logger()

def log_info(message: str) -> None:
    """记录信息日志"""
    logger.info(message)

def log_warning(message: str) -> None:
    """记录警告日志"""
    logger.warning(message)

def log_error(message: str) -> None:
    """记录错误日志"""
    logger.error(message)

def log_debug(message: str) -> None:
    """记录调试日志"""
    logger.debug(message)

def log_exception(message: str) -> None:
    """记录异常日志"""
    logger.exception(message)