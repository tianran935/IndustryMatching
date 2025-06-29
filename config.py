#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
集中管理所有配置参数
"""

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """模型配置"""
    name: str = 'BAAI/bge-small-zh-v1.5'
    batch_size: int = 64
    max_length: Optional[int] = None
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'

@dataclass
class FAISSConfig:
    """FAISS索引配置"""
    index_type: str = 'IndexFlatIP'  # 内积索引
    normalize_l2: bool = True
    nprobe: Optional[int] = None  # 用于IVF索引

@dataclass
class ProcessingConfig:
    """数据处理配置"""
    batch_size: int = 1000
    cache_dir: str = './joblib_cache'
    max_cache_size: int = 10000
    
    # 文本清理配置
    drop_texts: List[str] = None
    
    def __post_init__(self):
        if self.drop_texts is None:
            self.drop_texts = [
                '在中国法律允许', '依法须经批准的项目', '经相关部门批准后方可开展经营活动',
                '法律、法规、国务院决定规定应当许可（审批）的', '经相关部门批准后依批准的内容开展经营活动',
                '具体经营项目以审批结果为准', ''
            ]

@dataclass
class DataConfig:
    """数据文件配置"""
    industry_file: str = './data/国民经济分类_2.14.csv'
    chunks_dir: str = './chunks'
    output_file: str = 'JobMatching_Results_BGE_FAISS.csv'
    encoding: str = 'utf-8'
    
    def validate_paths(self) -> bool:
        """验证文件路径是否存在"""
        if not os.path.exists(self.industry_file):
            raise FileNotFoundError(f"行业分类文件不存在: {self.industry_file}")
        
        if not os.path.exists(self.chunks_dir):
            raise FileNotFoundError(f"企业数据目录不存在: {self.chunks_dir}")
        
        return True

@dataclass
class LogConfig:
    """日志配置"""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file: Optional[str] = 'matching.log'
    console: bool = True

class Config:
    """主配置类"""
    def __init__(self):
        self.model = ModelConfig()
        self.faiss = FAISSConfig()
        self.processing = ProcessingConfig()
        self.data = DataConfig()
        self.log = LogConfig()
    
    def validate(self) -> bool:
        """验证所有配置"""
        return self.data.validate_paths()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """从字典创建配置"""
        config = cls()
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'faiss' in config_dict:
            for key, value in config_dict['faiss'].items():
                if hasattr(config.faiss, key):
                    setattr(config.faiss, key, value)
        
        if 'processing' in config_dict:
            for key, value in config_dict['processing'].items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)
        
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if 'log' in config_dict:
            for key, value in config_dict['log'].items():
                if hasattr(config.log, key):
                    setattr(config.log, key, value)
        
        return config

# 默认配置实例
default_config = Config()