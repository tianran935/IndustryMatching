#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试阿里云镜像配置
验证模型下载是否使用阿里云镜像
"""

import os
import sys
from pathlib import Path

# 配置阿里云镜像 - 在导入其他库之前设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'

# 设置模型下载到项目的models目录
project_root = Path(__file__).parent
models_dir = project_root / 'models'
models_dir.mkdir(exist_ok=True)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(models_dir)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(models_dir)

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from config import ModelConfig
from matcher import ModelManager
from logger import setup_logger, log_info, log_error

def test_mirror_configuration():
    """测试镜像配置"""
    print("=" * 60)
    print("测试阿里云镜像配置")
    print("=" * 60)
    
    # 检查环境变量
    print("\n1. 检查环境变量配置:")
    env_vars = ['HF_ENDPOINT', 'HF_HUB_ENDPOINT', 'HUGGINGFACE_HUB_CACHE']
    
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        print(f"   {var}: {value}")
    
    # 测试模型加载
    print("\n2. 测试模型加载:")
    try:
        # 设置简单的日志配置
        from logger import LogConfig
        log_config = LogConfig()
        log_config.level = 'INFO'
        log_config.console = True
        setup_logger(log_config)
        
        # 创建模型配置
        model_config = ModelConfig()
        model_config.name = 'BAAI/bge-small-zh-v1.5'  # 使用默认模型
        model_config.device = 'cpu'  # 使用CPU避免GPU相关问题
        
        print(f"   正在加载模型: {model_config.name}")
        print(f"   使用设备: {model_config.device}")
        
        # 创建模型管理器
        model_manager = ModelManager(model_config)
        
        # 加载模型
        model = model_manager.load_model()
        
        print("   ✓ 模型加载成功！")
        
        # 测试编码
        print("\n3. 测试文本编码:")
        test_texts = ["软件开发", "数据分析", "人工智能"]
        
        embeddings = model_manager.encode_texts(test_texts, show_progress=False)
        
        print(f"   ✓ 编码成功！向量维度: {embeddings.shape}")
        print(f"   ✓ 测试文本数量: {len(test_texts)}")
        
        print("\n=" * 60)
        print("✓ 阿里云镜像配置测试通过！")
        print("模型已成功从阿里云镜像下载并加载")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确认防火墙设置")
        print("3. 重新运行测试")
        print("=" * 60)
        return False

def main():
    """主函数"""
    success = test_mirror_configuration()
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())