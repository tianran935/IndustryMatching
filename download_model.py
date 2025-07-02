#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预下载脚本
将BGE模型下载到本地缓存，避免运行时下载等待
"""

import os
import sys
import time
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

from sentence_transformers import SentenceTransformer
from config import ModelConfig
from logger import setup_logger, log_info, log_error, LogConfig

def download_model(model_name: str = None, device: str = 'cpu'):
    """
    下载指定的BGE模型到本地
    
    Args:
        model_name: 模型名称，默认使用配置文件中的模型
        device: 设备类型，默认使用CPU
    """
    print("=" * 60)
    print("BGE模型预下载工具")
    print("=" * 60)
    
    # 设置日志
    log_config = LogConfig()
    log_config.level = 'INFO'
    log_config.console = True
    setup_logger(log_config)
    
    # 获取模型名称
    if model_name is None:
        model_config = ModelConfig()
        model_name = model_config.name
    
    print(f"\n模型信息:")
    print(f"  模型名称: {model_name}")
    print(f"  下载源: 阿里云镜像 (https://hf-mirror.com)")
    print(f"  缓存目录: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")
    print(f"  设备类型: {device}")
    
    try:
        print(f"\n开始下载模型: {model_name}")
        print("注意: 首次下载可能需要几分钟时间，请耐心等待...")
        
        start_time = time.time()
        
        # 下载并加载模型
        log_info(f"正在从阿里云镜像下载模型: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        
        download_time = time.time() - start_time
        
        print(f"\n✓ 模型下载完成！")
        print(f"  下载耗时: {download_time:.2f}秒")
        print(f"  模型维度: {model.get_sentence_embedding_dimension()}")
        print(f"  最大序列长度: {model.max_seq_length}")
        
        # 测试编码功能
        print(f"\n测试模型编码功能...")
        test_texts = ["软件开发", "数据分析"]
        
        test_start = time.time()
        embeddings = model.encode(test_texts, show_progress_bar=False)
        test_time = time.time() - test_start
        
        print(f"✓ 编码测试成功！")
        print(f"  测试文本: {test_texts}")
        print(f"  编码耗时: {test_time:.3f}秒")
        print(f"  向量形状: {embeddings.shape}")
        
        # 显示缓存信息
        cache_dir = Path(os.environ.get('HUGGINGFACE_HUB_CACHE'))
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            cache_size_mb = cache_size / (1024 * 1024)
            print(f"\n缓存信息:")
            print(f"  缓存目录: {cache_dir}")
            print(f"  缓存大小: {cache_size_mb:.1f} MB")
        
        print(f"\n=" * 60)
        print("✓ 模型下载和测试完成！")
        print("现在可以正常运行主程序，无需等待模型下载。")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 模型下载失败: {e}")
        print(f"\n可能的解决方案:")
        print(f"1. 检查网络连接")
        print(f"2. 确认防火墙设置")
        print(f"3. 重新运行下载脚本")
        print(f"4. 尝试使用VPN")
        print("=" * 60)
        return False

def main():
    """
    主函数
    支持命令行参数指定模型名称和设备
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='BGE模型预下载工具')
    parser.add_argument('--model', type=str, default=None,
                       help='模型名称 (默认: BAAI/bge-small-zh-v1.5)')
    parser.add_argument('--device', type=str, default='cpu', 
                       choices=['cpu', 'cuda'],
                       help='设备类型 (默认: cpu)')
    
    args = parser.parse_args()
    
    success = download_model(args.model, args.device)
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())