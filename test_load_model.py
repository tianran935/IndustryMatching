#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 matcher.py 中 load_model() 方法的测试脚本
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置环境变量（如果需要）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = str(project_root / 'models')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(project_root / 'models')

from config import ModelConfig
from matcher import ModelManager
from logger import setup_logger, log_info, log_error

class TestLoadModel(unittest.TestCase):
    """测试 ModelManager 的 load_model 方法"""
    
    def setUp(self):
        """测试前的设置"""
        # 设置日志
        setup_logger()
        
        # 创建测试配置
        self.config = ModelConfig(
            name='BAAI/bge-small-zh-v1.5',
            batch_size=32,
            device='auto'
        )
        
        # 创建 ModelManager 实例
        self.model_manager = ModelManager(self.config)
    
    def tearDown(self):
        """测试后的清理"""
        # 清理模型实例
        if hasattr(self.model_manager, 'model') and self.model_manager.model is not None:
            del self.model_manager.model
            self.model_manager.model = None
    
    def test_device_selection_auto(self):
        """测试自动设备选择"""
        log_info("测试自动设备选择...")
        
        device = self.model_manager.get_device()
        self.assertIn(device, ['cpu', 'cuda'])
        
        log_info(f"选择的设备: {device}")
    
    def test_device_selection_cpu(self):
        """测试强制使用CPU"""
        log_info("测试强制使用CPU...")
        
        self.model_manager.config.device = 'cpu'
        device = self.model_manager.get_device()
        self.assertEqual(device, 'cpu')
        
        log_info(f"强制使用设备: {device}")
    
    def test_load_model_success(self):
        """测试成功加载模型"""
        log_info("测试模型加载...")
        
        try:
            start_time = time.time()
            model = self.model_manager.load_model()
            load_time = time.time() - start_time
            
            # 验证模型加载成功
            self.assertIsNotNone(model)
            self.assertIsNotNone(self.model_manager.model)
            
            # 验证模型属性
            self.assertTrue(hasattr(model, 'encode'))
            self.assertTrue(hasattr(model, 'max_seq_length'))
            
            log_info(f"模型加载成功，耗时: {load_time:.2f}秒")
            log_info(f"模型设备: {model.device}")
            log_info(f"最大序列长度: {getattr(model, 'max_seq_length', 'N/A')}")
            
        except Exception as e:
            log_error(f"模型加载失败: {e}")
            self.fail(f"模型加载失败: {e}")
    
    def test_load_model_idempotent(self):
        """测试重复加载模型的幂等性"""
        log_info("测试模型重复加载...")
        
        try:
            # 第一次加载
            model1 = self.model_manager.load_model()
            
            # 第二次加载应该返回同一个实例
            model2 = self.model_manager.load_model()
            
            self.assertIs(model1, model2)
            log_info("模型重复加载测试通过")
            
        except Exception as e:
            log_error(f"模型重复加载测试失败: {e}")
            self.fail(f"模型重复加载测试失败: {e}")
    
    @patch('matcher.SentenceTransformer')
    def test_load_model_with_max_length(self, mock_sentence_transformer):
        """测试设置最大序列长度"""
        log_info("测试设置最大序列长度...")
        
        # 创建模拟模型
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        
        # 设置最大长度
        self.model_manager.config.max_length = 256
        
        # 加载模型
        model = self.model_manager.load_model()
        
        # 验证最大长度被设置
        self.assertEqual(mock_model.max_seq_length, 256)
        log_info("最大序列长度设置测试通过")
    
    def test_load_model_with_invalid_path(self):
        """测试加载不存在的模型路径"""
        log_info("测试加载不存在的模型路径...")
        
        # 创建一个使用无效路径的配置
        invalid_config = ModelConfig(
            name='./invalid/model/path',
            device='cpu'
        )
        
        invalid_model_manager = ModelManager(invalid_config)
        
        # 应该抛出异常
        with self.assertRaises(Exception):
            invalid_model_manager.load_model()
        
        log_info("无效路径测试通过")
    
    def test_encode_simple_text(self):
        """测试简单文本编码"""
        log_info("测试文本编码功能...")
        
        try:
            # 加载模型
            self.model_manager.load_model()
            
            # 测试文本
            test_texts = [
                "计算机软件开发",
                "机械设备制造",
                "餐饮服务"
            ]
            
            # 编码文本
            start_time = time.time()
            embeddings = self.model_manager.encode_texts(test_texts, show_progress=False)
            encode_time = time.time() - start_time
            
            # 验证编码结果
            self.assertEqual(len(embeddings), len(test_texts))
            self.assertGreater(embeddings.shape[1], 0)  # 向量维度大于0
            
            log_info(f"文本编码成功，耗时: {encode_time:.2f}秒")
            log_info(f"编码向量形状: {embeddings.shape}")
            
        except Exception as e:
            log_error(f"文本编码测试失败: {e}")
            self.fail(f"文本编码测试失败: {e}")
    
    def test_encode_empty_text_list(self):
        """测试空文本列表编码"""
        log_info("测试空文本列表编码...")
        
        try:
            # 加载模型
            self.model_manager.load_model()
            
            # 编码空列表
            embeddings = self.model_manager.encode_texts([], show_progress=False)
            
            # 验证返回空数组
            self.assertEqual(len(embeddings), 0)
            
            log_info("空文本列表编码测试通过")
            
        except Exception as e:
            log_error(f"空文本列表编码测试失败: {e}")
            self.fail(f"空文本列表编码测试失败: {e}")

def run_performance_test():
    """运行性能测试"""
    log_info("=" * 60)
    log_info("开始性能测试")
    log_info("=" * 60)
    
    try:
        # 创建配置和管理器
        config = ModelConfig(name='BAAI/bge-small-zh-v1.5', batch_size=64, device='auto')
        model_manager = ModelManager(config)
        
        # 测试模型加载时间
        log_info("测试模型加载性能...")
        start_time = time.time()
        model_manager.load_model()
        load_time = time.time() - start_time
        log_info(f"模型加载时间: {load_time:.2f}秒")
        
        # 测试不同批次大小的编码性能
        test_texts = [f"测试文本{i}：计算机软件开发与技术服务" for i in range(100)]
        
        for batch_size in [16, 32, 64, 128]:
            log_info(f"测试批次大小 {batch_size} 的编码性能...")
            start_time = time.time()
            embeddings = model_manager.encode_texts(test_texts, batch_size=batch_size, show_progress=False)
            encode_time = time.time() - start_time
            
            log_info(f"批次大小 {batch_size}: {encode_time:.2f}秒, {len(test_texts)/encode_time:.2f} 条/秒")
        
        log_info("性能测试完成")
        
    except Exception as e:
        log_error(f"性能测试失败: {e}")

def main():
    """主函数"""
    log_info("开始 load_model() 方法测试")
    log_info("=" * 60)
    
    # 运行单元测试
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行性能测试
    run_performance_test()
    
    log_info("=" * 60)
    log_info("所有测试完成")

if __name__ == '__main__':
    main()