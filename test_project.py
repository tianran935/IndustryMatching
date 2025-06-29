#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目功能完整测试代码
测试所有模块的功能，包括单元测试和集成测试
"""

import os
import sys
import time
import tempfile
import shutil
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, ModelConfig, FAISSConfig, ProcessingConfig, DataConfig, LogConfig
from logger import Logger, setup_logger
from data_processor import TextCleaner, DataLoader, BusinessScopeProcessor, DataProcessor
from matcher import ModelManager, FAISSIndexManager, OptimizedMatcher
from main import JobMatcher

class TestConfig(unittest.TestCase):
    """配置模块测试"""
    
    def test_model_config(self):
        """测试模型配置"""
        config = ModelConfig()
        self.assertEqual(config.name, 'BAAI/bge-small-zh-v1.5')
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.device, 'auto')
    
    def test_faiss_config(self):
        """测试FAISS配置"""
        config = FAISSConfig()
        self.assertEqual(config.index_type, 'IndexFlatIP')
        self.assertTrue(config.normalize_l2)
    
    def test_processing_config(self):
        """测试处理配置"""
        config = ProcessingConfig()
        self.assertEqual(config.batch_size, 1000)
        self.assertIsNotNone(config.drop_texts)
        self.assertIn('在中国法律允许', config.drop_texts)
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        config_dict = {
            'model': {'batch_size': 128, 'device': 'cpu'},
            'processing': {'batch_size': 500}
        }
        config = Config.from_dict(config_dict)
        self.assertEqual(config.model.batch_size, 128)
        self.assertEqual(config.model.device, 'cpu')
        self.assertEqual(config.processing.batch_size, 500)

class TestLogger(unittest.TestCase):
    """日志模块测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_logger_setup(self):
        """测试日志设置"""
        log_config = LogConfig(level='DEBUG', file=self.log_file)
        logger = Logger()
        logger.setup(log_config)
        
        # 测试日志记录
        logger.info("测试信息")
        logger.warning("测试警告")
        logger.error("测试错误")
        
        # 验证日志文件存在
        self.assertTrue(os.path.exists(self.log_file))
        
        # 验证日志内容
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("测试信息", content)
            self.assertIn("测试警告", content)
            self.assertIn("测试错误", content)

class TestTextCleaner(unittest.TestCase):
    """文本清理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        config = ProcessingConfig(cache_dir=self.temp_dir)
        self.cleaner = TextCleaner(config)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_clean_text_basic(self):
        """测试基本文本清理"""
        # 测试HTML标签删除
        text = "<p>这是一个测试</p>"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn('<p>', cleaned)
        self.assertNotIn('</p>', cleaned)
        
        # 测试URL删除
        text = "访问 https://www.example.com 获取更多信息"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn('https://www.example.com', cleaned)
        
        # 测试数字删除
        text = "公司成立于2020年"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn('2020', cleaned)
        
        # 测试括号删除
        text = "软件开发（包括移动应用）"
        cleaned = self.cleaner.clean_text(text)
        self.assertNotIn('（', cleaned)
        self.assertNotIn('）', cleaned)
    
    def test_vectorized_clean(self):
        """测试批量文本清理"""
        texts = [
            "<p>软件开发</p>",
            "硬件销售（零售）",
            "在中国法律允许的范围内",
            "咨询服务123"
        ]
        
        cleaned = self.cleaner.vectorized_clean(texts)
        
        # 验证结果数量
        self.assertEqual(len(cleaned), len(texts))
        
        # 验证清理效果
        self.assertNotIn('<p>', cleaned[0])
        self.assertNotIn('（', cleaned[1])
        self.assertEqual(cleaned[2], '')  # 应该被过滤掉
        self.assertNotIn('123', cleaned[3])
    
    def test_empty_and_invalid_input(self):
        """测试空值和无效输入"""
        # 测试空字符串
        self.assertEqual(self.cleaner.clean_text(''), '')
        self.assertEqual(self.cleaner.clean_text('   '), '')
        
        # 测试None值
        self.assertEqual(self.cleaner.clean_text(None), '')
        
        # 测试非字符串类型
        self.assertEqual(self.cleaner.clean_text(123), '')

class TestDataLoader(unittest.TestCase):
    """数据加载器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试行业数据
        self.industry_file = os.path.join(self.temp_dir, 'industry.csv')
        industry_data = pd.DataFrame({
            '行业代码': ['A01', 'A02', 'B01'],
            '行业名称': ['农业', '林业', '制造业']
        })
        industry_data.to_csv(self.industry_file, index=False, encoding='utf-8')
        
        # 创建测试企业数据目录
        self.chunks_dir = os.path.join(self.temp_dir, 'chunks')
        os.makedirs(self.chunks_dir)
        
        # 创建测试企业数据文件
        chunk1_file = os.path.join(self.chunks_dir, 'chunk1.csv')
        chunk1_data = pd.DataFrame({
            '企业名称': ['公司A', '公司B'],
            '经营范围': ['软件开发', '硬件销售']
        })
        chunk1_data.to_csv(chunk1_file, index=False, encoding='utf-8')
        
        chunk2_file = os.path.join(self.chunks_dir, 'chunk2.csv')
        chunk2_data = pd.DataFrame({
            '企业名称': ['公司C'],
            '经营范围': ['咨询服务']
        })
        chunk2_data.to_csv(chunk2_file, index=False, encoding='utf-8')
        
        # 创建配置
        self.config = DataConfig(
            industry_file=self.industry_file,
            chunks_dir=self.chunks_dir,
            output_file=os.path.join(self.temp_dir, 'output.csv')
        )
        self.loader = DataLoader(self.config)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_industry_data(self):
        """测试加载行业数据"""
        df = self.loader.load_industry_data()
        
        # 验证数据结构
        self.assertIn('行业代码', df.columns)
        self.assertIn('行业名称', df.columns)
        self.assertEqual(len(df), 3)
        
        # 验证数据内容
        self.assertIn('A01', df['行业代码'].values)
        self.assertIn('农业', df['行业名称'].values)
    
    def test_load_enterprise_chunks(self):
        """测试加载企业数据分块"""
        chunks = self.loader.load_enterprise_chunks()
        
        # 验证分块数量
        self.assertEqual(len(chunks), 2)
        
        # 验证分块内容
        chunk_names = [chunk[0] for chunk in chunks]
        self.assertIn('chunk1.csv', chunk_names)
        self.assertIn('chunk2.csv', chunk_names)
        
        # 验证数据内容
        total_records = sum(len(chunk[1]) for chunk in chunks)
        self.assertEqual(total_records, 3)
    
    def test_save_results(self):
        """测试保存结果"""
        results = [
            {'企业名称': '公司A', '行业代码': 'A01', '相似度': 0.95},
            {'企业名称': '公司B', '行业代码': 'B01', '相似度': 0.88}
        ]
        
        output_file = self.loader.save_results(results)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(output_file))
        
        # 验证文件内容
        df = pd.read_csv(output_file, encoding='utf-8')
        self.assertEqual(len(df), 2)
        self.assertIn('企业名称', df.columns)
        self.assertIn('行业代码', df.columns)
        self.assertIn('相似度', df.columns)

class TestBusinessScopeProcessor(unittest.TestCase):
    """经营范围处理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        config = ProcessingConfig()
        text_cleaner = TextCleaner(config)
        self.processor = BusinessScopeProcessor(text_cleaner)
    
    def test_process_business_scope(self):
        """测试处理单个经营范围"""
        scope_text = "软件开发；硬件销售，技术咨询。"
        segments = self.processor.process_business_scope(scope_text)
        
        # 验证分割效果
        self.assertGreater(len(segments), 1)
        
        # 验证包含预期内容
        segment_text = ' '.join(segments)
        self.assertIn('软件开发', segment_text)
        self.assertIn('硬件销售', segment_text)
        self.assertIn('技术咨询', segment_text)
    
    def test_process_batch(self):
        """测试批量处理经营范围"""
        scope_texts = [
            "软件开发；硬件销售",
            "技术咨询，管理咨询",
            ""  # 空字符串
        ]
        
        results = self.processor.process_batch(scope_texts)
        
        # 验证结果数量
        self.assertEqual(len(results), 3)
        
        # 验证非空结果
        self.assertGreater(len(results[0]), 0)
        self.assertGreater(len(results[1]), 0)
        self.assertEqual(len(results[2]), 0)  # 空字符串应该返回空列表

class TestDataProcessor(unittest.TestCase):
    """数据处理器主类测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        industry_file = os.path.join(self.temp_dir, 'industry.csv')
        industry_data = pd.DataFrame({
            '行业代码': ['A01', 'A02'],
            '行业名称': ['软件开发', '硬件制造']
        })
        industry_data.to_csv(industry_file, index=False, encoding='utf-8')
        
        chunks_dir = os.path.join(self.temp_dir, 'chunks')
        os.makedirs(chunks_dir)
        
        # 创建配置
        processing_config = ProcessingConfig()
        data_config = DataConfig(
            industry_file=industry_file,
            chunks_dir=chunks_dir
        )
        
        self.processor = DataProcessor(processing_config, data_config)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_and_process_industry_data(self):
        """测试加载并处理行业数据"""
        industry_df, industry_names = self.processor.load_and_process_industry_data()
        
        # 验证返回结果
        self.assertIsInstance(industry_df, pd.DataFrame)
        self.assertIsInstance(industry_names, list)
        
        # 验证数据内容
        self.assertGreater(len(industry_df), 0)
        self.assertGreater(len(industry_names), 0)
        self.assertEqual(len(industry_df), len(industry_names))

class TestModelManager(unittest.TestCase):
    """模型管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 使用较小的模型进行测试
        config = ModelConfig(name='sentence-transformers/all-MiniLM-L6-v2', batch_size=2)
        self.manager = ModelManager(config)
    
    @patch('torch.cuda.is_available')
    def test_get_device(self, mock_cuda):
        """测试设备检测"""
        # 测试CUDA可用
        mock_cuda.return_value = True
        device = self.manager._get_device()
        self.assertEqual(device, 'cuda')
        
        # 测试CUDA不可用
        mock_cuda.return_value = False
        device = self.manager._get_device()
        self.assertEqual(device, 'cpu')
    
    def test_encode_texts(self):
        """测试文本编码（需要网络连接）"""
        # 跳过需要下载模型的测试
        self.skipTest("跳过需要下载模型的测试")
        
        texts = ['软件开发', '硬件销售']
        embeddings = self.manager.encode_texts(texts)
        
        # 验证编码结果
        self.assertEqual(len(embeddings), 2)
        self.assertGreater(embeddings.shape[1], 0)  # 验证向量维度

class TestFAISSIndexManager(unittest.TestCase):
    """FAISS索引管理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        config = FAISSConfig()
        self.manager = FAISSIndexManager(config)
    
    def test_build_index(self):
        """测试构建索引"""
        # 创建测试向量
        embeddings = np.random.rand(10, 128).astype(np.float32)
        
        # 构建索引
        index = self.manager.build_index(embeddings)
        
        # 验证索引
        self.assertIsNotNone(index)
        self.assertEqual(index.ntotal, 10)
        self.assertEqual(self.manager.dimension, 128)
    
    def test_search(self):
        """测试搜索"""
        # 创建测试向量
        embeddings = np.random.rand(10, 128).astype(np.float32)
        query_embeddings = np.random.rand(2, 128).astype(np.float32)
        
        # 构建索引
        self.manager.build_index(embeddings)
        
        # 执行搜索
        scores, indices = self.manager.search(query_embeddings, k=3)
        
        # 验证搜索结果
        self.assertEqual(scores.shape, (2, 3))
        self.assertEqual(indices.shape, (2, 3))
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 10))

class TestOptimizedMatcher(unittest.TestCase):
    """优化匹配器测试"""
    
    def setUp(self):
        """设置测试环境"""
        model_config = ModelConfig(name='sentence-transformers/all-MiniLM-L6-v2')
        faiss_config = FAISSConfig()
        self.matcher = OptimizedMatcher(model_config, faiss_config)
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        # 测试缓存信息
        cache_info = self.matcher.get_cache_info()
        self.assertIn('cache_size', cache_info)
        self.assertIn('model_loaded', cache_info)
        self.assertIn('index_built', cache_info)
        
        # 测试清空缓存
        self.matcher.clear_cache()
        cache_info = self.matcher.get_cache_info()
        self.assertEqual(cache_info['cache_size'], 0)

class TestJobMatcher(unittest.TestCase):
    """工作匹配器主类测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        config = Config()
        config.log.file = os.path.join(self.temp_dir, 'test.log')
        config.log.console = False
        
        # 模拟数据文件（不实际创建，用于测试配置）
        config.data.industry_file = os.path.join(self.temp_dir, 'industry.csv')
        config.data.chunks_dir = os.path.join(self.temp_dir, 'chunks')
        config.data.output_file = os.path.join(self.temp_dir, 'output.csv')
        
        self.matcher = JobMatcher(config)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_validate_environment(self):
        """测试环境验证"""
        # 由于缺少数据文件，验证应该失败
        with self.assertRaises(FileNotFoundError):
            self.matcher.validate_environment()
    
    def test_print_statistics(self):
        """测试统计信息打印"""
        # 设置一些测试统计数据
        self.matcher.stats = {
            'total_enterprises': 100,
            'processed_enterprises': 95,
            'matched_enterprises': 85,
            'total_time': 120.5,
            'load_time': 10.2,
            'process_time': 110.3
        }
        
        # 测试打印（不会抛出异常）
        try:
            self.matcher.print_statistics()
        except Exception as e:
            self.fail(f"print_statistics raised {e} unexpectedly!")

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """设置集成测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建完整的测试数据
        self.setup_test_data()
        
        # 创建配置
        self.config = Config()
        self.config.data.industry_file = self.industry_file
        self.config.data.chunks_dir = self.chunks_dir
        self.config.data.output_file = os.path.join(self.temp_dir, 'results.csv')
        self.config.log.file = os.path.join(self.temp_dir, 'test.log')
        self.config.log.console = False
        
        # 使用CPU和小批次进行测试
        self.config.model.device = 'cpu'
        self.config.model.batch_size = 2
    
    def tearDown(self):
        """清理集成测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def setup_test_data(self):
        """设置测试数据"""
        # 创建行业数据
        self.industry_file = os.path.join(self.temp_dir, 'industry.csv')
        industry_data = pd.DataFrame({
            '行业代码': ['C39', 'I65', 'M73'],
            '行业名称': ['计算机、通信和其他电子设备制造业', '软件和信息技术服务业', '科技推广和应用服务业']
        })
        industry_data.to_csv(self.industry_file, index=False, encoding='utf-8')
        
        # 创建企业数据目录和文件
        self.chunks_dir = os.path.join(self.temp_dir, 'chunks')
        os.makedirs(self.chunks_dir)
        
        chunk_file = os.path.join(self.chunks_dir, 'test_chunk.csv')
        enterprise_data = pd.DataFrame({
            '企业名称': ['科技公司A', '软件公司B', '制造公司C'],
            '统一社会信用代码': ['123456789', '987654321', '456789123'],
            '经营范围': [
                '软件开发；技术咨询；计算机系统服务',
                '应用软件服务；信息系统集成服务',
                '电子产品制造；硬件设备销售'
            ]
        })
        enterprise_data.to_csv(chunk_file, index=False, encoding='utf-8')
    
    def test_data_processing_pipeline(self):
        """测试数据处理流水线"""
        # 创建数据处理器
        processor = DataProcessor(self.config.processing, self.config.data)
        
        # 测试加载和处理行业数据
        industry_df, industry_names = processor.load_and_process_industry_data()
        self.assertGreater(len(industry_df), 0)
        self.assertGreater(len(industry_names), 0)
        
        # 测试加载企业数据
        enterprise_chunks = processor.load_enterprise_data()
        self.assertGreater(len(enterprise_chunks), 0)
        
        # 测试处理经营范围
        chunk_name, chunk_df = enterprise_chunks[0]
        business_scopes = chunk_df['经营范围'].tolist()
        processed_scopes = processor.scope_processor.process_batch(business_scopes)
        
        self.assertEqual(len(processed_scopes), len(business_scopes))
        self.assertGreater(len(processed_scopes[0]), 0)  # 第一个应该有处理结果
    
    def test_text_cleaning_pipeline(self):
        """测试文本清理流水线"""
        cleaner = TextCleaner(self.config.processing)
        
        # 测试各种文本清理场景
        test_cases = [
            ("软件开发（包括移动应用）", "软件开发"),
            ("<p>网站建设</p>", "网站建设"),
            ("技术咨询123服务", "技术咨询服务"),
            ("在中国法律允许的范围内经营", ""),  # 应该被过滤
        ]
        
        for original, expected_contains in test_cases:
            cleaned = cleaner.clean_text(original)
            if expected_contains:
                self.assertIn(expected_contains, cleaned)
            else:
                self.assertEqual(cleaned, expected_contains)

def create_test_suite():
    """创建测试套件"""
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestConfig,
        TestLogger,
        TestTextCleaner,
        TestDataLoader,
        TestBusinessScopeProcessor,
        TestDataProcessor,
        TestFAISSIndexManager,
        TestOptimizedMatcher,
        TestJobMatcher,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite

def run_performance_test():
    """运行性能测试"""
    print("\n" + "="*60)
    print("性能测试")
    print("="*60)
    
    # 创建临时测试环境
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建大量测试数据
        print("创建测试数据...")
        
        # 行业数据
        industry_file = os.path.join(temp_dir, 'industry.csv')
        industry_data = pd.DataFrame({
            '行业代码': [f'A{i:02d}' for i in range(100)],
            '行业名称': [f'行业{i}' for i in range(100)]
        })
        industry_data.to_csv(industry_file, index=False, encoding='utf-8')
        
        # 企业数据
        chunks_dir = os.path.join(temp_dir, 'chunks')
        os.makedirs(chunks_dir)
        
        chunk_file = os.path.join(chunks_dir, 'perf_test.csv')
        enterprise_data = pd.DataFrame({
            '企业名称': [f'公司{i}' for i in range(1000)],
            '经营范围': [f'业务{i}；服务{i}，产品{i}' for i in range(1000)]
        })
        enterprise_data.to_csv(chunk_file, index=False, encoding='utf-8')
        
        # 性能测试配置
        config = ProcessingConfig(cache_dir=temp_dir)
        data_config = DataConfig(
            industry_file=industry_file,
            chunks_dir=chunks_dir
        )
        
        # 测试文本清理性能
        print("测试文本清理性能...")
        cleaner = TextCleaner(config)
        
        test_texts = [f'软件开发{i}；硬件销售{i}，技术咨询{i}' for i in range(1000)]
        
        start_time = time.time()
        cleaned_texts = cleaner.vectorized_clean(test_texts)
        clean_time = time.time() - start_time
        
        print(f"清理 {len(test_texts)} 个文本耗时: {clean_time:.2f}秒")
        print(f"平均每个文本: {clean_time/len(test_texts)*1000:.2f}毫秒")
        
        # 测试数据加载性能
        print("\n测试数据加载性能...")
        loader = DataLoader(data_config)
        
        start_time = time.time()
        industry_df = loader.load_industry_data()
        load_time = time.time() - start_time
        print(f"加载 {len(industry_df)} 条行业数据耗时: {load_time:.2f}秒")
        
        start_time = time.time()
        chunks = loader.load_enterprise_chunks()
        load_time = time.time() - start_time
        total_enterprises = sum(len(chunk[1]) for chunk in chunks)
        print(f"加载 {total_enterprises} 条企业数据耗时: {load_time:.2f}秒")
        
        print("\n性能测试完成！")
        
    finally:
        shutil.rmtree(temp_dir)

def main():
    """主函数"""
    print("企业经营范围与行业分类匹配系统 - 完整功能测试")
    print("="*60)
    
    # 检查依赖
    print("检查依赖包...")
    try:
        import torch
        import faiss
        import sentence_transformers
        print("✓ 所有依赖包已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements_bge_faiss.txt")
        return 1
    
    # 运行单元测试
    print("\n运行单元测试...")
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 运行性能测试
    if result.wasSuccessful():
        run_performance_test()
    
    # 输出测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n🎉 所有测试通过！项目功能正常。")
        print("\n下一步:")
        print("1. 准备真实数据文件")
        print("2. 运行 python main.py 开始匹配")
        print("3. 查看生成的结果文件")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == '__main__':
    exit(main())