#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序入口
整合所有模块，提供完整的匹配流程
"""

import os
import time
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm

from config import Config, default_config
from logger import setup_logger, log_info, log_warning, log_error, log_exception
from data_processor import DataProcessor
from matcher import OptimizedMatcher

class JobMatcher:
    """工作匹配器主类"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 设置日志
        setup_logger(config.log)
        
        # 初始化组件
        self.data_processor = DataProcessor(config.processing, config.data)
        self.matcher = OptimizedMatcher(config.model, config.faiss)
        
        # 统计信息
        self.stats = {
            'total_enterprises': 0,
            'processed_enterprises': 0,
            'matched_enterprises': 0,
            'total_time': 0,
            'load_time': 0,
            'process_time': 0,
            'match_time': 0
        }
    
    def validate_environment(self) -> bool:
        """验证运行环境"""
        try:
            log_info("正在验证运行环境...")
            
            # 验证配置
            self.config.validate()
            
            # 检查必要的库
            import torch
            import faiss
            import sentence_transformers
            
            log_info(f"PyTorch版本: {torch.__version__}")
            log_info(f"FAISS版本: {faiss.__version__}")
            log_info(f"SentenceTransformers版本: {sentence_transformers.__version__}")
            
            # 检查CUDA
            if torch.cuda.is_available():
                log_info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    log_info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                log_info("CUDA不可用，将使用CPU")
            
            return True
            
        except Exception as e:
            log_exception(f"环境验证失败: {e}")
            return False
    
    def load_data(self) -> bool:
        """加载数据"""
        try:
            log_info("开始加载数据...")
            start_time = time.time()
            
            # 加载并处理行业数据
            self.industry_df, self.industry_names = self.data_processor.load_and_process_industry_data()
            
            # 加载企业数据
            self.enterprise_chunks = self.data_processor.load_enterprise_data()
            
            # 统计企业总数
            self.stats['total_enterprises'] = sum(len(chunk[1]) for chunk in self.enterprise_chunks)
            
            self.stats['load_time'] = time.time() - start_time
            log_info(f"数据加载完成，耗时: {self.stats['load_time']:.2f}秒")
            log_info(f"行业数量: {len(self.industry_names)}")
            log_info(f"企业数量: {self.stats['total_enterprises']}")
            
            return True
            
        except Exception as e:
            log_exception(f"数据加载失败: {e}")
            return False
    
    def initialize_matcher(self) -> bool:
        """初始化匹配器"""
        try:
            log_info("正在初始化匹配器...")
            start_time = time.time()
            
            self.matcher.initialize(self.industry_names, self.industry_df)
            
            init_time = time.time() - start_time
            log_info(f"匹配器初始化完成，耗时: {init_time:.2f}秒")
            
            return True
            
        except Exception as e:
            log_exception(f"匹配器初始化失败: {e}")
            return False
    
    def process_enterprises(self) -> List[Dict]:
        """处理企业数据"""
        all_results = []
        
        try:
            log_info("开始处理企业数据...")
            start_time = time.time()
            
            for chunk_name, chunk_df in tqdm(self.enterprise_chunks, desc="处理数据分块"):
                log_info(f"正在处理分块: {chunk_name} ({len(chunk_df)} 条记录)")
                
                # 处理经营范围 - 分批处理以减少内存使用
                business_scopes = chunk_df['经营范围'].fillna('').tolist()
                
                # 分批处理，避免内存问题
                batch_size = min(1000, len(business_scopes))  # 增大批次大小
                processed_scopes = []
                
                for i in range(0, len(business_scopes), batch_size):
                    batch = business_scopes[i:i+batch_size]
                    batch_processed = []
                    for scope_text in batch:
                        batch_processed.append(self.data_processor.scope_processor.process_business_scope(scope_text))
                    processed_scopes.extend(batch_processed)
                
                # 过滤空的经营范围
                valid_indices = [i for i, scope in enumerate(processed_scopes) if scope]
                
                if not valid_indices:
                    log_warning(f"分块 {chunk_name} 中没有有效的经营范围")
                    continue
                
                valid_scopes = [processed_scopes[i] for i in valid_indices]
                valid_chunk_df = chunk_df.iloc[valid_indices].copy()
                
                # 执行匹配 - 分批处理
                log_info(f"正在匹配 {len(valid_scopes)} 个有效经营范围...")
                match_results = []
                match_batch_size = min(500, len(valid_scopes))  # 进一步增大匹配批次
                
                for i in range(0, len(valid_scopes), match_batch_size):
                    batch_scopes = valid_scopes[i:i+match_batch_size]
                    batch_results = self.matcher.match_business_scopes(batch_scopes, top_k=1)
                    match_results.extend(batch_results)
                
                # 构建结果记录
                for i, (_, row) in enumerate(valid_chunk_df.iterrows()):
                    matches = match_results[i] if i < len(match_results) else []
                    
                    if matches:
                        # 取最佳匹配
                        best_match = max(matches, key=lambda x: x['similarity'])
                        
                        # 从行业数据中获取完整信息
                        industry_info = self.industry_df[self.industry_df['小类'] == best_match['industry_code']]
                        if not industry_info.empty:
                            industry_row = industry_info.iloc[0]
                            门类代码 = industry_row.get('门类', '')
                            大类代码 = industry_row.get('大类', '')
                            大类 = industry_row.get('大类类别', '')
                            行业小类 = industry_row.get('类别名称', '')
                        else:
                            门类代码 = ''
                            大类代码 = ''
                            大类 = ''
                            行业小类 = ''
                        
                        result_record = {
                            '企业代码': row.get('newgcid', ''),
                            '匹配行业': best_match['industry_name'],
                            '行业代码': best_match['industry_code'],
                            '门类代码': 门类代码,
                            '大类代码': 大类代码,
                            '大类': 大类,
                            '经营范围': row.get('经营范围', ''),
                            '相似度': best_match['similarity'],
                            'newgcid': '',  # 这个字段需要根据具体需求填充
                            '行业小类': 行业小类,
                            '企业名称': row.get('企业名称', ''),
                            '统一社会信用代码': row.get('newgcid', ''),
                            '匹配片段': best_match['segment'],
                            '数据来源': chunk_name
                        }
                        
                        all_results.append(result_record)
                        self.stats['matched_enterprises'] += 1
                    else:
                        # 无匹配结果
                        result_record = {
                            '企业代码': row.get('newgcid', ''),
                            '匹配行业': '',
                            '行业代码': '',
                            '门类代码': '',
                            '大类代码': '',
                            '大类': '',
                            '经营范围': row.get('经营范围', ''),
                            '相似度': 0.0,
                            'newgcid': '',
                            '行业小类': '',
                            '企业名称': row.get('企业名称', ''),
                            '统一社会信用代码': row.get('统一社会信用代码', ''),
                            '匹配片段': '',
                            '数据来源': chunk_name
                        }
                        
                        all_results.append(result_record)
                
                self.stats['processed_enterprises'] += len(valid_chunk_df)
                
                log_info(f"分块 {chunk_name} 处理完成，匹配 {len([r for r in match_results if r])} 条")
            
            self.stats['process_time'] = time.time() - start_time
            log_info(f"企业数据处理完成，耗时: {self.stats['process_time']:.2f}秒")
            
            return all_results
            
        except Exception as e:
            log_exception(f"企业数据处理失败: {e}")
            raise
    
    def save_results(self, results: List[Dict]) -> str:
        """保存结果"""
        try:
            output_file = self.data_processor.save_results(results)
            return output_file
            
        except Exception as e:
            log_exception(f"结果保存失败: {e}")
            raise
    
    def print_statistics(self) -> None:
        """打印统计信息"""
        log_info("=" * 60)
        log_info("处理统计信息:")
        log_info(f"总企业数量: {self.stats['total_enterprises']:,}")
        log_info(f"处理企业数量: {self.stats['processed_enterprises']:,}")
        log_info(f"匹配企业数量: {self.stats['matched_enterprises']:,}")
        log_info(f"匹配成功率: {self.stats['matched_enterprises']/max(self.stats['processed_enterprises'], 1)*100:.2f}%")
        log_info("")
        log_info("时间统计:")
        log_info(f"数据加载时间: {self.stats['load_time']:.2f}秒")
        log_info(f"数据处理时间: {self.stats['process_time']:.2f}秒")
        log_info(f"总耗时: {self.stats['total_time']:.2f}秒")
        log_info(f"平均处理速度: {self.stats['processed_enterprises']/max(self.stats['total_time'], 1):.2f} 条/秒")
        log_info("")
        log_info("缓存信息:")
        cache_info = self.matcher.get_cache_info()
        for key, value in cache_info.items():
            log_info(f"{key}: {value}")
        log_info("=" * 60)
    
    def run(self) -> bool:
        """运行完整的匹配流程"""
        try:
            log_info("开始执行工作匹配流程")
            total_start_time = time.time()
            
            # 1. 验证环境
            if not self.validate_environment():
                return False
            
            # 2. 加载数据
            if not self.load_data():
                return False
            
            # 3. 初始化匹配器
            if not self.initialize_matcher():
                return False
            
            # 4. 处理企业数据
            results = self.process_enterprises()
            
            # 5. 保存结果
            output_file = self.save_results(results)
            
            # 6. 统计信息
            self.stats['total_time'] = time.time() - total_start_time
            self.print_statistics()
            
            log_info(f"匹配流程完成！结果已保存到: {output_file}")
            return True
            
        except Exception as e:
            log_exception(f"匹配流程执行失败: {e}")
            return False
        finally:
            # 清理缓存
            if hasattr(self, 'matcher'):
                self.matcher.clear_cache()

def create_config_from_args(args) -> Config:
    """从命令行参数创建配置"""
    config = Config()
    
    if args.industry_file:
        config.data.industry_file = args.industry_file
    
    if args.chunks_dir:
        config.data.chunks_dir = args.chunks_dir
    
    if args.output_file:
        config.data.output_file = args.output_file
    
    if args.model_name:
        config.model.name = args.model_name
    
    if args.batch_size:
        config.model.batch_size = args.batch_size
    
    if args.device:
        config.model.device = args.device
    
    if args.log_level:
        config.log.level = args.log_level
    
    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='企业经营范围与行业分类匹配工具')
    
    # 数据文件参数
    parser.add_argument('--industry-file', type=str, 
                       help='行业分类文件路径')
    parser.add_argument('--chunks-dir', type=str,
                       help='企业数据分块目录')
    parser.add_argument('--output-file', type=str,
                       help='输出结果文件路径')
    
    # 模型参数
    parser.add_argument('--model-name', type=str,
                       help='BGE模型名称')
    parser.add_argument('--batch-size', type=int,
                       help='批处理大小')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'],
                       help='计算设备')
    
    # 其他参数
    parser.add_argument('--log-level', type=str, 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--config-file', type=str,
                       help='配置文件路径（JSON格式）')
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config_file and os.path.exists(args.config_file):
        import json
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = create_config_from_args(args)
    
    # 创建并运行匹配器
    matcher = JobMatcher(config)
    success = matcher.run()
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())