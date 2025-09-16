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
from joblib import Parallel, delayed
from joblib import dump, load  # 新增：在本模块内直接使用 dump/load

from config import Config, default_config
from logger import setup_logger, get_logger, log_info, log_warning, log_error, log_exception
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
            chunks_dir = self.config.data.chunks_dir
            try:
                has_chunks = os.path.isdir(chunks_dir) and any(
                    f.endswith(".joblib") and not f.endswith("_result.joblib")
                    for f in os.listdir(chunks_dir)
                )
            except Exception:
                has_chunks = False
            
            if not has_chunks:
                log_info(f"chunks 目录不存在或为空，将生成/加载企业分块数据: {chunks_dir}")
                self.data_processor.load_enterprise_data()
            else:
                log_info(f"检测到现有分块文件，跳过企业数据加载: {chunks_dir}")
            
            # 统计企业总数
            # self.stats['total_enterprises'] = sum(len(chunk[1]) for chunk in self.enterprise_chunks)
            
            self.stats['load_time'] = time.time() - start_time
            log_info(f"数据加载完成，耗时: {self.stats['load_time']:.2f}秒")
            # log_info(f"行业数量: {len(self.industry_names)}")
            # log_info(f"企业数量: {self.stats['total_enterprises']}")
            
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
        # all_results = []
        # chunk_counter = 0
        # output_counter = 1
        
        try:
            log_info("开始处理企业数据...")
            start_time = time.time()

            chunks_dir = self.config.data.chunks_dir
            results_dir = os.path.join(chunks_dir, "results_joblib")
            os.makedirs(results_dir, exist_ok=True)

            if not os.path.isdir(chunks_dir):
                log_warning(f"chunks 目录不存在: {chunks_dir}")
                self.stats['process_time'] = time.time() - start_time
                return []

            # 枚举 chunks 目录下的 joblib 分块文件（排除已保存的结果文件）
            all_files = os.listdir(chunks_dir)
            chunk_files = [
                os.path.join(chunks_dir, f)
                for f in all_files
                if f.endswith(".joblib") and not f.endswith("_result.joblib")
            ]
            chunk_files.sort()

            if not chunk_files:
                log_warning(f"未在 {chunks_dir} 找到任何 .joblib 分块文件")
                self.stats['process_time'] = time.time() - start_time
                return []

            # 先处理断点续处理：已有结果的分块直接统计并跳过
            files_to_process = []
            for cf in chunk_files:
                chunk_name = os.path.basename(cf)
                result_path = os.path.join(results_dir, f"{os.path.splitext(chunk_name)[0]}_result.joblib")
                if os.path.exists(result_path):
                    try:
                        prev_results = load(result_path)
                        self.stats['processed_enterprises'] += len(prev_results)
                        matched_count = sum(
                            1 for r in prev_results
                            if r.get('行业代码') or (r.get('相似度', 0) and r.get('相似度', 0) > 0)
                        )
                        self.stats['matched_enterprises'] += matched_count
                        log_info(f"检测到已存在结果，跳过分块: {chunk_name} (已有 {len(prev_results)} 条)")
                    except Exception as e:
                        log_warning(f"加载已存在结果失败，将重新处理分块 {chunk_name}: {e}")
                        files_to_process.append(cf)
                else:
                    files_to_process.append(cf)

            # 单个分块处理函数（供并行调用）
            def _process_one_chunk(chunk_file: str):
                # 确保子进程中已初始化日志（Windows/loky 子进程为全新进程）
                try:
                    get_logger()
                except Exception:
                    try:
                        setup_logger(self.config.log)
                    except Exception:
                        pass
                
                chunk_name = os.path.basename(chunk_file)
                result_path = os.path.join(results_dir, f"{os.path.splitext(chunk_name)[0]}_result.joblib")
                try:
                    chunk_df = load(chunk_file)

                    # 处理jing_ying_fan_wei（分批）
                    business_scopes = chunk_df['jing_ying_fan_wei'].fillna('').tolist()
                    batch_size = min(1000, len(business_scopes))
                    processed_scopes = []
                    for i in range(0, len(business_scopes), batch_size):
                        batch = business_scopes[i:i+batch_size]
                        batch_processed = []
                        for scope_text in batch:
                            batch_processed.append(self.data_processor.scope_processor.process_business_scope(scope_text))
                        processed_scopes.extend(batch_processed)

                    # 过滤空的jing_ying_fan_wei
                    valid_indices = [i for i, scope in enumerate(processed_scopes) if scope]
                    if not valid_indices:
                        dump([], result_path)
                        log_warning(f"分块 {chunk_name} 中没有有效的经营范围")
                        return (chunk_name, 0, 0)

                    valid_scopes = [processed_scopes[i] for i in valid_indices]
                    valid_chunk_df = chunk_df.iloc[valid_indices].copy()

                    # 匹配（分批）
                    match_results = []
                    match_batch_size = min(500, len(valid_scopes))
                    for i in range(0, len(valid_scopes), match_batch_size):
                        batch_scopes = valid_scopes[i:i+match_batch_size]
                        batch_results = self.matcher.match_business_scopes(batch_scopes, top_k=1)
                        match_results.extend(batch_results)

                    # 构建并保存结果
                    per_chunk_results = []
                    matched_count_local = 0
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
                                '经营范围': row.get('jing_ying_fan_wei', ''),
                                '相似度': best_match['similarity'],
                                '行业小类': 行业小类,
                                '企业名称': row.get('qi_ye_ming_cheng', ''),
                                '匹配片段': best_match['segment'],
                                '数据来源': chunk_name
                            }
                            per_chunk_results.append(result_record)
                            matched_count_local += 1
                        else:
                            # 无匹配结果
                            result_record = {
                                '企业代码': row.get('newgcid', ''),
                                '匹配行业': '',
                                '行业代码': '',
                                '门类代码': '',
                                '大类代码': '',
                                '大类': '',
                                'jing_ying_fan_wei': row.get('经营范围', ''),
                                '相似度': 0.0,
                                '行业小类': '',
                                '企业名称': row.get('qi_ye_ming_cheng', ''),
                                '匹配片段': '',
                                '数据来源': chunk_name
                            }
                            per_chunk_results.append(result_record)

                    dump(per_chunk_results, result_path)
                    log_info(f"分块 {chunk_name} 结果已保存到: {result_path} (共 {len(per_chunk_results)} 条)")
                    return (chunk_name, len(valid_chunk_df), matched_count_local)
                except Exception as e:
                    log_error(f"保存分块结果失败/处理失败 {chunk_name}: {e}")
                    return (chunk_name, 0, 0)

            # 并行处理所有需要处理的分块
            if files_to_process:
                log_info(f"开始并行处理 {len(files_to_process)} 个分块 (n_jobs=4)...")
                results = Parallel(n_jobs=4)(
                    delayed(_process_one_chunk)(cf) for cf in files_to_process
                )
                for chunk_name, processed_count, matched_count in results:
                    self.stats['processed_enterprises'] += processed_count
                    self.stats['matched_enterprises'] += matched_count

            self.stats['process_time'] = time.time() - start_time
            chunk_counter = len(chunk_files)
            log_info(f"企业数据处理完成，耗时: {self.stats['process_time']:.2f}秒")
            log_info(f"总共输出了 {len(files_to_process)} 个新分块结果（joblib文件），已有 {len(chunk_files) - len(files_to_process)} 个分块已存在结果")
            return []  # 结果已按分块保存（joblib）
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
            
            # 4. 处理企业数据（结果已在处理过程中分批保存）
            self.process_enterprises()
            
            # 5. 汇总分块结果并保存到单个CSV
            results_dir = os.path.join(self.config.data.chunks_dir, "results_joblib")
            all_results = []
            if os.path.isdir(results_dir):
                result_files = [f for f in os.listdir(results_dir) if f.endswith("_result.joblib")]
                result_files.sort()
                for rf in result_files:
                    rp = os.path.join(results_dir, rf)
                    try:
                        part = load(rp)
                        if isinstance(part, list):
                            all_results.extend(part)
                        else:
                            log_warning(f"结果文件格式异常（非列表），已跳过: {rf}")
                    except Exception as e:
                        log_warning(f"加载分块结果失败，已跳过 {rf}: {e}")
            else:
                log_warning(f"未找到结果目录: {results_dir}")
            
            if all_results:
                output_path = self.save_results(all_results)
                log_info(f"已汇总并保存至CSV: {output_path}")
            else:
                log_warning("没有可汇总的分块结果，跳过CSV导出")
            
            # 6. 统计信息
            self.stats['total_time'] = time.time() - total_start_time
            self.print_statistics()
            
            log_info("匹配流程完成！结果已分批保存到多个文件")
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