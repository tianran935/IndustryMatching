#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块
负责文本清理、数据加载和预处理
"""

import os
import re
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from joblib import Memory
from tqdm import tqdm
# 新增：导入 joblib 的 dump/load 以便在模块内直接使用
from joblib import dump, load

from config import ProcessingConfig, DataConfig
from logger import log_info, log_warning, log_error, log_exception

class TextCleaner:
    """文本清理器"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.drop_texts = set(config.drop_texts)
        
        # 预编译正则表达式
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!\*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.bracket_pattern = re.compile(r'[（）()【】\[\]{}「」『』〈〉《》]')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # 设置缓存
        if config.cache_dir:
            os.makedirs(config.cache_dir, exist_ok=True)
            self.memory = Memory(config.cache_dir, verbose=0)
            self.clean_text_cached = self.memory.cache(self._clean_text_impl, ignore=['self'])
        else:
            self.clean_text_cached = lru_cache(maxsize=config.max_cache_size)(self._clean_text_impl)
    
    def _clean_text_impl(self, text: str) -> str:
        """文本清理实现"""
        if not isinstance(text, str) or not text.strip():
            return ''
        
        # 转换为小写
        text = text.lower()
        
        # 删除HTML标签
        text = self.html_pattern.sub('', text)
        
        # 删除URL
        text = self.url_pattern.sub('', text)
        
        # 删除标点符号和括号
        text = re.sub(r'[，。！？；：、","（）【】\[\]{}「」『』〈〉《》]', '', text)
        
        # 删除特定括号内容
        text = self.bracket_pattern.sub('', text)
        
        # 删除数字
        text = self.number_pattern.sub('', text)
        
        # 删除多余空白字符
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def clean_text(self, text: str) -> str:
        """清理单个文本"""
        return self.clean_text_cached(text)
    
    def vectorized_clean(self, texts: List[str]) -> List[str]:
        """批量清理文本"""
        if not texts:
            return []
        
        # 转换为numpy数组进行向量化操作
        texts_array = np.array(texts, dtype=object)
        
        # 批量处理
        cleaned_texts = []
        for text in texts_array:
            cleaned = self.clean_text(text)
            # 过滤掉预定义的无用文本
            if cleaned and cleaned not in self.drop_texts:
                cleaned_texts.append(cleaned)
            else:
                cleaned_texts.append('')
        
        return cleaned_texts

class DataLoader:
    """数据加载器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def load_industry_data(self) -> pd.DataFrame:
        """加载行业分类数据"""
        try:
            log_info(f"正在加载行业分类数据: {self.config.industry_file}")
            
            df = pd.read_csv(self.config.industry_file, encoding=self.config.encoding)
            
            # 验证必要列 - 保留原始列名以便后续使用
            required_columns = ['门类', '大类', '小类', '类别名称', '大类类别']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"行业数据缺少必要列: {missing_columns}，实际列名: {list(df.columns)}")
            
            # 添加行业代码和行业名称的映射列（保持向后兼容）
            df['行业代码'] = df['小类']
            df['行业名称'] = df['类别名称']
            
            # 数据清理
            df = df.dropna(subset=['小类', '类别名称'])
            df = df.drop_duplicates(subset=['小类'])
            
            log_info(f"成功加载 {len(df)} 条行业分类数据")
            return df
            
        except Exception as e:
            log_exception(f"加载行业数据失败: {e}")
            raise
    
    def load_enterprise_chunks(self) -> List[Tuple[str, pd.DataFrame]]:
        """加载企业数据分块（从data目录读取原始xlsx，并用joblib分块）"""
        """改变接口，删除函数返回值"""
        try:
            # 直接读取配置中的 input_file（.dta 文件）
            input_file = getattr(self.config, 'input_file', None)
            if not input_file:
                raise FileNotFoundError("未配置 data.input_file，请在配置中指定 .dta 文件路径")
            if not os.path.isfile(input_file):
                raise FileNotFoundError(f"输入文件不存在: {input_file}")
            log_info(f"正在从原始 Stata 文件读取企业数据: {input_file}")
            
            # 读取 .dta（整个表）
            df = pd.read_stata(input_file)
            
            # 验证必要的列
            if 'jing_ying_fan_wei' not in df.columns:
                raise ValueError(f"原始数据缺少'jing_ying_fan_wei'列，实际列名: {list(df.columns)}")
            
            # 确保分块输出目录存在（沿用配置中的 chunks_dir）
            os.makedirs(self.config.chunks_dir, exist_ok=True)
            
            total = len(df)
            if total == 0:
                log_warning("原始数据为空，无数据可处理")
                return []
            
            # 设置分块大小与分块数
            chunk_size = 1000  # 可按需调整
            n_chunks = (total + chunk_size - 1) // chunk_size
            
            chunks: List[Tuple[str, pd.DataFrame]] = []
            total_records = 0
            
            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, total)
                chunk = df.iloc[start:end].copy()
                
                chunk_name = f"chunk_{i}.joblib"
                chunk_path = os.path.join(self.config.chunks_dir, chunk_name)
                
                # 将分块以 joblib 格式持久化到磁盘
                try:
                    dump(chunk, chunk_path)
                except Exception as e:
                    log_warning(f"保存分块失败 {chunk_name}: {e}")
                
                # # 加入返回列表（保持原有接口结构）
                # chunks.append((chunk_name, chunk))
                # total_records += len(chunk)
            
            # 释放内存
            del df
            
            log_info(f"成功生成并加载 {len(chunks)} 个数据分块，共 {total_records} 条记录")
            # return chunks
        except Exception as e:
            log_exception(f"加载企业数据失败: {e}")
            raise
   
    
    def save_results(self, results: List[Dict], output_file: Optional[str] = None) -> str:
        """保存匹配结果"""
        if output_file is None:
            output_file = self.config.output_file
        
        try:
            log_info(f"正在保存结果到: {output_file}")
            
            if not results:
                log_warning("没有结果需要保存")
                return output_file
            
            # 转换为DataFrame
            df = pd.DataFrame(results)
            
            # 保存到CSV
            df.to_csv(output_file, index=False, encoding=self.config.encoding)
            
            log_info(f"成功保存 {len(results)} 条结果到 {output_file}")
            return output_file
            
        except Exception as e:
            log_exception(f"保存结果失败: {e}")
            raise

class BusinessScopeProcessor:
    """经营范围处理器"""
    
    def __init__(self, text_cleaner: TextCleaner):
        self.text_cleaner = text_cleaner
    
    def process_business_scope(self, scope_text: str) -> List[str]:
        """处理单个经营范围文本"""
        if not isinstance(scope_text, str) or not scope_text.strip():
            return []
        
        # 按分隔符分割
        separators = ['；', ';', '，', ',', '。', '.', '、']
        segments = [scope_text]
        
        for sep in separators:
            new_segments = []
            for segment in segments:
                new_segments.extend(segment.split(sep))
            segments = new_segments
        
        # 清理每个片段
        cleaned_segments = []
        for segment in segments:
            cleaned = self.text_cleaner.clean_text(segment.strip())
            if cleaned and len(cleaned) > 2:  # 过滤太短的片段
                cleaned_segments.append(cleaned)
        
        return cleaned_segments
    
    def process_batch(self, scope_texts: List[str]) -> List[List[str]]:
        """批量处理经营范围"""
        results = []
        for scope_text in tqdm(scope_texts, desc="处理经营范围"):
            results.append(self.process_business_scope(scope_text))
        return results

class DataProcessor:
    """数据处理器主类"""
    
    def __init__(self, processing_config: ProcessingConfig, data_config: DataConfig):
        self.processing_config = processing_config
        self.data_config = data_config
        
        self.text_cleaner = TextCleaner(processing_config)
        self.data_loader = DataLoader(data_config)
        self.scope_processor = BusinessScopeProcessor(self.text_cleaner)
    
    def load_and_process_industry_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """加载并处理行业数据"""
        # 加载原始数据
        industry_df = self.data_loader.load_industry_data()
        
        # 处理行业名称
        log_info("正在处理行业名称...")
        industry_names = industry_df['行业名称'].tolist()
        cleaned_names = self.text_cleaner.vectorized_clean(industry_names)
        
        # 过滤空的清理结果
        valid_indices = [i for i, name in enumerate(cleaned_names) if name]
        industry_df_filtered = industry_df.iloc[valid_indices].copy()
        cleaned_names_filtered = [cleaned_names[i] for i in valid_indices]
        
        log_info(f"处理后保留 {len(cleaned_names_filtered)} 个有效行业名称")
        
        return industry_df_filtered, cleaned_names_filtered
    
    def load_enterprise_data(self) -> List[Tuple[str, pd.DataFrame]]:
        """加载企业数据"""
        return self.data_loader.load_enterprise_chunks()
    
    def save_results(self, results: List[Dict], output_file: Optional[str] = None) -> str:
        """保存结果"""
        return self.data_loader.save_results(results, output_file)
    
      