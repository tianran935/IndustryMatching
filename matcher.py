#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
匹配器模块
负责BGE模型加载、FAISS索引构建和相似度匹配
"""

import os
import time
import numpy as np
import pandas as pd
import faiss
import torch
from typing import List, Tuple, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from functools import lru_cache

from config import ModelConfig, FAISSConfig
from logger import log_info, log_warning, log_error, log_exception, log_debug

class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """获取设备类型"""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                log_info(f"检测到CUDA，使用GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                log_info("未检测到CUDA，使用CPU")
        else:
            device = self.config.device
            log_info(f"使用指定设备: {device}")
        
        return device
    
    def load_model(self) -> SentenceTransformer:
        """加载BGE模型"""
        if self.model is not None:
            return self.model
        
        try:
            # 使用本地模型路径
            local_model_path = "C:\\Users\\tianr\\Desktop\\mj\\tr\\models\\BAAI\\models--BAAI--bge-small-zh-v1.5\\snapshots\\7999e1d3359715c523056ef9478215996d62a620"
            log_info(f"正在加载本地BGE模型: {local_model_path}")
            start_time = time.time()
            
            self.model = SentenceTransformer(
                local_model_path,
                device=self.device
            )
            
            # 设置最大序列长度
            if self.config.max_length:
                self.model.max_seq_length = self.config.max_length
            
            load_time = time.time() - start_time
            log_info(f"模型加载完成，耗时: {load_time:.2f}秒")
            
            return self.model
            
        except Exception as e:
            log_exception(f"模型加载失败: {e}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: Optional[int] = None, 
                    show_progress: bool = True) -> np.ndarray:
        """编码文本列表"""
        if not texts:
            return np.array([])
        
        if self.model is None:
            self.load_model()
        
        batch_size = batch_size or self.config.batch_size
        
        try:
            log_debug(f"开始编码 {len(texts)} 个文本，批次大小: {batch_size}")
            start_time = time.time()
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2标准化
            )
            
            encode_time = time.time() - start_time
            log_debug(f"编码完成，耗时: {encode_time:.2f}秒，平均: {encode_time/len(texts)*1000:.2f}ms/条")
            
            return embeddings
            
        except Exception as e:
            log_exception(f"文本编码失败: {e}")
            raise

class FAISSIndexManager:
    """FAISS索引管理器"""
    
    def __init__(self, config: FAISSConfig):
        self.config = config
        self.index = None
        self.dimension = None
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """构建FAISS索引"""
        if len(embeddings) == 0:
            raise ValueError("嵌入向量为空，无法构建索引")
        
        try:
            log_info(f"正在构建FAISS索引，向量数量: {len(embeddings)}")
            start_time = time.time()
            
            self.dimension = embeddings.shape[1]
            log_debug(f"向量维度: {self.dimension}")
            
            # L2标准化
            if self.config.normalize_l2:
                faiss.normalize_L2(embeddings)
            
            # 创建索引
            if self.config.index_type == 'IndexFlatIP':
                # 内积索引（适用于标准化向量）
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.config.index_type == 'IndexFlatL2':
                # L2距离索引
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"不支持的索引类型: {self.config.index_type}")
            
            # 添加向量到索引
            self.index.add(embeddings.astype(np.float32))
            
            build_time = time.time() - start_time
            log_info(f"FAISS索引构建完成，耗时: {build_time:.2f}秒")
            
            return self.index
            
        except Exception as e:
            log_exception(f"FAISS索引构建失败: {e}")
            raise
    
    def search(self, query_embeddings: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最相似的向量"""
        if self.index is None:
            raise RuntimeError("索引未构建，请先调用build_index()")
        
        try:
            # L2标准化查询向量
            if self.config.normalize_l2:
                faiss.normalize_L2(query_embeddings)
            
            # 执行搜索
            scores, indices = self.index.search(query_embeddings.astype(np.float32), k)
            
            return scores, indices
            
        except Exception as e:
            log_exception(f"FAISS搜索失败: {e}")
            raise

class OptimizedMatcher:
    """优化的匹配器"""
    
    def __init__(self, model_config: ModelConfig, faiss_config: FAISSConfig):
        self.model_config = model_config
        self.faiss_config = faiss_config
        
        self.model_manager = ModelManager(model_config)
        self.index_manager = FAISSIndexManager(faiss_config)
        
        self.industry_embeddings = None
        self.industry_data = None
        
        # 编码缓存
        self._encoding_cache = {}
    
    def initialize(self, industry_names: List[str], industry_data: pd.DataFrame) -> None:
        """初始化匹配器"""
        try:
            log_info("正在初始化匹配器...")
            
            # 加载模型
            self.model_manager.load_model()
            
            # 编码行业名称
            log_info("正在编码行业名称...")
            self.industry_embeddings = self.model_manager.encode_texts(
                industry_names, 
                show_progress=True
            )
            
            # 构建FAISS索引
            self.index_manager.build_index(self.industry_embeddings)
            
            # 保存行业数据
            self.industry_data = industry_data.copy()
            
            log_info("匹配器初始化完成")
            
        except Exception as e:
            log_exception(f"匹配器初始化失败: {e}")
            raise
    
    def batch_encode_with_cache(self, texts: List[str]) -> np.ndarray:
        """带缓存的批量编码"""
        if not texts:
            return np.array([])
        
        # 检查缓存
        uncached_texts = []
        uncached_indices = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            if text in self._encoding_cache:
                cached_embeddings[i] = self._encoding_cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 编码未缓存的文本
        if uncached_texts:
            log_debug(f"编码 {len(uncached_texts)} 个未缓存文本")
            new_embeddings = self.model_manager.encode_texts(
                uncached_texts, 
                show_progress=len(uncached_texts) > 100
            )
            
            # 更新缓存
            for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                self._encoding_cache[text] = embedding
                cached_embeddings[uncached_indices[i]] = embedding
        
        # 组装结果
        result_embeddings = np.array([cached_embeddings[i] for i in range(len(texts))])
        
        return result_embeddings
    
    def match_business_scopes(self, business_scopes: List[List[str]], 
                            top_k: int = 1) -> List[List[Dict]]:
        """匹配经营范围"""
        if not business_scopes:
            return []
        
        try:
            log_info(f"开始匹配 {len(business_scopes)} 个经营范围")
            start_time = time.time()
            
            all_results = []
            
            for scope_segments in tqdm(business_scopes, desc="匹配经营范围"):
                if not scope_segments:
                    all_results.append([])
                    continue
                
                # 编码经营范围片段
                scope_embeddings = self.batch_encode_with_cache(scope_segments)
                
                if len(scope_embeddings) == 0:
                    all_results.append([])
                    continue
                
                # FAISS搜索
                scores, indices = self.index_manager.search(scope_embeddings, k=top_k)
                
                # 构建结果
                scope_results = []
                for i, (segment_scores, segment_indices) in enumerate(zip(scores, indices)):
                    segment_text = scope_segments[i]
                    
                    segment_matches = []
                    for j in range(top_k):
                        if j < len(segment_scores) and segment_indices[j] != -1:
                            industry_idx = segment_indices[j]
                            similarity = float(segment_scores[j])
                            
                            industry_info = self.industry_data.iloc[industry_idx]
                            
                            match_result = {
                                'segment': segment_text,
                                'industry_code': industry_info['行业代码'],
                                'industry_name': industry_info['行业名称'],
                                'similarity': similarity,
                                'rank': j + 1
                            }
                            segment_matches.append(match_result)
                    
                    scope_results.extend(segment_matches)
                
                all_results.append(scope_results)
            
            match_time = time.time() - start_time
            log_info(f"匹配完成，耗时: {match_time:.2f}秒，平均: {match_time/len(business_scopes)*1000:.2f}ms/条")
            
            return all_results
            
        except Exception as e:
            log_exception(f"经营范围匹配失败: {e}")
            raise
    
    def get_best_match(self, business_scope: str, threshold: float = 0.5) -> Optional[Dict]:
        """获取单个经营范围的最佳匹配"""
        if not business_scope or not business_scope.strip():
            return None
        
        try:
            # 编码
            embedding = self.batch_encode_with_cache([business_scope])
            
            if len(embedding) == 0:
                return None
            
            # 搜索
            scores, indices = self.index_manager.search(embedding, k=1)
            
            if len(scores[0]) == 0 or indices[0][0] == -1:
                return None
            
            similarity = float(scores[0][0])
            
            if similarity < threshold:
                return None
            
            industry_idx = indices[0][0]
            industry_info = self.industry_data.iloc[industry_idx]
            
            return {
                'business_scope': business_scope,
                'industry_code': industry_info['行业代码'],
                'industry_name': industry_info['行业名称'],
                'similarity': similarity
            }
            
        except Exception as e:
            log_exception(f"单个匹配失败: {e}")
            return None
    
    def clear_cache(self) -> None:
        """清空编码缓存"""
        self._encoding_cache.clear()
        log_info("编码缓存已清空")
    
    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        return {
            'cache_size': len(self._encoding_cache),
            'model_loaded': self.model_manager.model is not None,
            'index_built': self.index_manager.index is not None,
            'industry_count': len(self.industry_data) if self.industry_data is not None else 0
        }