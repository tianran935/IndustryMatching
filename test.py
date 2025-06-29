import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer, util
import torch
import re
import string
import time
from joblib import Memory
import csv
from functools import lru_cache
import faiss

memory = Memory("./joblib_cache", verbose=0)

# 预编译正则表达式以提高性能
HTML_PATTERN = re.compile(r'<.*?>')
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+', re.MULTILINE)
BRACKET_PATTERN = re.compile(r'【|】')
DIGIT_PATTERN = re.compile(r'\d+')
WHITESPACE_PATTERN = re.compile(r'\s+')

# 预定义要删除的文本
DROP_TEXTS = [
    '在中国法律允许', '依法须经批准的项目', '经相关部门批准后方可开展经营活动',
    '法律、法规、国务院决定规定应当许可（审批）的', '经相关部门批准后依批准的内容开展经营活动',
    '具体经营项目以审批结果为准', ''
]

@lru_cache(maxsize=10000)
def clean_text_cached(text):
    """缓存的文本清理函数"""
    if pd.isna(text) or text == '':
        return ''
    
    # 转换为小写
    text = str(text).lower()
    
    # 使用预编译的正则表达式
    text = HTML_PATTERN.sub('', text)
    text = URL_PATTERN.sub('', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = BRACKET_PATTERN.sub('', text)
    text = DIGIT_PATTERN.sub('', text)
    text = text.strip()
    text = WHITESPACE_PATTERN.sub(' ', text)
    
    # 批量删除不需要的文本
    for drop_text in DROP_TEXTS:
        if drop_text:
            text = text.replace(drop_text, '')
    
    return text.strip()

def vectorized_clean(series):
    """向量化的文本清理"""
    # 确保为字符串类型
    series = series.astype(str)
    
    # 向量化操作
    series = series.str.lower()
    series = series.str.replace(HTML_PATTERN, '', regex=True)
    series = series.str.replace(URL_PATTERN, '', regex=True)
    series = series.str.translate(str.maketrans('', '', string.punctuation))
    series = series.str.replace(BRACKET_PATTERN, '', regex=True)
    series = series.str.replace(DIGIT_PATTERN, '', regex=True)
    
    # 批量替换
    for drop_text in DROP_TEXTS:
        if drop_text:
            series = series.str.replace(drop_text, '', regex=False)
    
    series = series.str.strip()
    series = series.str.replace(WHITESPACE_PATTERN, ' ', regex=True)
    
    return series

@memory.cache
def process_business_scope_optimized(data_company):
    """优化的企业经营范围处理函数"""
    # 向量化文本清理
    data_company['经营范围'] = vectorized_clean(data_company['经营范围'])
    
    # 处理空值
    empty_mask = (data_company['经营范围'] == '') | data_company['经营范围'].isna()
    empty_business_scopes = data_company[empty_mask].copy()
    
    if not empty_business_scopes.empty:
        empty_business_scopes['匹配行业'] = np.where(
            empty_business_scopes['行业小类'] == '',
            np.nan,
            empty_business_scopes['行业小类']
        )
        empty_business_scopes['企业代码'] = empty_business_scopes['newgcid']
    
    # 非空数据
    data_nonempty_company = data_company[~empty_mask].copy()
    
    return data_nonempty_company, empty_business_scopes

class OptimizedMatcher:
    """使用BGE模型和FAISS的优化匹配器"""
    def __init__(self, model_name='BAAI/bge-small-zh-v1.5'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.industry_names = None
        self.industry_code = None
        self.industry_menlei = None
        self.industry_dalei = None
        self.industry_dalei_name = None
    
    def build_index(self, industry_names, industry_code, industry_menlei, industry_dalei, industry_dalei_name):
        """构建FAISS索引"""
        self.industry_names = industry_names
        self.industry_code = industry_code
        self.industry_menlei = industry_menlei
        self.industry_dalei = industry_dalei
        self.industry_dalei_name = industry_dalei_name
        
        print("编码行业数据并构建FAISS索引...")
        # 批量编码行业名称
        embeddings = self.model.encode(industry_names, 
                                     convert_to_tensor=False,
                                     show_progress_bar=True,
                                     batch_size=64)
        
        # 创建FAISS索引
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)  # L2标准化
        
        # 使用内积索引（适合标准化后的向量）
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        
        print(f"FAISS索引构建完成，包含 {len(industry_names)} 个行业")
    
    def batch_match(self, business_scopes, batch_size=1000):
        """批量匹配企业经营范围"""
        results = []
        
        print(f"开始批量匹配 {len(business_scopes)} 个企业经营范围...")
        for i in tqdm(range(0, len(business_scopes), batch_size), desc="批量匹配"):
            batch = business_scopes[i:i+batch_size]
            
            # 批量编码
            batch_embeddings = self.model.encode(batch, 
                                                convert_to_tensor=False,
                                                batch_size=64)
            batch_embeddings = np.array(batch_embeddings).astype('float32')
            faiss.normalize_L2(batch_embeddings)  # L2标准化
            
            # 批量搜索最相似的行业
            scores, indices = self.index.search(batch_embeddings, 1)
            
            # 收集结果
            for j, (score, idx) in enumerate(zip(scores, indices)):
                results.append({
                    'business_scope': batch[j],
                    'matched_industry': self.industry_names[idx[0]],
                    'industry_code': self.industry_code[idx[0]],
                    'industry_menlei': self.industry_menlei[idx[0]],
                    'industry_dalei': self.industry_dalei[idx[0]],
                    'industry_dalei_name': self.industry_dalei_name[idx[0]],
                    'similarity': float(score[0])
                })
        
        return results

def batch_encode_with_cache(model, texts, batch_size=32, device='cpu'):
    """批量编码并缓存结果（保留兼容性）"""
    # 去重文本以减少编码次数
    unique_texts = list(set(texts))
    
    # 批量编码
    embeddings_dict = {}
    for i in tqdm(range(0, len(unique_texts), batch_size), desc="批量编码文本"):
        batch_texts = unique_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, device=device)
        
        for text, embedding in zip(batch_texts, batch_embeddings):
            embeddings_dict[text] = embedding
    
    # 根据原始顺序返回嵌入
    return torch.stack([embeddings_dict[text] for text in texts])

def optimized_matching(data_nonempty_company, industry_embeddings, industry_names, 
                      industry_code, industry_menlei, industry_dalei, industry_dalei_name, 
                      model, device):
    """优化的匹配算法"""
    # 获取所有经营范围文本
    business_scopes = data_nonempty_company['经营范围'].tolist()
    
    # 批量编码企业经营范围
    print("批量编码企业经营范围...")
    business_embeddings = batch_encode_with_cache(model, business_scopes, device=device)
    
    # 批量计算相似度
    print("计算相似度矩阵...")
    similarities = util.pytorch_cos_sim(business_embeddings, industry_embeddings)
    
    # 找到最佳匹配
    best_matches = similarities.argmax(dim=1)
    best_scores = similarities.max(dim=1).values
    
    # 构建结果
    results = []
    for idx, (_, row) in enumerate(data_nonempty_company.iterrows()):
        match_idx = best_matches[idx].item()
        results.append({
            "企业代码": row['newgcid'],
            "匹配行业": industry_names[match_idx],
            "行业代码": industry_code[match_idx],
            "门类代码": industry_menlei[match_idx],
            "大类代码": industry_dalei[match_idx],
            "大类": industry_dalei_name[match_idx],
            "经营范围": row['经营范围'],
            "相似度": best_scores[idx].item()
        })
    
    return results

# 主程序优化
def main_optimized():
    """使用BGE模型和FAISS的优化主程序"""
    start_time = time.time()
    
    # 加载行业数据
    print("加载行业数据...")
    industry_names, industry_code, industry_menlei, industry_dalei, industry_dalei_name = [], [], [], [], []
    
    with open('./data/国民经济分类_2.14.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            industry_names.append(row['类别名称'])
            industry_code.append(row['小类'])
            industry_menlei.append(row.get('门类', ''))
            industry_dalei.append(row.get('大类', ''))
            industry_dalei_name.append(row.get('大类类别', ''))
    
    # 初始化优化匹配器
    print("初始化BGE模型和FAISS索引...")
    matcher = OptimizedMatcher('BAAI/bge-small-zh-v1.5')
    
    # 构建FAISS索引
    matcher.build_index(industry_names, industry_code, industry_menlei, industry_dalei, industry_dalei_name)
    
    # 读取企业数据（假设有一个大文件而不是分块）
    print("读取企业数据...")
    # 这里需要根据实际情况修改数据源
    # data_company = pd.read_csv('企业数据.csv')  # 替换为实际文件路径
    
    # 如果必须从chunks读取，可以先合并
    chunk_files = [os.path.join("chunks", f) for f in os.listdir("chunks") 
                   if f.endswith('.csv') and f.startswith('chunk_')]
    
    print("合并分块数据...")
    data_frames = []
    for chunk_file in tqdm(chunk_files, desc="读取分块"):
        df = pd.read_csv(chunk_file)
        data_frames.append(df)
    
    data_company = pd.concat(data_frames, ignore_index=True)
    print(f"总共加载了 {len(data_company)} 条企业数据")
    
    # 处理企业数据
    print("处理企业数据...")
    data_nonempty_company, empty_business_scopes = process_business_scope_optimized(data_company)
    
    # 处理空经营范围的记录
    empty_records = []
    if not empty_business_scopes.empty:
        for _, row in empty_business_scopes.iterrows():
            empty_records.append({
                "企业代码": row['企业代码'],
                "匹配行业": row['匹配行业'],
                "经营范围": row['经营范围'],
                "行业代码": np.nan,
                "门类代码": np.nan,
                "大类代码": np.nan,
                "大类": np.nan,
                "相似度": np.nan
            })
    
    # 使用FAISS优化匹配过程
    print("开始FAISS优化匹配过程...")
    business_scopes = data_nonempty_company['经营范围'].tolist()
    match_results = matcher.batch_match(business_scopes, batch_size=1000)
    
    # 构建匹配记录
    matched_records = []
    for idx, (_, row) in enumerate(data_nonempty_company.iterrows()):
        result = match_results[idx]
        matched_records.append({
            "企业代码": row['newgcid'],
            "匹配行业": result['matched_industry'],
            "行业代码": result['industry_code'],
            "门类代码": result['industry_menlei'],
            "大类代码": result['industry_dalei'],
            "大类": result['industry_dalei_name'],
            "经营范围": row['经营范围'],
            "相似度": result['similarity']
        })
    
    # 合并结果
    all_results = matched_records + empty_records
    
    # 保存结果
    print("保存结果...")
    final_df = pd.DataFrame(all_results)
    final_df.to_csv("JobMatching_Results_BGE_FAISS.csv", index=False, encoding='utf-8-sig')
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"BGE+FAISS优化后总耗时：{elapsed/60:.2f} 分钟（{elapsed:.2f} 秒）")
    print(f"处理了 {len(final_df)} 条记录")
    print(f"平均每条记录耗时：{elapsed/len(final_df)*1000:.2f} 毫秒")
    print("\n性能提升说明：")
    print("- 使用BGE-small-zh-v1.5模型，专为中文优化")
    print("- 使用FAISS索引进行快速相似度搜索")
    print("- 批量处理减少模型调用次数")
    print("- 预计比原始方案快5-20倍")

if __name__ == "__main__":
    main_optimized()