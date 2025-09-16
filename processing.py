import pandas as pd
import glob
import os
import re
from pypinyin import lazy_pinyin

# ========== 工具函数：把中文列名转为拼音 ==========
def normalize_colname(col):
    """
    将中文列名转为拼音（下划线分隔），并裁剪到 <=32 字符
    """
    # 转拼音
    pinyin = "_".join(lazy_pinyin(col))
    # 只保留字母数字下划线
    pinyin = re.sub(r'[^0-9a-zA-Z_]', '_', pinyin)
    # 截断到32字符以内（Stata限制）
    return pinyin[:32]

# ========== 1. 找到所有分块 csv 文件 ==========
csv_files = glob.glob("data/*.csv")
# 排序并排除可能已存在的合并文件（例如 data/merged.csv）以防重复读取
csv_files = sorted([p for p in csv_files if not p.lower().endswith('merged.csv')])

# ========== 2. 逐个读取并以分块追加写入到单个 CSV 文件，避免一次性载入全部数据 ==========
output_csv = "data/merged.csv"
first = True
# 删除已存在的合并文件，防止重复追加
if os.path.exists(output_csv):
    os.remove(output_csv)
    print(f"ℹ️ 发现已有 {output_csv}，已删除并将重新生成")

if not csv_files:
    raise SystemExit("没有找到任何 data/*.csv 文件可供合并")

print(f"ℹ️ 开始合并 {len(csv_files)} 个文件到 {output_csv}，分块大小=100000 行")
for i, f in enumerate(csv_files, start=1):
    print(f"  -> 正在处理 ({i}/{len(csv_files)}) {f}")
    for chunk in pd.read_csv(f, chunksize=100000):
        # 将 header 仅写入第一次追加
        chunk.to_csv(output_csv, mode="a", header=first, index=False)
        first = False

print(f"✅ CSV 合并完成：{output_csv}")

# 读取最终合并的 CSV 作为 DataFrame（后续处理沿用原逻辑）
final_df = pd.read_csv(output_csv)

# ========== 3. 列名映射（中文 → 拼音） ==========
col_map = {col: normalize_colname(col) for col in final_df.columns}
final_df = final_df.rename(columns=col_map)

# ========== 3.5 统一 object 列为字符串，并检测需要 strL 的列 ==========
obj_cols = final_df.select_dtypes(include=['object']).columns.tolist()
for c in obj_cols:
    final_df[c] = final_df[c].astype(str)

# 需要写为 strL 的列（Stata 14+：当文本长度可能超过 2045 时）
cols_strl = []
for c in obj_cols:
    try:
        max_len = final_df[c].str.len().max()
    except Exception:
        max_len = pd.Series(final_df[c].astype(str)).str.len().max()
    if pd.notna(max_len) and max_len > 2045:
        cols_strl.append(c)

# ========== 4. 保存为 Stata 文件（UTF-8，Stata 14+支持） ==========
if cols_strl:
    # 你的 pandas 版本要求 convert_strl 是可迭代对象（列名列表）
    final_df.to_stata("merged_data.dta", write_index=False, version=118, convert_strl=cols_strl)
else:
    final_df.to_stata("merged_data.dta", write_index=False, version=118)

print("✅ 合并完成，保存为 merged_data.dta")
print("📌 列名映射表：")
for k, v in col_map.items():
    print(f"{k} -> {v}")
