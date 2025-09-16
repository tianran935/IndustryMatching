import pandas as pd
import os
import argparse

def split_data(year):
    # 根据年份动态设置输入文件路径
    input_file = f"I:\\工商注册数据_分年\\{year}.dta\\{year}.dta"
    output_dir = "chunks"
    chunk_size = 1000  # 每1000条数据保存一个文件

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return

    # 分块读取Stata文件并保存为CSV
    chunk_number = 1
    try:
        for chunk in pd.read_stata(input_file, chunksize=chunk_size):
            output_path = os.path.join(output_dir, f"chunk_{chunk_number}.csv")
            chunk.to_csv(output_path, index=False, encoding='utf-8')
            print(f"已保存: {output_path} (行数: {len(chunk)})")
            chunk_number += 1
        print(f"年份 {year} 的数据分割完成！")
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从Stata文件分割数据到CSV块.')
    parser.add_argument('year', type=int, help='要处理的年份 (例如: 2012)')
    args = parser.parse_args()
    split_data(args.year)