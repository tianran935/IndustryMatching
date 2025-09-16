import json
import os
import subprocess
import shutil

def run_split_stata_to_csv(year):
    print(f"正在为年份 {year} 运行 split_stata_to_csv.py...")
    try:
        result = subprocess.run(['python', 'split_stata_to_csv.py', str(year)], check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"年份 {year} 的 split_stata_to_csv.py 运行成功。")
        print("输出:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("split_stata_to_csv.py 运行失败。")
        print(f"返回码: {e.returncode}")
        print("错误输出:")
        print(e.stderr)
    except FileNotFoundError:
        print("错误: 'python' 命令未找到。请确保 Python 已经安装并且在系统的 PATH 中。")

def run_main_for_year(year):
    config_path = 'config.json'

    # 读取当前的 config.json
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 修改 output_file
    new_output_file = f"I:\\工商注册数据_分年\\{year}.dta\\result.csv"
    config['data']['output_file'] = new_output_file

    # 写回 config.json
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"为年份 {year} 更新配置文件，输出到 {new_output_file}")

    # 运行 main.py
    print(f"正在为年份 {year} 运行 main.py...")
    try:
        result = subprocess.run(['python', 'main.py'], check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"年份 {year} 的 main.py 运行成功。")
        print("输出:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"年份 {year} 的 main.py 运行失败。")
        print(f"返回码: {e.returncode}")
        print("错误输出:")
        print(e.stderr)
    except FileNotFoundError:
        print("错误: 'python' 命令未找到。请确保 Python 已经安装并且在系统的 PATH 中。")

def delete_chunks_directory():
    chunks_dir = 'chunks'
    if os.path.isdir(chunks_dir):
        print(f"正在删除目录: {chunks_dir}")
        try:
            shutil.rmtree(chunks_dir)
            print(f"目录 {chunks_dir} 删除成功。")
        except OSError as e:
            print(f"删除目录 {chunks_dir} 失败: {e}")
    else:
        print(f"目录 {chunks_dir} 不存在，无需删除。")


if __name__ == '__main__':
    original_config = None
    config_path = 'config.json'

    # 保存原始配置
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            original_config = f.read()

    try:
        for year in range(2013, 2024):
            print(f"\n{'='*20} 开始处理年份: {year} {'='*20}")
            
            # 1. 运行 split_stata_to_csv.py
            run_split_stata_to_csv(year)
            
            # 2. 运行 main.py
            run_main_for_year(year)
            
            # 3. 删除 chunks 目录
            delete_chunks_directory()
            
            print(f"{'='*20} 年份: {year} 处理完成 {'='*20}")

    finally:
        # 恢复原始配置
        if original_config:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(original_config)
            print("\n已恢复原始 config.json 文件。")