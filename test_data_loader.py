import os
import pandas as pd
import joblib
import pytest
from pathlib import Path

import data_processor as dp
from config import DataConfig, LogConfig
import logger as app_logger


@pytest.fixture(autouse=True)
def _init_logger():
    # 显式创建项目根目录下的 ./chunks 目录
    os.makedirs("./chunks", exist_ok=True)
    # 在测试开始前初始化日志，避免 Logger not initialized 错误
    app_logger.setup_logger(LogConfig(level='INFO', file=None, console=False))
    yield


def _make_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        '经营范围': [f'scope{i}' for i in range(n)],
        '其他列': list(range(n)),
    })


def test_load_enterprise_chunks_creates_joblib_chunks(tmp_path, monkeypatch):
    # Arrange：把输出目录固定为项目根目录的 ./chunks
    chunks_dir = Path("./chunks")

    # 清理已有的 chunk_*.joblib，确保断言稳定
    for fp in chunks_dir.glob("chunk_*.joblib"):
        try:
            fp.unlink()
        except FileNotFoundError:
            pass

    cfg = DataConfig(chunks_dir=str(chunks_dir), input_file="./input.dta")
    loader = dp.DataLoader(cfg)

    # Mock: 输入文件存在 + 读取 .dta 返回 2500 行数据 + dump 指向 joblib.dump（原文件里未显式导入 dump）
    monkeypatch.setattr(dp.os.path, 'isfile', lambda p: True)
    monkeypatch.setattr(dp.pd, 'read_stata', lambda p: _make_df(2500))
    monkeypatch.setattr(dp, 'dump', joblib.dump, raising=False)

    # Act
    ret = loader.load_enterprise_chunks()

    # Assert: 函数在正常情况下不返回数据（兼容新接口），但会落盘分块到 ./chunks
    assert chunks_dir.exists()
    files = sorted(os.listdir(str(chunks_dir)))
    assert files == ['chunk_0.joblib', 'chunk_1.joblib', 'chunk_2.joblib']

    # 校验分块大小
    df0 = joblib.load(str(chunks_dir / 'chunk_0.joblib'))
    df1 = joblib.load(str(chunks_dir / 'chunk_1.joblib'))
    df2 = joblib.load(str(chunks_dir / 'chunk_2.joblib'))
    assert len(df0) == 1000
    assert len(df1) == 1000
    assert len(df2) == 500
    assert '经营范围' in df0.columns


if __name__ == '__main__':
    # 建议使用 pytest 运行：D:/python/anaconda/python.exe -m pytest -q test_data_loader.py
    pass