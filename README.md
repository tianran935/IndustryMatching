# 企业经营范围与行业分类匹配系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![BGE Model](https://img.shields.io/badge/Model-BGE--small--zh--v1.5-orange.svg)](https://huggingface.co/BAAI/bge-small-zh-v1.5)
[![FAISS](https://img.shields.io/badge/Search-FAISS-red.svg)](https://github.com/facebookresearch/faiss)

## 📋 项目概述

本项目是一个基于**BGE模型**和**FAISS索引**的企业经营范围与行业分类智能匹配系统。通过深度学习和向量检索技术，实现高效、准确的文本匹配，帮助企业快速找到对应的行业分类代码。

### 🎯 核心功能

- **智能匹配**: 基于语义相似度的企业经营范围与行业分类匹配
- **高性能检索**: 使用FAISS向量数据库，支持毫秒级检索
- **批量处理**: 支持大规模企业数据的批量匹配处理
- **结果导出**: 自动生成详细的匹配结果CSV报告
- **可配置**: 灵活的配置管理，支持多种参数调优

## ✨ 主要特性

- **🚀 高性能模型**: 使用BAAI/bge-small-zh-v1.5模型，相比原版模型大小减少4.4倍
- **⚡ 快速检索**: 集成FAISS索引，搜索速度提升5-20倍
- **🏗️ 模块化架构**: 清晰的代码结构，易于维护和扩展
- **💾 智能缓存**: 减少重复计算，提升整体性能
- **📝 完善日志**: 详细的日志记录，便于调试和监控
- **📊 批量处理**: 支持大规模数据的高效处理
- **🔧 易于配置**: JSON配置文件，支持灵活参数调整

## 📁 项目结构

```
.
├── README.md                    # 项目说明文档
├── requirements.txt              # 生产环境依赖包
├── requirements_bge_faiss.txt    # 原始依赖包列表
├── .gitignore                    # Git忽略文件配置
├── config.py                     # 配置管理模块
├── config.json                   # 运行时配置文件
├── config_example.json           # 配置文件示例
├── logger.py                     # 日志管理模块
├── data_processor.py             # 数据处理模块
├── matcher.py                    # 匹配器模块
├── main.py                       # 主程序入口
├── test_optimized.py             # 优化版测试文件
├── test_bge_faiss.py             # 环境测试脚本
├── test.py                       # 原始版本测试
├── test_project.py               # 项目测试脚本
├── BGE_FAISS_优化说明.md          # 详细优化说明
├── data/                         # 数据目录
│   └── 国民经济分类_2.14.csv      # 行业分类标准数据
├── chunks/                       # 企业数据分块目录
├── logs/                         # 日志文件目录
└── joblib_cache/                 # 缓存目录
```

## 🏗️ 核心模块说明

### 1. 配置管理 (config.py)

- `ModelConfig`: 模型相关配置（模型名称、批处理大小、设备等）
- `FAISSConfig`: FAISS索引配置（索引类型、归一化等）
- `ProcessingConfig`: 数据处理配置（批处理大小、缓存设置等）
- `DataConfig`: 数据文件配置（文件路径、输出设置等）
- `LogConfig`: 日志配置（日志级别、输出格式等）

### 2. 日志管理 (logger.py)

- 统一的日志记录接口
- 支持控制台和文件输出
- 可配置的日志级别和格式
- 异常追踪和性能监控

### 3. 数据处理 (data_processor.py)

- `TextCleaner`: 文本清理和预处理
- `DataLoader`: 数据文件加载和验证
- `BusinessScopeProcessor`: 经营范围文本处理
- `DataProcessor`: 数据处理主类

### 4. 匹配器 (matcher.py)

- `ModelManager`: BGE模型管理和加载
- `FAISSIndexManager`: FAISS索引构建和管理
- `OptimizedMatcher`: 优化的匹配器主类

### 5. 主程序 (main.py)

- `JobMatcher`: 工作匹配器主类
- 完整的匹配流程控制
- 命令行参数支持
- 性能统计和报告

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 内存: 建议8GB以上
- 存储: 至少2GB可用空间
- GPU: 可选，支持CUDA加速

### 安装依赖

#### 方法1: 使用requirements文件（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd enterprise-industry-matching

# 安装依赖
pip install -r requirements.txt
```

#### 方法2: 手动安装核心依赖

```bash
pip install torch sentence-transformers faiss-cpu pandas numpy tqdm joblib
```

#### GPU版本（可选，性能更好）

```bash
# 先安装基础依赖
pip install -r requirements.txt

# 替换为GPU版本的FAISS
pip install faiss-gpu --force-reinstall
```

### 数据准备

1. **行业分类数据**: 确保 `./data/国民经济分类_2.14.csv` 文件存在
2. **企业数据**: 将企业数据文件放置在 `./chunks/` 目录下

### 配置设置

复制配置示例文件并根据需要修改：

```bash
cp config_example.json config.json
```

主要配置项：

```json
{
  "model": {
    "name": "BAAI/bge-small-zh-v1.5",
    "batch_size": 512,
    "device": "auto"
  },
  "processing": {
    "batch_size": 2000
  },
  "data": {
    "industry_file": "./data/国民经济分类_2.14.csv",
    "chunks_dir": "./chunks",
    "output_file": "JobMatching_Results_BGE_FAISS.csv"
  }
}
```

## 📖 使用方法

### 1. 环境测试

首先运行测试脚本确保环境正常：

```bash
python test_bge_faiss.py
```

### 2. 运行匹配程序

#### 基础用法

```bash
# 使用默认配置
python main.py

# 指定配置文件
python main.py --config custom_config.json

# 指定输出文件
python main.py --output custom_results.csv
```

#### 高级用法

```bash
# 启用详细日志
python main.py --log-level DEBUG

# 指定设备
python main.py --device cuda

# 自定义批处理大小
python main.py --batch-size 1000
```

### 3. 查看结果

匹配完成后，结果将保存在指定的CSV文件中，包含以下字段：

| 字段名   | 描述               |
| -------- | ------------------ |
| 企业名称 | 企业名称           |
| 企业代码 | newcid             |
| 经营范围 | 企业经营范围描述   |
| 匹配行业 | 匹配到的行业名称   |
| 行业代码 | 对应的行业分类代码 |
| 门类代码 | 行业门类代码       |
| 大类代码 | 行业大类代码       |
| 大类     | 行业大类名称       |
| 相似度   | 匹配相似度分数     |
| 数据来源 | 数据来源标识       |

## 🔧 API 文档

### JobMatcher 类

```python
from main import JobMatcher
from config import default_config

# 初始化匹配器
matcher = JobMatcher(default_config)

# 运行匹配
results = matcher.run_matching()

# 获取统计信息
stats = matcher.get_statistics()
```

### OptimizedMatcher 类

```python
from matcher import OptimizedMatcher
from config import default_config

# 初始化匹配器
matcher = OptimizedMatcher(default_config.model, default_config.faiss)

# 构建索引
matcher.build_index(industry_texts)

# 执行匹配
results = matcher.match_batch(business_scopes, top_k=5)
```

## 📊 性能优化

### 批处理大小调优

根据您的硬件配置调整批处理大小：

```json
{
  "model": {
    "batch_size": 512  // GPU内存充足时可增大
  },
  "processing": {
    "batch_size": 2000  // 根据CPU和内存调整
  }
}
```

### 内存优化

- 启用缓存: 设置合适的 `cache_dir` 和 `max_cache_size`
- 分批处理: 对于大型数据集，使用较小的批处理大小
- GPU加速: 安装 `faiss-gpu` 以获得更好的性能

### 性能基准

在标准配置下的性能表现：

| 数据规模 | 处理时间 | 内存使用 | 准确率 |
| -------- | -------- | -------- | ------ |
| 1K企业   | ~5秒     | ~2GB     | >90%   |
| 10K企业  | ~30秒    | ~4GB     | >90%   |
| 100K企业 | ~5分钟   | ~8GB     | >90%   |

## 🧪 测试

运行测试套件：

```bash
# 基础功能测试
python test_bge_faiss.py

# 性能测试
python test_optimized.py

# 项目完整测试
python test_project.py
```

## 📝 日志

系统提供详细的日志记录：

- **控制台日志**: 实时显示处理进度
- **文件日志**: 保存在 `logs/` 目录下
- **性能日志**: 记录处理时间和资源使用情况

日志级别：

- `DEBUG`: 详细调试信息
- `INFO`: 一般信息（默认）
- `WARNING`: 警告信息
- `ERROR`: 错误信息

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 使用 `black` 进行代码格式化
- 使用 `flake8` 进行代码检查
- 使用 `mypy` 进行类型检查
- 添加适当的文档字符串和注释

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) - 优秀的中文文本嵌入模型
- [Facebook FAISS](https://github.com/facebookresearch/faiss) - 高效的相似性搜索库
- [Sentence Transformers](https://www.sbert.net/) - 便捷的句子嵌入框架

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue: [GitHub Issues](../../issues)
- 邮箱: [your-email@example.com]

## 🔄 更新日志

### v1.2.0 (最新)

- ✨ 增加批处理大小优化
- 🐛 修复 SentenceTransformer 兼容性问题
- 📊 改进结果输出格式
- 🚀 性能提升 15-20%

### v1.1.0

- ✨ 添加 FAISS 索引支持
- 🏗️ 重构代码架构
- 📝 完善日志系统
- 🔧 添加配置管理

### v1.0.0

- 🎉 初始版本发布
- 🤖 基础 BGE 模型集成
- 📊 企业行业匹配功能
- 📁 批量数据处理支持

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！**
