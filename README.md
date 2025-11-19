# JupyterProject - 大语言模型学习环境 (RTX 2070 优化版)

## 🎯 项目概述

这是一个专为大型语言模型（LLM）学习配置的 PyTorch 环境项目，针对 **RTX 2070 (8GB VRAM)** 笔记本进行了特别优化。本项目集成了 **SiliconFlow DeepSeek V3.2 Exp** 在线 API，让您既能学习本地模型（如 GPT-2）的底层原理，又能体验最先进大模型的强大能力。

## 🏗️ 当前配置状态

### ✅ 已成功配置
- **Python 3.12+**（现代 ML 库必需）
- **PyTorch 2.x + CUDA**（支持 RTX 2070 加速）
- **JupyterLab** 用于交互式开发
- **核心 ML 库**：transformers, datasets, tokenizers
- **在线大模型集成**：DeepSeek V3.2 Exp (via SiliconFlow)
- **离线模型系统**：网络受限时的自动回退机制

### ⚠️ GPU 注意事项
RTX 2070 拥有 8GB 显存，非常适合：
- 推理和微调 1B 以下参数的小模型（如 GPT-2, TinyLlama）
- 使用量化技术（4-bit/8-bit）运行 7B 级别的模型
- 学习 Transformer 架构、Attention 机制等核心概念

对于超大模型（如 70B+），本项目提供了 **DeepSeek V3.2 Exp** 的 API 接口，让您无需本地算力也能进行高级实验（如 RAG、复杂推理）。

## 🚀 快速入门

### 1. 配置 API Key (推荐)
为了使用强大的 DeepSeek 模型，请编辑 `llm_toy/configs/llm_api_config.json`：

```json
{
  "provider": "siliconflow",
  "siliconflow": {
    "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxx",
    "base_url": "https://api.siliconflow.cn/v1",
    "default_model": "deepseek-ai/DeepSeek-V3.2-Exp"
  }
}
```
或者设置环境变量 `SILICONFLOW_API_KEY`。

### 2. 测试环境
```bash
cd llm_toy
python main.py --test all
```

### 3. 启动 JupyterLab
```bash
jupyter lab
```

### 4. 开始学习
按顺序学习这些 Notebook（全中文注释）：

1. `llm_toy/notebooks/00_setup_troubleshooting.ipynb` - **环境自检**：修复 GPU 问题
2. `llm_toy/notebooks/01_pytorch_setup.ipynb` - **PyTorch 基础**：验证 GPU 加速
3. `llm_toy/notebooks/02_simple_llm_demo.ipynb` - **LLM 初体验**：本地 GPT-2 vs 在线 DeepSeek
4. `llm_toy/notebooks/03_training_demo.ipynb` - **模型训练**：从零训练一个小模型
5. `llm_toy/notebooks/07_rag_intro.ipynb` - **RAG 实战**：检索增强生成 (DeepSeek 加持)
6. `llm_toy/notebooks/04_fine_tuning_demo.ipynb` - **微调实战**：让模型学会特定风格
7. `llm_toy/notebooks/05_attention_visualization.ipynb` - **可视化**：看见 Attention 注意力
8. `llm_toy/notebooks/06_tokenization_basics.ipynb` - **分词原理**：训练 BPE Tokenizer
9. `llm_toy/notebooks/08_evaluation_metrics.ipynb` - **评测指标**：BLEU, Perplexity

## 📚 项目结构

```
llm_toy/
├── src/                    # 核心代码
│   ├── model.py           # 本地模型包装器
│   ├── online_model.py    # 在线 API 客户端 (DeepSeek)
│   ├── offline_model.py   # 离线回退模型
│   └── trainer.py         # 训练工具
├── notebooks/             # 学习教程 (Notebooks)
├── configs/               # 配置文件
├── data/                  # 数据集
└── main.py               # 测试脚本
```

## 🎓 学习路径

### 初学者 (CPU/GPU 皆可)
- **文本生成**：使用预训练 GPT-2
- **参数调优**：理解 Temperature, Top-k, Top-p
- **在线体验**：对比本地小模型与 DeepSeek 大模型的差距

### 进阶 (推荐 GPU)
- **模型微调**：在特定数据集上 Fine-tune GPT-2
- **RAG 系统**：搭建知识库问答系统
- **分词器训练**：理解 Tokenizer 如何工作

### 高级 (深入理解)
- **Attention 可视化**：探究模型内部机制
- **自定义架构**：修改 Transformer 结构
- **评测体系**：如何科学评价模型好坏

## 🔧 常见问题

### 1. 显存不足 (OOM)
- **现象**：`CUDA out of memory`
- **解决**：
    - 减小 `batch_size` (例如从 8 降到 4 或 1)
    - 减小 `max_length` (序列长度)
    - 使用 `torch.cuda.empty_cache()` 清理缓存

### 2. 网络连接问题
- **现象**：无法下载 Hugging Face 模型
- **解决**：
    - 本项目内置了离线回退机制，会自动切换到本地 demo 模型
    - 检查网络或配置代理
    - 使用 DeepSeek API 替代本地大模型下载

## 🌐 在线模型系统 (DeepSeek)

本项目封装了统一的接口，让您可以像使用本地模型一样使用在线 API：

```python
from online_model import create_online_model

# 自动读取配置文件中的 Key 和 Model ID
model = create_online_model(provider="siliconflow")

response = model.generate_text("解释一下 Transformer 的原理")
print(response)
```

---

**🎉 祝您在 LLM 的世界探索愉快！**
