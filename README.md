# JupyterProject - 大语言模型学习环境

## 🎯 项目概述

这是一个专为大型语言模型（LLM）学习配置的PyTorch环境项目，针对RTX 2070 GPU进行了优化。项目具有双重用途：既可用于通用数据科学工作，也可用于教育性LLM实验。

## 🏗️ 当前配置状态

### ✅ 已成功配置
- **Python 3.12+**（现代ML库必需）
- **PyTorch 2.7.1+cu118**（支持RTX 2070的CUDA 11.8）
- **JupyterLab**用于交互式开发
- **核心ML库**：transformers、datasets、tokenizers
- **完整的LLM玩具项目**和教学笔记本
- **离线模型系统**用于网络受限环境

### ⚠️ GPU状态
系统检测到CUDA驱动版本不匹配。项目仍可**以CPU模式运行**进行学习，网络问题时会自动回退到离线模型。

## 🚀 快速入门

### 1. 测试环境
```bash
cd llm_toy
python main.py --test all
```

### 2. 启动JupyterLab
```bash
jupyter lab
```

### 2.5 使用在线模型（OpenRouter / 硅基流动）

- 编辑 `llm_toy/configs/llm_api_config.json` 填写 API Key，或设置环境变量 `OPENROUTER_API_KEY` / `SILICONFLOW_API_KEY`。
- 快速测试（非流式）：
  - `cd llm_toy`
  - OpenRouter：`python main.py --test online --provider openrouter --model-id openrouter/auto`
  - 硅基流动：`python main.py --test online --provider siliconflow --model-id <你的模型ID>`

客户端实现：`llm_toy/src/online_model.py`（OpenAI兼容 `chat/completions` 接口）。

### 3. 开始学习
按顺序学习这些笔记本：
1. `llm_toy/notebooks/00_setup_troubleshooting.ipynb` - 修复GPU问题
2. `llm_toy/notebooks/01_pytorch_setup.ipynb` - 验证环境配置
3. `llm_toy/notebooks/02_simple_llm_demo.ipynb` - 基础LLM使用
4. `llm_toy/notebooks/02_offline_llm_demo.ipynb` - 离线模型演示
5. `llm_toy/notebooks/03_training_demo.ipynb` - 中文教程：小数据训练流程
6. `llm_toy/notebooks/04_fine_tuning_demo.ipynb` - 中文教程：指令风格小规模Fine-tuning
7. `llm_toy/notebooks/05_attention_visualization.ipynb` - 中文教程：Attention可视化
8. `llm_toy/notebooks/06_tokenization_basics.ipynb` - 中文教程：从零训练BPE Tokenizer
9. `llm_toy/notebooks/07_rag_intro.ipynb` - 中文教程：RAG检索增强（TF-IDF Retriever）
10. `llm_toy/notebooks/08_evaluation_metrics.ipynb` - 中文教程：Perplexity/BLEU/ROUGE评测基础

## 📚 LLM玩具项目结构

```
llm_toy/
├── src/                    # 核心实现
│   ├── model.py           # GPT模型包装器 + 简单Transformer
│   ├── offline_model.py   # 离线友好回退模型
│   ├── trainer.py         # 训练工具
│   └── utils.py           # 辅助函数
├── notebooks/             # 学习笔记本
│   ├── 00_setup_troubleshooting.ipynb
│   ├── 01_pytorch_setup.ipynb
│   ├── 02_simple_llm_demo.ipynb
│   ├── 02_offline_llm_demo.ipynb
│   └── [更多高级笔记本]
├── configs/               # 配置文件
├── data/                  # 数据集存储
├── demo_offline_fallback.py # 离线模型演示
├── test_offline_model.py    # 离线模型测试
├── OFFLINE_MODEL_GUIDE.md   # 离线模型文档
└── main.py               # 快速测试脚本
```

## 🎓 学习路径

### 初学者（适合CPU）
- **文本生成**：使用预训练GPT-2模型
- **参数调优**：实验温度、top-k、top-p采样
- **基础概念**：理解分词、注意力机制、生成
- **离线操作**：网络不可用时使用回退模型学习

### 中级（推荐GPU）
- **微调**：将模型适配到您的数据
- **自定义训练**：从头训练较小模型
- **评估**：学习困惑度、BLEU分数等
- **分词**：训练自定义BPE分词器

### 高级（需要GPU）
- **注意力可视化**：查看模型学习内容
- **自定义架构**：构建自己的Transformer
- **RAG系统**：构建检索增强生成
- **优化**：加速训练和推理

## 🔧 GPU故障排除

### 当前问题：CUDA驱动不匹配
您的系统显示：
- 检测到RTX 2070
- 已安装PyTorch与CUDA 11.8
- 驱动/库版本不匹配

### 解决方案：
1. **快速修复**：使用CPU模式学习（较慢但可用）
2. **正确修复**：更新NVIDIA驱动以匹配CUDA 11.8
3. **替代方案**：重新安装PyTorch以匹配CUDA版本

### CPU模式说明
所有笔记本将自动回退到CPU模式。训练会更慢，但学习概念完全相同。

## 📦 已安装包

### 核心ML栈
- `torch>=2.1.0` - 支持CUDA 11.8的PyTorch
- `transformers>=4.35.0` - Hugging Face Transformer
- `datasets>=2.14.0` - 数据集处理
- `tokenizers>=0.15.0` - 快速分词

### 训练与优化
- `accelerate>=0.24.0` - 训练加速
- `wandb>=0.16.0` - 实验跟踪（可选）
- `scikit-learn>=1.3.0` - ML工具

### 开发环境
- `jupyterlab>=4.4.10` - 交互式开发
- `matplotlib>=3.10.7` - 可视化
- `pandas>=2.3.3` - 数据处理
- `numpy>=1.26` - 数值计算
- `seaborn>=0.13.2` - 统计可视化
- `tqdm>=4.64.0` - 进度条

## 🎯 可以学习的内容

### 核心概念
- **Transformer架构**：注意力机制、位置编码
- **文本生成**：采样策略、束搜索
- **模型训练**：损失函数、优化、评估
- **微调**：迁移学习、领域适配
- **分词**：BPE、wordpiece、子词算法

### 实用技能
- **PyTorch**：张量、自动求导、nn.Module
- **Hugging Face**：模型中心、数据集、分词器
- **实验跟踪**：Weights & Biases集成
- **GPU计算**：CUDA、内存管理、优化
- **离线开发**：回退系统、本地模型使用

## 🚀 后续步骤

1. **立即**：运行设置故障排除笔记本
2. **短期**：完成初学者笔记本系列
3. **中期**：尝试在您自己的数据上进行微调
4. **长期**：构建自定义架构和应用程序

## 📖 推荐资源

- [Hugging Face课程](https://huggingface.co/course/chapter1)
- [PyTorch教程](https://pytorch.org/tutorials/)
- [图解Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)（原始论文）

## 💡 成功技巧

### 学习方法
1. **从简单开始**：先使用预训练模型
2. **自由实验**：尝试不同的参数和设置
3. **记录一切**：记录有效的方法
4. **提问**：该领域不断发展

### 技术技巧
- **内存管理**：从小模型和批量大小开始
- **可重现性**：始终设置随机种子
- **监控**：从一开始就使用实验跟踪
- **社区**：加入Hugging Face和PyTorch社区
- **离线模式**：使用回退模型练习可靠性

## 🆘 获取帮助

### 常见问题
- **CUDA错误**：查看故障排除笔记本
- **内存问题**：减小批量大小或模型大小
- **模型下载**：检查互联网连接或使用离线模型
- **性能**：使用混合精度训练
- **网络问题**：离线模型提供无缝回退

### 支持渠道
- 项目README和文档
- 在线社区（Reddit、Discord、Stack Overflow）
- PyTorch和Hugging Face官方文档

## 🌐 离线模型系统

### 特性
- **自动检测**：检测网络问题并回退到演示模型
- **相同API**：离线模型提供与真实模型相同的接口
- **上下文相关**：生成上下文相关的占位符文本
- **教育价值**：在没有互联网连接的情况下保持学习体验

### 用法
```python
from offline_model import create_model

# 自动回退（推荐）
model = create_model("gpt2")

# 强制离线模式
model = create_model("gpt2", force_offline=True)

# 与真实模型相同的API
text = model.generate_text("AI is", max_length=50, temperature=0.7)
info = model.get_model_info()
```

---

**🎉 您的LLM学习之旅从这里开始！**

不要让GPU问题阻碍您 - 从CPU模式开始，解决驱动问题后再升级。您将学习的概念完全相同！

---

**英文版本**：[README.md](README.md)

**中文学习导航**：另请参考 `llm_toy/README_CN.md` 和 `llm_toy/notebooks/03~06` 系列中文Notebook（注释与讲解为中文，专有名词保持英文）。
