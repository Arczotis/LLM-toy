# LLM Toy Project（中文学习版）

本项目面向入门与进阶学习者，基于 PyTorch 与 Transformers，提供可运行的最小示例与动手练习。文档与注释尽量中文说明，遇到专有名词（如 GPT-2、Tokenizer、Fine-tuning、Attention、DataLoader 等）保持英文原名。

## 学习路径（建议顺序）

1. `notebooks/01_pytorch_setup.ipynb`：检查 PyTorch 与设备（CPU/GPU）是否可用。
2. `notebooks/02_simple_llm_demo.ipynb`：用预训练 GPT-2 做基本文本生成，熟悉 `temperature`、`top-k`、`top-p` 等采样参数。
3. `notebooks/03_training_demo.ipynb`：在小数据上跑通完整训练流程，理解 Tokenizer、Dataset、DataLoader、训练循环与保存。
4. `notebooks/04_fine_tuning_demo.ipynb`：对指令风格数据做小规模 Fine-tuning，体验“冻结/解冻”与风格拟合。
5. `notebooks/05_attention_visualization.ipynb`：提取并可视化 Attention，直观理解模型“关注”了哪些 token。
6. `notebooks/06_tokenization_basics.ipynb`：从零训练一个BPE Tokenizer，并与 GPT-2 Tokenizer 对比切分差异。

> 说明：首次加载模型可能需要联网从 Hugging Face Hub 下载权重；若无网络请确保本地已有缓存。

## 数据

- `data/tiny_corpus.txt`：极小toy语料。用于快速演示训练、分词与可视化。

建议逐步替换为你自己的领域文本，以观察训练/生成风格的变化。

## 常见实践要点

- 设定随机种子：保证实验可复现。
- 小步快跑：先用小 batch、小 epoch 跑通流程，再增加规模。
- 记录实验：适当记录参数与现象，利于对比与复盘。
- 及时可视化：Loss曲线、Attention heatmap 等帮助直观理解模型行为。

## 练习清单（动手为主）

- 更换 `temperature`、`top-k`、`top-p`，记录生成风格变化。
- 修改 `max_length` 与 `batch_size`，感受显存/速度/性能的权衡。
- 在 `04_fine_tuning_demo` 中切换“冻结/不冻结”策略，对比微调效率与效果。
- 在 `06_tokenization_basics` 中调大词表、更换语料，观察 token 粒度与下游表现的关系。

## 进阶方向

- 数据处理：清洗、去重、分块、去除噪声，构建更稳健的训练数据流水线。
- 评测：引入 perplexity、BLEU、ROUGE 等指标，系统地观察改动带来的影响。
- 推理优化：了解 quantization、FlashAttention、Speculative Decoding 等方法（本项目暂不内置，建议另行实验）。

祝学习顺利！

