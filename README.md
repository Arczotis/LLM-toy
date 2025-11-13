# JupyterProject - LLM Learning Environment

## 🎯 Project Overview

This project has been configured as a comprehensive learning environment for Large Language Models (LLMs) with PyTorch, optimized for your RTX 2070 GPU. It serves dual purposes: general data science work and educational LLM experimentation.

## 🏗️ Current Setup Status

### ✅ Successfully Configured
- **Python 3.12+** (required for modern ML libraries)
- **PyTorch 2.7.1+cu118** (CUDA 11.8 support for RTX 2070)
- **JupyterLab** for interactive development
- **Core ML libraries**: transformers, datasets, tokenizers
- **Complete LLM toy project** with educational notebooks
- **Offline model system** for network-constrained environments

### ⚠️ GPU Status
There appears to be a CUDA driver version mismatch. The project can still run in **CPU mode** for learning purposes, with automatic fallback to offline models when network issues occur.

## 🚀 Quick Start Guide

### 1. Test Your Setup
```bash
cd llm_toy
python main.py --test all
```

### 2. Launch JupyterLab
```bash
jupyter lab
```

### 3. Start Learning
Work through the notebooks in this order:
1. `llm_toy/notebooks/00_setup_troubleshooting.ipynb` - Fix any GPU issues
2. `llm_toy/notebooks/01_pytorch_setup.ipynb` - Verify your setup
3. `llm_toy/notebooks/02_simple_llm_demo.ipynb` - Basic LLM usage
4. `llm_toy/notebooks/02_offline_llm_demo.ipynb` - Offline model demonstration
5. `llm_toy/notebooks/03_training_demo.ipynb` - 中文教程：小数据训练流程
6. `llm_toy/notebooks/04_fine_tuning_demo.ipynb` - 中文教程：指令风格小规模Fine-tuning
7. `llm_toy/notebooks/05_attention_visualization.ipynb` - 中文教程：Attention可视化
8. `llm_toy/notebooks/06_tokenization_basics.ipynb` - 中文教程：从零训练BPE Tokenizer
9. `llm_toy/notebooks/07_rag_intro.ipynb` - 中文教程：RAG检索增强（TF-IDF Retriever）
10. `llm_toy/notebooks/08_evaluation_metrics.ipynb` - 中文教程：Perplexity/BLEU/ROUGE评测基础

## 📚 LLM Toy Project Structure

```
llm_toy/
├── src/                    # Core implementations
│   ├── model.py           # GPT model wrapper + simple transformer
│   ├── offline_model.py   # Offline-friendly fallback models
│   ├── trainer.py         # Training utilities
│   └── utils.py           # Helper functions
├── notebooks/             # Learning notebooks
│   ├── 00_setup_troubleshooting.ipynb
│   ├── 01_pytorch_setup.ipynb
│   ├── 02_simple_llm_demo.ipynb
│   ├── 02_offline_llm_demo.ipynb
│   └── [More advanced notebooks]
├── configs/               # Configuration files
├── data/                  # Dataset storage
├── demo_offline_fallback.py # Offline model demonstration
├── test_offline_model.py    # Offline model testing
├── OFFLINE_MODEL_GUIDE.md   # Offline model documentation
└── main.py               # Quick test script
```

## 🎓 Learning Path

### Beginner (CPU-friendly)
- **Text Generation**: Use pre-trained GPT-2 models
- **Parameter Tuning**: Experiment with temperature, top-k, top-p
- **Basic Concepts**: Understand tokenization, attention, generation
- **Offline Operation**: Learn with fallback models when network unavailable

### Intermediate (GPU recommended)
- **Fine-tuning**: Adapt models to your data
- **Custom Training**: Train smaller models from scratch
- **Evaluation**: Learn perplexity, BLEU scores, etc.
- **Tokenization**: Train custom BPE tokenizers

### Advanced (GPU required)
- **Attention Visualization**: See what models learn
- **Custom Architectures**: Build your own transformers
- **RAG Systems**: Build retrieval-augmented generation
- **Optimization**: Speed up training and inference

## 🔧 GPU Troubleshooting

### Current Issue: CUDA Driver Mismatch
Your system shows:
- RTX 2070 detected
- PyTorch with CUDA 11.8 installed
- Driver/library version mismatch

### Solutions:
1. **Quick Fix**: Use CPU mode for learning (slower but works)
2. **Proper Fix**: Update NVIDIA drivers to match CUDA 11.8
3. **Alternative**: Reinstall PyTorch with matching CUDA version

### CPU Mode Instructions
All notebooks will automatically fall back to CPU mode. Training will be slower, but learning concepts remain identical.

## 📦 Installed Packages

### Core ML Stack
- `torch>=2.1.0` - PyTorch with CUDA 11.8 support
- `transformers>=4.35.0` - Hugging Face transformers
- `datasets>=2.14.0` - Dataset handling
- `tokenizers>=0.15.0` - Fast tokenization

### Training & Optimization
- `accelerate>=0.24.0` - Training acceleration
- `wandb>=0.16.0` - Experiment tracking (optional)
- `scikit-learn>=1.3.0` - ML utilities

### Development Environment
- `jupyterlab>=4.4.10` - Interactive development
- `matplotlib>=3.10.7` - Visualization
- `pandas>=2.3.3` - Data manipulation
- `numpy>=1.26` - Numerical computing
- `seaborn>=0.13.2` - Statistical visualization
- `tqdm>=4.64.0` - Progress bars

## 🎯 What You Can Learn

### Core Concepts
- **Transformer Architecture**: Attention mechanisms, positional encoding
- **Text Generation**: Sampling strategies, beam search
- **Model Training**: Loss functions, optimization, evaluation
- **Fine-tuning**: Transfer learning, domain adaptation
- **Tokenization**: BPE, wordpiece, subword algorithms

### Practical Skills
- **PyTorch**: Tensors, autograd, nn.Module
- **Hugging Face**: Model hub, datasets, tokenizers
- **Experiment Tracking**: Weights & Biases integration
- **GPU Computing**: CUDA, memory management, optimization
- **Offline Development**: Fallback systems, local model usage

## 🚀 Next Steps

1. **Immediate**: Run the setup troubleshooting notebook
2. **Short-term**: Complete the beginner notebook series
3. **Medium-term**: Try fine-tuning on your own data
4. **Long-term**: Build custom architectures and applications

## 📖 Recommended Resources

- [Hugging Face Course](https://huggingface.co/course/chapter1)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (original paper)

## 💡 Tips for Success

### Learning Approach
1. **Start Simple**: Use pre-trained models first
2. **Experiment Freely**: Try different parameters and settings
3. **Document Everything**: Keep notes on what works
4. **Ask Questions**: The field is constantly evolving

### Technical Tips
- **Memory Management**: Start with small models and batch sizes
- **Reproducibility**: Always set random seeds
- **Monitoring**: Use experiment tracking from the start
- **Community**: Join Hugging Face and PyTorch communities
- **Offline Mode**: Practice with fallback models for reliability

## 🆘 Getting Help

### Common Issues
- **CUDA Errors**: Check the troubleshooting notebook
- **Memory Issues**: Reduce batch size or model size
- **Model Downloads**: Check internet connection or use offline models
- **Performance**: Use mixed precision training
- **Network Issues**: Offline models provide seamless fallback

### Support Channels
- Project README and documentation
- Online communities (Reddit, Discord, Stack Overflow)
- Official documentation for PyTorch and Hugging Face

## 🌐 Offline Model System

### Features
- **Automatic Detection**: Detects network issues and falls back to demo models
- **Same API**: Offline models provide identical interface to real models
- **Contextual Responses**: Generates contextually relevant placeholder text
- **Educational Value**: Maintains learning experience without internet connectivity

### Usage
```python
from offline_model import create_model

# Automatic fallback (recommended)
model = create_model("gpt2")

# Force offline mode
model = create_model("gpt2", force_offline=True)

# Same API as real model
text = model.generate_text("AI is", max_length=50, temperature=0.7)
info = model.get_model_info()
```

---

**🎉 Your LLM learning journey starts here!**

Don't let the GPU issues discourage you - start with CPU mode and upgrade when you resolve the driver issues. The concepts you'll learn are identical!

---

中文学习导航：请参考 `llm_toy/README_CN.md` 和 `llm_toy/notebooks/03~06` 系列中文Notebook（注释与讲解为中文，专有名词保持英文）。