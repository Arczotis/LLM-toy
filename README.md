# JupyterProject - LLM Learning Environment

## 🎯 Project Overview

This project has been configured as a comprehensive learning environment for Large Language Models (LLMs) with PyTorch, optimized for your RTX 2070 GPU.

## 🏗️ Current Setup Status

### ✅ Successfully Configured
- **Python 3.12** (downgraded from 3.13 for better PyTorch compatibility)
- **PyTorch 2.7.1+cu118** (CUDA 11.8 support for RTX 2070)
- **JupyterLab** for interactive development
- **Core ML libraries**: transformers, datasets, tokenizers
- **Complete LLM toy project** with educational notebooks

### ⚠️ GPU Status
There appears to be a CUDA driver version mismatch. The project can still run in **CPU mode** for learning purposes.

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

## 📚 LLM Toy Project Structure

```
llm_toy/
├── src/                    # Core implementations
│   ├── model.py           # GPT model wrapper + simple transformer
│   ├── trainer.py         # Training utilities
│   └── utils.py           # Helper functions
├── notebooks/             # Learning notebooks
│   ├── 00_setup_troubleshooting.ipynb
│   ├── 01_pytorch_setup.ipynb
│   ├── 02_simple_llm_demo.ipynb
│   └── [More advanced notebooks]
├── configs/               # Configuration files
├── data/                  # Dataset storage
└── main.py               # Quick test script
```

## 🎓 Learning Path

### Beginner (CPU-friendly)
- **Text Generation**: Use pre-trained GPT-2 models
- **Parameter Tuning**: Experiment with temperature, top-k, top-p
- **Basic Concepts**: Understand tokenization, attention, generation

### Intermediate (GPU recommended)
- **Fine-tuning**: Adapt models to your data
- **Custom Training**: Train smaller models from scratch
- **Evaluation**: Learn perplexity, BLEU scores, etc.

### Advanced (GPU required)
- **Attention Visualization**: See what models learn
- **Custom Architectures**: Build your own transformers
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
- `tqdm>=4.64.0` - Progress bars

## 🎯 What You Can Learn

### Core Concepts
- **Transformer Architecture**: Attention mechanisms, positional encoding
- **Text Generation**: Sampling strategies, beam search
- **Model Training**: Loss functions, optimization, evaluation
- **Fine-tuning**: Transfer learning, domain adaptation

### Practical Skills
- **PyTorch**: Tensors, autograd, nn.Module
- **Hugging Face**: Model hub, datasets, tokenizers
- **Experiment Tracking**: Weights & Biases integration
- **GPU Computing**: CUDA, memory management, optimization

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

## 🆘 Getting Help

### Common Issues
- **CUDA Errors**: Check the troubleshooting notebook
- **Memory Issues**: Reduce batch size or model size
- **Model Downloads**: Check internet connection
- **Performance**: Use mixed precision training

### Support Channels
- Project README and documentation
- Online communities (Reddit, Discord, Stack Overflow)
- Official documentation for PyTorch and Hugging Face

---

**🎉 Your LLM learning journey starts here!**

Don't let the GPU issues discourage you - start with CPU mode and upgrade when you resolve the driver issues. The concepts you'll learn are identical!