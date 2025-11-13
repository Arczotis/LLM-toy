# LLM Toy Project

A simple, educational project for learning Large Language Models (LLMs) with PyTorch and Transformers.

## 🎯 Purpose

This project is designed for beginners who want to:
- Understand how LLMs work
- Experiment with pre-trained models
- Learn about text generation
- Practice fine-tuning techniques
- Build a foundation for more advanced LLM projects

## 🏗️ Project Structure

```
llm_toy/
├── src/                    # Source code
│   ├── __init__.py
│   ├── model.py           # Model implementations
│   ├── trainer.py         # Training utilities
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks for learning
│   ├── 01_pytorch_setup.ipynb        # GPU verification
│   ├── 02_simple_llm_demo.ipynb      # Basic LLM usage
│   ├── 03_training_demo.ipynb        # Training on a tiny dataset (CN tutorial)
│   ├── 04_fine_tuning_demo.ipynb     # Small-scale fine-tuning (CN tutorial)
│   ├── 05_attention_visualization.ipynb  # Attention visualization (CN tutorial)
│   ├── 06_tokenization_basics.ipynb  # Train a BPE tokenizer (CN tutorial)
│   ├── 07_rag_intro.ipynb            # RAG with TF-IDF retriever (CN)
│   └── 08_evaluation_metrics.ipynb   # Perplexity/BLEU/ROUGE basics (CN)
├── configs/               # Configuration files
│   └── training_config.json
├── data/                  # Data storage
└── notebooks/            # Jupyter notebooks

```

## 🚀 Quick Start

### 1. Verify Your Setup

Run the first notebook to check if PyTorch can access your RTX 2070:

```bash
jupyter lab
# Open notebooks/01_pytorch_setup.ipynb
```

### 2. Try Text Generation

Open the second notebook to see LLM in action:

```bash
# Open notebooks/02_simple_llm_demo.ipynb
```

### 3. Train Your First Model

Follow the notebooks in order to build your understanding:
1. `01_pytorch_setup.ipynb` - Verify GPU setup
2. `02_simple_llm_demo.ipynb` - Basic text generation
3. `03_training_demo.ipynb` - Train from scratch
4. `04_fine_tuning_demo.ipynb` - Fine-tune existing models
5. `05_attention_visualization.ipynb` - Understand attention

## 💻 Requirements

- Python 3.12+
- CUDA-compatible GPU (RTX 2070 recommended)
- 8GB+ GPU memory
- 16GB+ system RAM

## 📦 Key Dependencies

- PyTorch with CUDA support
- Transformers (Hugging Face)
- Jupyter Lab
- Weights & Biases (optional, for experiment tracking)
- Matplotlib & Seaborn (for visualization)

## 🎓 Learning Path

### Beginner Level
1. **Setup Verification** - Ensure GPU works with PyTorch
2. **Basic Generation** - Use pre-trained models for text generation
3. **Parameter Tuning** - Experiment with temperature, top-k, top-p sampling

### Intermediate Level
4. **Custom Training** - Train small models from scratch
5. **Fine-tuning** - Adapt pre-trained models to your data
6. **Evaluation** - Learn to measure model performance

### Advanced Level
7. **Attention Analysis** - Visualize what models learn
8. **Custom Architectures** - Build your own transformer variants
9. **Optimization** - Speed up training and inference

## 🔧 Configuration

Edit `configs/training_config.json` to customize:
- Model parameters
- Training settings
- Generation parameters
- Logging preferences

## 📊 Monitoring

The project supports Weights & Biases for experiment tracking:
1. Sign up at [wandb.ai](https://wandb.ai)
2. Set your API key: `wandb login`
3. Training metrics will be automatically logged

## 🤝 Contributing

This is a learning project. Feel free to:
- Add new notebooks
- Improve existing code
- Share your experiments
- Ask questions

## 📚 Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 🆘 Troubleshooting

### GPU Memory Issues
- Reduce batch size in config
- Use gradient accumulation
- Enable mixed precision training

### CUDA Errors
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA version matches system CUDA
- Restart kernel if needed

### Model Loading Issues
- Check internet connection for downloading models
- Verify Hugging Face Hub access
- Try different model names

## 📖 License

This project is for educational purposes. Feel free to use and modify as needed.

---

**Happy Learning! 🎉**

Start with `notebooks/01_pytorch_setup.ipynb` and work your way through the notebooks to build your LLM knowledge step by step.

---

For a fully Chinese learning path and hands-on notes, see: `llm_toy/README_CN.md`.
