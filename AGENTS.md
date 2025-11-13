# JupyterProject

## Project Overview

This is a Python-based JupyterLab project configured for data science and Large Language Model (LLM) learning workflows. The project uses modern Python packaging tools and is set up with PyCharm IDE integration. It serves dual purposes: general data science work and educational LLM experimentation.

## Technology Stack

- **Python Version**: 3.12+ (required)
- **Package Manager**: uv (modern Python package manager)
- **IDE**: PyCharm with dedicated project configuration
- **Primary Environment**: JupyterLab for interactive development
- **GPU Support**: CUDA-compatible GPUs (RTX 2070 recommended)

## Key Dependencies

Core dependencies defined in `pyproject.toml`:
- **jupyterlab** (>=4.4.10): Interactive development environment
- **matplotlib** (>=3.10.7): Data visualization library
- **pandas** (>=2.3.3): Data manipulation and analysis
- **torch** (>=2.1.0): PyTorch deep learning framework
- **torchvision** (>=0.16.0): Computer vision utilities
- **transformers** (>=4.35.0): Hugging Face pre-trained models
- **accelerate** (>=0.24.0): Training optimization
- **datasets** (>=2.14.0): Dataset handling
- **tokenizers** (>=0.15.0): Text tokenization
- **scikit-learn** (>=1.3.0): Machine learning utilities
- **tqdm** (>=4.64.0): Progress bars
- **wandb** (>=0.16.0): Experiment tracking

## Project Structure

```
JupyterProject/
├── .idea/                    # PyCharm IDE configuration
├── .venv/                    # Virtual environment (uv-managed)
├── data/                     # Data storage directory (empty)
├── models/                   # Model storage directory (empty)
├── llm_toy/                  # Educational LLM learning project
│   ├── src/                  # Source code
│   │   ├── __init__.py
│   │   ├── model.py         # Model implementations (SimpleGPTModel, SimpleTransformer)
│   │   ├── trainer.py       # Training utilities (LLMTrainer, SimpleDataset)
│   │   └── utils.py         # Utility functions (GPU memory, plotting, sampling)
│   ├── notebooks/           # Jupyter notebooks for learning
│   │   ├── 00_setup_troubleshooting.ipynb  # GPU setup issues
│   │   ├── 01_pytorch_setup.ipynb          # GPU verification
│   │   ├── 02_simple_llm_demo.ipynb        # Basic LLM usage
│   │   └── 99_post_gpu_fix.ipynb           # Post-fix verification
│   ├── configs/             # Configuration files
│   │   └── training_config.json
│   ├── main.py              # Main test script
│   └── README.md            # LLM project documentation
├── pyproject.toml            # Project configuration and dependencies
├── requirements.txt          # Empty requirements file
├── uv.lock                   # Lock file for reproducible builds
├── sample.ipynb              # Sample Jupyter notebook
├── test_gpu.py              # GPU driver test program (Chinese)
├── gpu_fix_guide.md         # GPU driver fix guide (Chinese)
├── fix_gpu_quick.sh         # Quick GPU fix script
└── README.md                 # Project documentation
```

## Build and Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running JupyterLab
```bash
# Start JupyterLab server
jupyter lab

# Or with specific configuration
jupyter lab --no-browser --ip=0.0.0.0
```

### LLM Toy Project Commands
```bash
# Test basic setup
cd llm_toy
python main.py --test setup

# Test text generation
python main.py --test generation

# Test training setup
python main.py --test training

# Run all tests
python main.py --test all
```

### GPU Testing
```bash
# Test GPU drivers and CUDA
python test_gpu.py

# Quick GPU fix
./fix_gpu_quick.sh
```

## Code Organization

### Main Project (Data Science)
- **data/**: Intended for storing datasets and input files
- **models/**: Intended for storing trained models and model artifacts
- **sample.ipynb**: Demonstration notebook showing basic Jupyter functionality

### LLM Toy Project (Educational)
- **src/model.py**: Contains `SimpleGPTModel` wrapper for pre-trained models and `SimpleTransformer` for custom implementations
- **src/trainer.py**: `LLMTrainer` class for training loops, `SimpleDataset` for data handling
- **src/utils.py**: Comprehensive utilities for GPU management, text processing, visualization, and sampling strategies
- **notebooks/**: Progressive learning sequence from setup troubleshooting to advanced LLM concepts
- **configs/training_config.json**: Configuration for model parameters, training settings, and generation parameters

## Development Environment

The project is configured for PyCharm IDE with:
- Python SDK: `uv (JupyterProject)`
- Excluded directories: `.venv` and `models` (to avoid indexing)
- Code formatting: Black integration configured

## Learning Path

### Beginner Level
1. **Setup Verification** - Ensure GPU works with PyTorch (01_pytorch_setup.ipynb)
2. **Basic Generation** - Use pre-trained models for text generation (02_simple_llm_demo.ipynb)
3. **Parameter Tuning** - Experiment with temperature, top-k, top-p sampling

### Intermediate Level
4. **Custom Training** - Train small models from scratch
5. **Fine-tuning** - Adapt pre-trained models to your data
6. **Evaluation** - Learn to measure model performance

### Advanced Level
7. **Attention Analysis** - Visualize what models learn
8. **Custom Architectures** - Build your own transformer variants
9. **Optimization** - Speed up training and inference

## Testing Strategy

- **GPU Setup Tests**: Comprehensive GPU driver and CUDA testing in `test_gpu.py`
- **LLM Project Tests**: Modular testing via `main.py` with setup, generation, and training tests
- **Notebook Progression**: Step-by-step verification through numbered notebooks
- **Configuration Validation**: JSON-based configuration with validation

## GPU Requirements and Troubleshooting

### Hardware Requirements
- **CUDA-compatible GPU**: RTX 2070 recommended
- **GPU Memory**: 8GB+ recommended
- **System RAM**: 16GB+ recommended

### GPU Driver Support
The project includes comprehensive GPU troubleshooting resources:
- **Chinese language guides**: `gpu_fix_guide.md` and `test_gpu.py`
- **RTX 2070 specific fixes**: Driver version recommendations and installation procedures
- **Quick fix script**: `fix_gpu_quick.sh` for automated driver fixes
- **Version compatibility**: Detailed guidance on NVIDIA driver versions (535 series recommended)

### Common Issues Addressed
- NVIDIA driver version mismatches
- CUDA/PyTorch compatibility issues
- Kernel module conflicts
- Memory management problems

## Security Considerations

- Virtual environment is isolated in `.venv/`
- No sensitive configuration files are present
- JupyterLab should be configured with appropriate authentication for production use
- GPU driver fixes require sudo access and system-level changes

## Monitoring and Experiment Tracking

- **Weights & Biases Integration**: Automatic experiment logging via `wandb`
- **GPU Memory Monitoring**: Real-time GPU memory usage tracking
- **Training History**: JSON-based training history with plotting capabilities
- **Attention Visualization**: Tools for visualizing transformer attention weights

## Special Features

### Educational Focus
- Progressive notebook-based learning path
- Comprehensive error handling and troubleshooting
- Both pre-trained model usage and custom training examples
- Real-world GPU setup and maintenance skills

### Multilingual Support
- Primary documentation in English
- GPU troubleshooting guides in Chinese
- Code comments and error messages in English

### Production Considerations
- Modular, reusable code architecture
- Configuration-driven approach
- Comprehensive logging and monitoring
- Model checkpointing and recovery

## Notes

- This project combines general data science capabilities with specialized LLM education
- The `llm_toy` subdirectory is a complete educational environment for learning modern NLP
- GPU setup is extensively documented due to common driver issues with RTX series cards
- The project uses modern Python tooling (uv) instead of traditional pip/requirements.txt approach
- CPU-only operation is fully supported for learning purposes, though GPU acceleration is recommended for training