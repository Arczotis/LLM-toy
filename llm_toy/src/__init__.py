"""
LLM Toy Package - Educational tools for learning about Large Language Models
"""

# Import the main model classes
from .model import SimpleGPTModel, SimpleTransformer, PositionalEncoding

# Import offline-friendly model and factory function
try:
    from .offline_model import OfflineGPTModel, create_model, create_offline_model
    OFFLINE_AVAILABLE = True
except ImportError:
    OFFLINE_AVAILABLE = False

# Import utility functions
try:
    from .utils import set_seed, get_device, print_gpu_memory
except ImportError:
    # Fallback if utils are not available
    def set_seed(seed):
        import torch
        torch.manual_seed(seed)
    
    def get_device():
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def print_gpu_memory():
        import torch
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            print("No CUDA GPU available")

# Package version
__version__ = "0.1.0"

# Make key components available at package level
__all__ = [
    "SimpleGPTModel",
    "SimpleTransformer", 
    "PositionalEncoding",
    "set_seed",
    "get_device", 
    "print_gpu_memory",
    "__version__"
]

# Add offline components if available
if OFFLINE_AVAILABLE:
    __all__.extend([
        "OfflineGPTModel",
        "create_model",
        "create_offline_model"
    ])

def get_model(prefer_offline=False, force_offline=False):
    """
    Convenience function to get a model with offline fallback support.
    
    Args:
        prefer_offline: If True, prefer offline model when available
        force_offline: If True, always use offline model
    
    Returns:
        A GPT model instance (real or offline demo)
    """
    if OFFLINE_AVAILABLE:
        return create_model(force_offline=force_offline)
    else:
        # Fallback to regular model if offline not available
        return SimpleGPTModel()