"""
Utility functions for the LLM toy project
"""
import torch
import numpy as np
import random
import json
import os
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available")


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config: Dict, save_path: str):
    """Save configuration to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {save_path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {config_path}")
    return config


def plot_training_history(history: List[Dict], save_path: Optional[str] = None):
    """Plot training history"""
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    
    if any("val_loss" in entry for entry in history):
        val_losses = [entry.get("val_loss", None) for entry in history]
        plt.plot(epochs, val_losses, label="Validation Loss", marker='s')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_attention_weights(attention_weights: torch.Tensor, tokens: List[str], 
                          save_path: Optional[str] = None):
    """Plot attention weights heatmap"""
    if attention_weights.dim() > 2:
        # Take the first head and first layer for visualization
        attention_weights = attention_weights[0, 0].cpu().detach().numpy()
    else:
        attention_weights = attention_weights.cpu().detach().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        annot=True,
        fmt=".2f"
    )
    plt.title("Attention Weights")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention weights plot saved to {save_path}")
    
    plt.show()


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from loss"""
    return torch.exp(torch.tensor(loss)).item()


def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> int:
    """Top-k sampling for text generation"""
    # Apply temperature
    logits = logits / temperature
    
    # Get top k logits
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # Convert to probabilities
    probabilities = torch.softmax(top_k_logits, dim=-1)
    
    # Sample from top k
    next_token = torch.multinomial(probabilities, num_samples=1)
    
    return top_k_indices[next_token].item()


def top_p_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> int:
    """Top-p (nucleus) sampling for text generation"""
    # Apply temperature
    logits = logits / temperature
    
    # Convert to probabilities
    probabilities = torch.softmax(logits, dim=-1)
    
    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find tokens with cumulative probability > p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    
    # Remove low probability tokens
    sorted_probs[sorted_indices_to_remove] = 0.0
    
    # Renormalize
    sorted_probs /= sorted_probs.sum()
    
    # Sample
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    
    return sorted_indices[next_token].item()


def beam_search(logits: torch.Tensor, beam_size: int = 5, temperature: float = 1.0) -> List[int]:
    """Simple beam search implementation"""
    # This is a simplified version - real beam search is more complex
    logits = logits / temperature
    probabilities = torch.softmax(logits, dim=-1)
    
    # Get top beam_size tokens
    top_probs, top_indices = torch.topk(probabilities, beam_size)
    
    return top_indices.tolist()


def analyze_text_statistics(texts: List[str]) -> Dict:
    """Analyze text statistics"""
    stats = {
        "num_texts": len(texts),
        "total_characters": sum(len(text) for text in texts),
        "avg_length": np.mean([len(text) for text in texts]),
        "min_length": min(len(text) for text in texts),
        "max_length": max(len(text) for text in texts),
        "unique_words": set(),
        "word_frequency": defaultdict(int)
    }
    
    for text in texts:
        words = text.lower().split()
        stats["unique_words"].update(words)
        for word in words:
            stats["word_frequency"][word] += 1
    
    stats["num_unique_words"] = len(stats["unique_words"])
    stats["avg_words_per_text"] = np.mean([len(text.split()) for text in texts])
    
    return stats


def create_vocabulary(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    """Create vocabulary from texts"""
    word_freq = defaultdict(int)
    
    for text in texts:
        words = text.lower().split()
        for word in words:
            word_freq[word] += 1
    
    # Filter by minimum frequency
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def texts_to_sequences(texts: List[str], vocab: Dict[str, int]) -> List[List[int]]:
    """Convert texts to sequences of integers"""
    sequences = []
    
    for text in texts:
        words = text.lower().split()
        sequence = [vocab.get(word, vocab["<UNK>"]) for word in words]
        sequences.append(sequence)
    
    return sequences


def pad_sequences(sequences: List[List[int]], max_length: int, padding_value: int = 0) -> torch.Tensor:
    """Pad sequences to the same length"""
    padded = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = torch.tensor(seq[:length])
    
    return padded


def split_data(texts: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple:
    """Split data into train, validation, and test sets"""
    np.random.shuffle(texts)
    
    n = len(texts)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_texts = texts[:train_size]
    val_texts = texts[train_size:train_size + val_size]
    test_texts = texts[train_size + val_size:]
    
    return train_texts, val_texts, test_texts


def print_model_summary(model, input_size=None):
    """Print model summary"""
    print("\nModel Summary:")
    print("=" * 50)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    if input_size:
        # Calculate model size in memory
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size = (param_size + buffer_size) / 1024**2
        print(f"Model size in memory: {model_size:.2f} MB")
    
    print("\nModel Architecture:")
    print(model)
    print("=" * 50)