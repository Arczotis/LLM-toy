"""
Offline-friendly GPT Model for educational purposes
"""
import torch
import torch.nn as nn
import warnings
import random
from typing import Dict, Any, Optional


class OfflineGPTModel:
    """
    Offline/demo version of GPT model for educational purposes.
    Provides the same interface as SimpleGPTModel but returns dummy responses
    when real model cannot be loaded due to network issues.
    """
    
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        warnings.warn(
            "⚠️  Running in OFFLINE/DEMO MODE - This is a placeholder model for educational purposes!\n"
            "   The responses are simulated and not from a real GPT model.",
            UserWarning,
            stacklevel=2
        )
        
        # Store model name for info purposes
        self.model_name = model_name
        
        # Create a simple dummy tokenizer for basic operations
        self.vocab_size = 50257  # GPT-2 vocab size
        self.hidden_size = 768   # GPT-2 base model hidden size
        self.num_layers = 12     # GPT-2 base model layers
        self.num_heads = 12      # GPT-2 base model attention heads
        
        # Simple vocabulary for demo purposes
        self.dummy_vocab = [
            "the", "and", "of", "to", "in", "a", "is", "that", "it", "for", "on", "with", "as", "was", "were",
            "by", "an", "be", "this", "have", "from", "or", "one", "had", "but", "word", "not", "what",
            "all", "were", "we", "when", "your", "can", "said", "there", "each", "which", "she", "do",
            "how", "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these",
            "so", "some", "her", "would", "make", "like", "into", "him", "has", "two", "more", "very",
            "after", "words", "long", "than", "first", "been", "call", "who", "oil", "sit", "now", "find",
            "down", "day", "did", "get", "come", "made", "may", "part", "over", "new", "sound", "take",
            "only", "little", "work", "know", "place", "year", "live", "back", "give", "most", "good",
            "man", "think", "say", "great", "where", "help", "much", "too", "line", "right", "old",
            "tell", "boy", "follow", "came", "want", "show", "also", "around", "form", "three", "small",
            "set", "put", "end", "why", "again", "turn", "ask", "went", "men", "read", "need", "land",
            "different", "home", "us", "move", "try", "kind", "hand", "picture", "again", "change",
            "off", "play", "spell", "air", "away", "animal", "house", "point", "page", "letter", "mother",
            "answer", "found", "study", "still", "learn", "should", "america", "world", "high", "every",
            "near", "add", "food", "between", "own", "below", "country", "plant", "last", "school", "father",
            "keep", "tree", "never", "start", "city", "earth", "eye", "light", "thought", "head", "under",
            "story", "saw", "left", "dont", "few", "while", "along", "might", "close", "something", "seem",
            "next", "hard", "open", "example", "begin", "life", "always", "those", "both", "paper", "together",
            "got", "group", "often", "run", "important", "until", "children", "side", "feet", "car", "mile",
            "night", "walk", "white", "sea", "began", "grow", "took", "river", "four", "carry", "state",
            "once", "book", "hear", "stop", "without", "second", "later", "miss", "idea", "enough", "eat",
            "face", "watch", "far", "indian", "really", "almost", "let", "above", "girl", "sometimes",
            "mountain", "cut", "young", "talk", "soon", "list", "song", "being", "leave", "family", "body"
        ]
        
        # Add some AI/ML related words for more realistic responses
        ai_vocab = [
            "artificial", "intelligence", "machine", "learning", "neural", "network", "deep", "algorithm",
            "data", "model", "training", "prediction", "classification", "regression", "optimization",
            "gradient", "descent", "backpropagation", "layer", "activation", "function", "parameter",
            "weight", "bias", "epoch", "batch", "dataset", "feature", "label", "supervised", "unsupervised",
            "reinforcement", "computer", "vision", "natural", "language", "processing", "transformer",
            "attention", "mechanism", "embedding", "token", "sequence", "generation", "text", "speech",
            "recognition", "synthesis", "robotics", "automation", "analytics", "statistics", "probability"
        ]
        
        self.dummy_vocab.extend(ai_vocab)
        
        print(f"✅ Initialized OfflineGPTModel (Demo Mode)")
        print(f"   Vocabulary size: {len(self.dummy_vocab)} words")
        print(f"   Simulating: {model_name} architecture")
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7, do_sample: bool = True) -> str:
        """
        Generate dummy text based on input prompt.
        Simulates text generation with realistic-looking responses.
        """
        # Start with the original prompt
        result = prompt
        
        # Add some contextually relevant words based on the prompt
        prompt_lower = prompt.lower()
        
        # Determine continuation based on prompt content
        if any(word in prompt_lower for word in ["ai", "artificial intelligence", "machine learning"]):
            continuations = [
                " a rapidly evolving field that continues to transform industries",
                " expected to revolutionize how we interact with technology",
                " becoming increasingly sophisticated with each passing year",
                " helping solve complex problems across various domains",
                " enabling new possibilities in automation and decision making"
            ]
        elif any(word in prompt_lower for word in ["future", "tomorrow", "next"]):
            continuations = [
                " likely to bring significant changes and innovations",
                " full of possibilities and opportunities",
                " shaped by the decisions we make today",
                " uncertain but promising in many ways",
                " dependent on how we address current challenges"
            ]
        elif any(word in prompt_lower for word in ["technology", "tech", "digital"]):
            continuations = [
                " advancing at an unprecedented pace",
                " reshaping the way we live and work",
                " creating new opportunities for innovation",
                " becoming more integrated into our daily lives",
                " helping solve complex global challenges"
            ]
        else:
            continuations = [
                " an important consideration in modern contexts",
                " something that requires careful thought and analysis",
                " a topic of growing interest and relevance",
                " likely to have significant implications",
                " worth exploring in greater depth"
            ]
        
        # Add the main continuation
        result += random.choice(continuations)
        
        # Add some additional sentences to reach desired length
        current_length = len(result.split())
        
        while current_length < max_length:
            # Add connecting phrases
            connectors = [" Furthermore, ", " Additionally, ", " Moreover, ", " In addition, ", " Also, "]
            connector = random.choice(connectors) if current_length > 10 else ""
            
            # Generate a sentence with random words
            sentence_length = random.randint(8, 15)
            sentence_words = [random.choice(self.dummy_vocab) for _ in range(sentence_length)]
            sentence = " ".join(sentence_words).capitalize() + "."
            
            result += connector + sentence
            current_length = len(result.split())
            
            # Stop if we're getting too long
            if current_length > max_length + 20:
                break
        
        # Trim to approximate word count (split by spaces and rejoin)
        words = result.split()
        if len(words) > max_length:
            result = " ".join(words[:max_length])
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information (simulated GPT-2 info for demo purposes).
        """
        return {
            "model_name": "gpt2-offline-demo",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "device": str(self.device),
            "mode": "OFFLINE_DEMO",
            "warning": "This is a demonstration model - responses are simulated"
        }


def create_model(model_name: str = "gpt2", force_offline: bool = False) -> Optional['SimpleGPTModel']:
    """
    Factory function to create a GPT model.
    
    Args:
        model_name: Name of the model to load (default: "gpt2")
        force_offline: If True, always use offline mode (default: False)
    
    Returns:
        SimpleGPTModel instance or OfflineGPTModel if loading fails
    """
    
    if force_offline:
        print("🔧 Forcing offline mode as requested...")
        return OfflineGPTModel(model_name)
    
    # Try to import the real SimpleGPTModel
    try:
        from .model import SimpleGPTModel as RealSimpleGPTModel
        
        print(f"🌐 Attempting to load real GPT-2 model: {model_name}")
        
        # Try to load the real model
        try:
            model = RealSimpleGPTModel(model_name)
            print("✅ Successfully loaded real GPT-2 model!")
            return model
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for common network/download related errors
            if any(keyword in error_msg for keyword in [
                "connection", "timeout", "network", "internet", "download", 
                "offline", "huggingface", "transformers", "from_pretrained"
            ]):
                print(f"⚠️  Network error detected: {e}")
                print("🔄 Falling back to offline demo mode...")
                return OfflineGPTModel(model_name)
            else:
                # Re-raise if it's not a network-related error
                print(f"❌ Non-network error occurred: {e}")
                raise
                
    except ImportError:
        print("⚠️  Could not import SimpleGPTModel - using offline demo mode")
        return OfflineGPTModel(model_name)
    
    except Exception as e:
        print(f"⚠️  Unexpected error loading model: {e}")
        print("🔄 Falling back to offline demo mode...")
        return OfflineGPTModel(model_name)


# For backward compatibility and easy import
def create_offline_model(model_name: str = "gpt2") -> OfflineGPTModel:
    """
    Convenience function to create an offline model directly.
    
    Args:
        model_name: Name of the model to simulate (default: "gpt2")
    
    Returns:
        OfflineGPTModel instance
    """
    return OfflineGPTModel(model_name)