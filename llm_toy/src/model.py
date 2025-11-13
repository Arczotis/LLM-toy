"""
Simple Transformer model for LLM learning
"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class SimpleGPTModel:
    """A wrapper class for GPT-2 model for learning purposes"""
    
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, do_sample=True):
        """Generate text based on input prompt"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def get_model_info(self):
        """Get model information"""
        return {
            "model_name": self.model.config.model_type,
            "vocab_size": self.model.config.vocab_size,
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_hidden_layers,
            "num_heads": self.model.config.num_attention_heads,
            "device": str(self.device)
        }


class SimpleTransformer(nn.Module):
    """A very simple transformer implementation for learning"""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        
    def forward(self, src):
        # Embedding and positional encoding
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src)
        
        # Decode to vocabulary
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]