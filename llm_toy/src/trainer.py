"""
Training utilities for the LLM toy project
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import json
import os
from typing import Dict, List, Optional


class SimpleDataset(Dataset):
    """Simple dataset for text data"""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids  # For language modeling, labels are same as input_ids
        }


class LLMTrainer:
    """Trainer class for language models"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader, optimizer, scheduler=None, log_interval=10):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log to wandb
            if step % log_interval == 0 and wandb.run:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"]
                })
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs=3, 
              learning_rate=5e-5, warmup_steps=0, save_dir="./checkpoints"):
        """Full training loop"""
        
        # Create optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Create scheduler
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader, optimizer, scheduler)
            
            # Evaluate
            val_loss = None
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(os.path.join(save_dir, "best_model"))
            else:
                print(f"Train Loss: {train_loss:.4f}")
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt"))
            
            # Log to wandb
            if wandb.run:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                }
                if val_loss:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)
            
            # Update history
            history_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
            }
            if val_loss:
                history_entry["val_loss"] = val_loss
            training_history.append(history_entry)
        
        # Save training history
        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(training_history, f, indent=2)
        
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        return training_history
    
    def save_model(self, save_path):
        """Save model and tokenizer"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """Load model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        
        print(f"Model loaded from {model_path}")


def init_wandb(project_name="llm-toy", config=None):
    """Initialize wandb for experiment tracking"""
    if config is None:
        config = {}
    
    wandb.init(
        project=project_name,
        config=config
    )
    
    print(f"Wandb initialized for project: {project_name}")


def create_sample_data():
    """Create some sample text data for testing"""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Deep learning models require large amounts of data to train effectively.",
        "Natural language processing enables computers to understand human language.",
        "The transformer architecture revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Training large language models requires significant computational resources.",
        "Fine-tuning pre-trained models can be more efficient than training from scratch.",
        "Transfer learning has become a standard practice in natural language processing."
    ]
    
    return texts