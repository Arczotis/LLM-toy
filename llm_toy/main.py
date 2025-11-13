#!/usr/bin/env python3
"""
Main script for the LLM Toy Project
Run this to quickly test your setup and start learning!
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from model import SimpleGPTModel
from utils import set_seed, get_device, print_gpu_memory
from trainer import create_sample_data
from trainer import LLMTrainer, SimpleDataset, init_wandb
try:
    from online_model import create_online_model
except Exception:
    create_online_model = None  # type: ignore


def test_setup():
    """Test basic setup and GPU availability"""
    print("🚀 Testing LLM Toy Project Setup")
    print("=" * 50)
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Print GPU memory
    print_gpu_memory()
    
    print("✅ Basic setup test passed!")
    return device


def test_text_generation():
    """Test text generation with GPT-2"""
    print("\n📝 Testing Text Generation")
    print("=" * 50)
    
    try:
        # Initialize model
        print("Loading GPT-2 model...")
        model = SimpleGPTModel(model_name="gpt2")
        
        # Test prompts
        test_prompts = [
            "The future of AI is",
            "Machine learning can help us",
            "In the world of technology,"
        ]
        
        print("\nGenerating text examples:")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Prompt: '{prompt}'")
            generated = model.generate_text(
                prompt,
                max_length=50,
                temperature=0.7,
                do_sample=True
            )
            print(f"   Generated: '{generated}'")
        
        print("\n✅ Text generation test passed!")
        
    except Exception as e:
        print(f"❌ Text generation test failed: {e}")
        return False
    
    return True


def test_online_generation(provider: str | None = None, model_id: str | None = None) -> bool:
    """Test text generation via online provider (OpenRouter/SiliconFlow)."""
    print("\n🌐 Testing Online Text Generation")
    print("=" * 50)

    if create_online_model is None:
        print("❌ Online model client not available")
        return False

    try:
        print("Initializing online model client...")
        online = create_online_model(model=model_id, provider=provider)

        print("\nProvider/Model:")
        print(online.get_model_info())

        prompt = "用一句话描述一下这个项目的用途。"
        print(f"\n📝 Prompt: {prompt}")
        out = online.generate_text(prompt, max_length=80, temperature=0.7)
        print(f"   Generated: {out}")

        print("\n✅ Online generation test passed!")
        return True
    except Exception as e:
        print(f"❌ Online generation test failed: {e}")
        print("提示: 请在 llm_toy/configs/llm_api_config.json 中填写 API Key，或设置环境变量 OPENROUTER_API_KEY / SILICONFLOW_API_KEY。")
        return False


def test_training_setup():
    """Test training setup with sample data"""
    print("\n🏋️ Testing Training Setup")
    print("=" * 50)
    
    try:
        # Create sample data
        print("Creating sample data...")
        sample_texts = create_sample_data()
        print(f"Created {len(sample_texts)} sample texts")
        
        # Initialize model and tokenizer
        model = SimpleGPTModel(model_name="gpt2")
        
        # Create dataset
        dataset = SimpleDataset(
            texts=sample_texts,
            tokenizer=model.tokenizer,
            max_length=64
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        print(f"Sample data shape: {sample['input_ids'].shape}")
        
        print("\n✅ Training setup test passed!")
        
    except Exception as e:
        print(f"❌ Training setup test failed: {e}")
        return False
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLM Toy Project Main Script")
    parser.add_argument("--test", choices=["setup", "generation", "training", "online", "all"], 
                       default="all", help="What to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--provider", type=str, default=None, help="Online provider: openrouter or siliconflow")
    parser.add_argument("--model-id", type=str, default=None, help="Online provider model id")
    
    args = parser.parse_args()
    
    print("🎯 LLM Toy Project - Main Script")
    print("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    if args.test in ["setup", "all"]:
        total_tests += 1
        try:
            device = test_setup()
            tests_passed += 1
        except Exception as e:
            print(f"❌ Setup test failed: {e}")
    
    if args.test in ["generation", "all"]:
        total_tests += 1
        if test_text_generation():
            tests_passed += 1
    
    if args.test in ["training", "all"]:
        total_tests += 1
        if test_training_setup():
            tests_passed += 1

    if args.test in ["online", "all"]:
        total_tests += 1
        if test_online_generation(provider=args.provider, model_id=args.model_id):
            tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your LLM Toy Project is ready to use!")
        print("\nNext steps:")
        print("1. Run: jupyter lab")
        print("2. Open notebooks/01_pytorch_setup.ipynb")
        print("3. Work through the notebooks in order")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("- Check GPU drivers and CUDA installation")
        print("- Verify internet connection for model downloads")
        print("- Check available GPU memory")
    
    print("\nFor help, check the README.md file or open an issue.")


if __name__ == "__main__":
    main()
