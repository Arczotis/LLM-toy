#!/usr/bin/env python3
"""
Test script for the offline GPT model
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from offline_model import create_model, OfflineGPTModel, create_offline_model


def test_offline_model():
    """Test the offline model functionality"""
    print("🧪 Testing OfflineGPTModel...")
    print("=" * 60)
    
    # Test 1: Direct creation of offline model
    print("\n1. Testing direct offline model creation:")
    offline_model = OfflineGPTModel("gpt2")
    
    # Test 2: Get model info
    print("\n2. Testing get_model_info():")
    model_info = offline_model.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Test 3: Text generation
    print("\n3. Testing text generation:")
    test_prompts = [
        "The future of artificial intelligence is",
        "Machine learning can help us",
        "In the world of technology,",
        "The most important programming language for AI is"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n   Test {i}: {prompt}")
        generated = offline_model.generate_text(prompt, max_length=30, temperature=0.7)
        print(f"   Generated: {generated}")
        print(f"   Length: {len(generated.split())} words")
    
    print("\n" + "=" * 60)


def test_factory_function():
    """Test the factory function"""
    print("\n🔧 Testing factory function...")
    print("=" * 60)
    
    # Test force offline mode
    print("\n1. Testing force_offline=True:")
    model = create_model(force_offline=True)
    print(f"   Model type: {type(model).__name__}")
    
    # Test with actual model loading attempt (will fail if offline)
    print("\n2. Testing automatic fallback (trying real model):")
    try:
        model = create_model()
        print(f"   Model type: {type(model).__name__}")
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            print(f"   Mode: {info.get('mode', 'REAL')}")
    except Exception as e:
        print(f"   Error (expected if offline): {e}")
    
    print("\n" + "=" * 60)


def test_parameter_variations():
    """Test different generation parameters"""
    print("\n🎛️  Testing parameter variations...")
    print("=" * 60)
    
    model = create_offline_model()
    prompt = "Deep learning is"
    
    # Test different temperatures
    temperatures = [0.3, 0.7, 1.0, 1.5]
    print(f"\nTesting different temperatures with prompt: '{prompt}'")
    
    for temp in temperatures:
        print(f"\n   Temperature: {temp}")
        result = model.generate_text(prompt, max_length=25, temperature=temp)
        print(f"   Result: {result}")
    
    # Test different max lengths
    print(f"\nTesting different max lengths with prompt: '{prompt}'")
    max_lengths = [10, 25, 50]
    
    for max_len in max_lengths:
        print(f"\n   Max length: {max_len}")
        result = model.generate_text(prompt, max_length=max_len, temperature=0.7)
        actual_length = len(result.split())
        print(f"   Result: {result}")
        print(f"   Actual length: {actual_length} words")
    
    print("\n" + "=" * 60)


def main():
    """Run all tests"""
    print("🚀 Starting Offline Model Tests")
    print("=" * 80)
    
    try:
        test_offline_model()
        test_factory_function()
        test_parameter_variations()
        
        print("\n✅ All tests completed successfully!")
        print("\nThe offline model is ready for use in educational notebooks.")
        print("Notebooks will automatically fall back to offline mode when:")
        print("  - No internet connection is available")
        print("  - Hugging Face models cannot be downloaded")
        print("  - Explicitly forced with force_offline=True")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())