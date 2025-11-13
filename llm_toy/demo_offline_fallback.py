#!/usr/bin/env python3
"""
Demonstration script showing offline model fallback behavior
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demonstrate_offline_fallback():
    """Demonstrate the offline fallback functionality"""
    print("🎯 Offline Model Fallback Demonstration")
    print("=" * 60)
    
    # Method 1: Try to load real model with fallback
    print("\n1️⃣  Attempting to load real GPT-2 model with offline fallback...")
    try:
        from offline_model import create_model
        
        # This will try real model first, fallback to offline if needed
        model = create_model("gpt2")
        
        # Check what type of model we got
        model_type = type(model).__name__
        print(f"   ✅ Model loaded: {model_type}")
        
        # Get model info
        info = model.get_model_info()
        mode = info.get('mode', 'REAL')
        print(f"   📊 Mode: {mode}")
        
        if mode == 'OFFLINE_DEMO':
            print("   ⚠️  Running in offline demo mode (real model unavailable)")
        else:
            print("   🌐 Using real GPT-2 model from Hugging Face")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   🔄 This would trigger offline fallback in normal operation")
    
    # Method 2: Force offline mode
    print("\n2️⃣  Forcing offline mode for comparison...")
    try:
        offline_model = create_model("gpt2", force_offline=True)
        print(f"   ✅ Forced offline model: {type(offline_model).__name__}")
        
        # Test generation
        prompt = "The future of AI is"
        print(f"   📝 Testing with prompt: '{prompt}'")
        result = offline_model.generate_text(prompt, max_length=30, temperature=0.7)
        print(f"   🎯 Generated: {result}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 3: Direct offline model creation
    print("\n3️⃣  Direct offline model creation...")
    try:
        from offline_model import OfflineGPTModel
        
        direct_model = OfflineGPTModel("gpt2")
        print(f"   ✅ Direct offline model: {type(direct_model).__name__}")
        
        # Quick test
        test_prompt = "Machine learning helps us"
        response = direct_model.generate_text(test_prompt, max_length=20)
        print(f"   📝 Response: {response}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def show_usage_examples():
    """Show practical usage examples"""
    print("\n\n📚 Practical Usage Examples")
    print("=" * 60)
    
    print("\n✅ Recommended approach for notebooks:")
    print("```python")
    print("from offline_model import create_model")
    print("model = create_model('gpt2')  # Automatic fallback")
    print("```")
    
    print("\n✅ Force offline for consistent demo:")
    print("```python")
    print("model = create_model('gpt2', force_offline=True)")
    print("```")
    
    print("\n✅ Same API as original model:")
    print("```python")
    print("# Generate text")
    print("text = model.generate_text('AI is', max_length=50, temperature=0.7)")
    print("")
    print("# Get model info")
    print("info = model.get_model_info()")
    print("```")

def show_key_benefits():
    """Show key benefits of the offline model"""
    print("\n\n🌟 Key Benefits")
    print("=" * 60)
    
    benefits = [
        "🌐  Works without internet connectivity",
        "🔄  Automatic fallback from real to demo model",
        "📚  Perfect for educational environments", 
        "⚡  No download delays or network dependencies",
        "🔧  Same API as real GPT-2 model",
        "🎯  Contextually aware responses",
        "📊  Parameter-sensitive generation",
        "⚠️  Clear warnings about demo mode"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")

def main():
    """Main demonstration"""
    print("🚀 Offline-Friendly GPT Model Demo")
    print("=" * 80)
    print("This demo shows how the offline model provides fallback functionality")
    print("when the real GPT-2 model cannot be loaded due to network issues.")
    print("=" * 80)
    
    try:
        demonstrate_offline_fallback()
        show_usage_examples()
        show_key_benefits()
        
        print("\n\n✅ Demo completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Try the offline model in your notebooks")
        print("   2. Compare online vs offline responses")
        print("   3. Experiment with different parameters")
        print("   4. Use in classroom environments without internet")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())