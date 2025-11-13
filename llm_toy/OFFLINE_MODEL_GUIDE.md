# Offline Model Usage Guide

This guide explains how to use the new offline-friendly GPT model that automatically falls back to demo mode when internet connectivity is unavailable.

## Quick Start

### Option 1: Automatic Fallback (Recommended)

Replace your existing import:

```python
# Old way
from model import SimpleGPTModel
model = SimpleGPTModel("gpt2")

# New way - with automatic fallback
from offline_model import create_model
model = create_model("gpt2")  # Falls back to offline mode if needed
```

### Option 2: Force Offline Mode

```python
from offline_model import create_model
model = create_model("gpt2", force_offline=True)  # Always use offline mode
```

### Option 3: Direct Offline Model

```python
from offline_model import OfflineGPTModel
model = OfflineGPTModel("gpt2")  # Direct offline model creation
```

## Interface Compatibility

The offline model provides **100% interface compatibility** with the original `SimpleGPTModel`:

### Available Methods

- `generate_text(prompt, max_length=100, temperature=0.7, do_sample=True)`
- `get_model_info()`

### Method Signatures

All method signatures are identical to the original model, so existing code will work without changes.

## Behavior Differences

### Real GPT-2 Model
- Generates actual text using pre-trained GPT-2 weights
- Requires internet connection for initial download
- Produces coherent, contextually accurate responses

### Offline Demo Model
- Generates simulated responses for educational purposes
- Works completely offline
- Produces contextually relevant but simulated responses
- Shows clear warnings about demo mode

## Example Usage

### Basic Text Generation

```python
from offline_model import create_model

# Create model with automatic fallback
model = create_model("gpt2")

# Generate text (same API as original)
prompt = "The future of AI is"
generated_text = model.generate_text(prompt, max_length=50, temperature=0.7)
print(generated_text)
```

### Model Information

```python
# Get model info (same API as original)
info = model.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Mode: {info.get('mode', 'REAL')}")  # Will show OFFLINE_DEMO for offline mode
```

## Offline Model Features

### Contextual Awareness
The offline model analyzes your prompt and generates contextually appropriate responses:

- AI/ML prompts → Responses about technology and innovation
- Future-focused prompts → Forward-looking statements
- General prompts → Educational, relevant content

### Parameter Sensitivity
- **Temperature**: Controls randomness (0.1 = deterministic, 1.5 = creative)
- **Max Length**: Controls response length (approximately)
- **Do Sample**: Respected but doesn't change behavior significantly

### Educational Value
- Demonstrates LLM concepts without requiring internet
- Shows how parameters affect text generation
- Provides realistic-looking responses for learning purposes

## Migration Checklist

### For Existing Notebooks

1. **Replace Import**: Change `from model import SimpleGPTModel` to `from offline_model import create_model`
2. **Update Model Creation**: Change `SimpleGPTModel()` to `create_model()`
3. **Test**: Run the notebook to ensure compatibility
4. **Document**: Add a note about offline mode capability

### For New Notebooks

1. **Use create_model()**: Start with the factory function for automatic fallback
2. **Handle Warnings**: Users will see clear offline mode warnings
3. **Document**: Explain that offline mode provides educational simulations

## Error Handling

### Network Errors
The offline model automatically activates when:
- No internet connection is available
- Hugging Face servers are unreachable
- Model download fails
- Firewall blocks external requests

### Import Errors
If there are issues importing the real model, the system automatically falls back to offline mode.

## Best Practices

### For Educators
- Use offline mode for consistent classroom experiences
- Students can run code without internet dependency
- Demonstrate LLM concepts reliably

### For Students
- Understand that offline responses are simulated
- Focus on learning the API and parameters
- Experiment with temperature and max_length settings

### For Developers
- Test both online and offline modes
- Provide clear user feedback about current mode
- Handle fallback gracefully in applications

## Troubleshooting

### Offline Mode Won't Activate
- Check that you imported `create_model` from `offline_model`
- Ensure the offline_model.py file is in your src directory
- Try forcing offline mode with `force_offline=True`

### Responses Seem Random
- This is expected for offline demo mode
- The model generates educational simulations
- Real GPT-2 would provide more coherent responses

### Warnings About Demo Mode
- These warnings are intentional and informative
- They ensure users understand the simulation nature
- Warnings can be ignored for educational purposes

## Technical Details

### Offline Model Architecture
- Simulates GPT-2 architecture parameters
- Uses contextual word selection
- Implements parameter-sensitive generation
- Provides realistic response patterns

### Fallback Logic
1. Attempt to load real GPT-2 model
2. Catch network/download related errors
3. Activate offline demo model
4. Provide clear user messaging
5. Continue with educational simulation

---

**Remember**: The offline model is designed for educational purposes. It provides a realistic simulation of LLM behavior without requiring internet connectivity, making it perfect for classroom environments, offline development, and learning scenarios.