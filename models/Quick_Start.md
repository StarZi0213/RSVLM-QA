# Visual Language Models - Quick Start Guide

This guide provides a quick overview of how to use the VL model inference scripts for RSVL-VQA tasks.

## Installation

1. Create a Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. Install common dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install transformers accelerate
   pip install tqdm pillow requests
   ```

## Basic Usage

Each model follows the same basic usage pattern:

```bash
python <model_name>_inference.py --input <input_jsonl> --output <output_jsonl> --device <cuda|mps|cpu>
```

For example:
```bash
python gemma3_inference.py --input data/RSVL-VQA_test.jsonl --output results/gemma3_results.jsonl --device cuda
```

## Input Format

All models expect input in JSONL format with entries containing:
- `id`: Unique identifier for the image
- `image`: Path to the image file (absolute path recommended)
- `vqa_pairs`: Array of question-answer pairs

Example:
```json
{
  "id": "image_001",
  "image": "/path/to/image.jpg",
  "vqa_pairs": [
    {"question": "What is in this image?", "answer": "A cat"},
    {"question": "What color is the animal?", "answer": "Orange"}
  ]
}
```

## Available Models

### Gemma 3
```bash
python gemma3_inference.py

# Required: Hugging Face token for access to Gemma models
# Set this in the script or via environment variable
```

### LLaVA
```bash
python llava_inference.py

# Optional: Choose model variant by editing MODEL_ID in the script
# Default: "llava-hf/llava-1.5-7b-hf"
```

### BLIP2
```bash
python blip2_inference.py --model Salesforce/blip2-opt-6.7b

# Optional models:
# - Salesforce/blip2-opt-2.7b (faster)
# - Salesforce/blip2-opt-6.7b (default)
# - Salesforce/blip2-flan-t5-xl
```

### Qwen-VL
```bash
python qwen_vl_inference.py

# Additional dependency:
pip install qwen-vl-utils[decord]==0.0.8
```

### InternVL3
```bash
python internvl3_inference.py

# Default model: OpenGVLab/InternVL-Chat-V1-5
```

### OVIS2
```bash
python ovis2_inference.py

# Default model: BAAI/OVIS2-SFT-LLAMA3-8B
```

## Resuming Interrupted Runs

All scripts automatically track progress and can resume from interruptions by:
- Detecting previously processed images in the output file
- Using a checkpoint file created during processing

## Common Issues

1. **Out of memory errors**: Reduce image size by adjusting MIN_PIXELS/MAX_PIXELS in the script
2. **Missing libraries**: Check model-specific requirements
3. **Slow processing**: Consider using a GPU or adjusting batch settings

## Full Documentation

For more detailed information, see the [USAGE.md](USAGE.md) file. 