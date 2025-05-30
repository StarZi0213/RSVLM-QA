# VL Models Inference Tools

This repository contains various tools for visual language model (VL) inference tasks. Each script is designed to process visual question answering (VQA) tasks using different large multimodal models.

## Available Models

- **Qwen2.5-VL**: Implementation of batch VQA inference using Qwen2.5-VL-3B-Instruct model
- **Gemma 3**: Implementation using Gemma-3-4b-it model for visual question answering
- **BLIP2**: Implementation using BLIP2 models for visual question answering
- **InternVL3**: Implementation using InternVL3 for visual question answering
- **LLaVA**: Implementation using LLaVA models for visual question answering
- **OVIS2**: Implementation using OVIS2 model for visual question answering

## Features

- Batch processing of JSONL-formatted visual question answering data
- Automatic hardware acceleration selection (CUDA, MPS, CPU)
- Memory optimization for processing large volumes of images
- Checkpoint resumption capability for interrupted processing
- Progress bars and detailed logging
- Error handling and automatic retry mechanisms

## Installation

```bash
# Install the latest version of transformers
pip install git+https://github.com/huggingface/transformers accelerate

# Install qwen-vl-utils (with decord feature for faster video processing)
pip install qwen-vl-utils[decord]==0.0.8

# Install other dependencies
pip install torch tqdm pillow requests
```

## Data Format

Input data should be in JSONL format, with each line containing a JSON object structured as follows:

```json
{
  "id": "image_001",
  "image": "/path/to/image.jpg",
  "vqa_pairs": [
    {"question": "What is in this image?", "answer": "A cat"},
    {"question": "What color is the animal in the image?", "answer": "Orange"}
  ]
}
```

## Configuration

Before running any script, modify the following configuration items in the respective Python file:

```python
INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl"  # Input file path
OUTPUT_JSONL_FILE = "model_VQA_test_results.jsonl"  # Output file path

# Image min/max pixel settings (for performance)
MIN_PIXELS = 256 * 28 * 28  # Minimum pixel count
MAX_PIXELS = 1280 * 28 * 28  # Maximum pixel count
```

## Running the Scripts

```bash
python <model_name>_inference.py
```

For example:
```bash
python qwen_vl_inference.py
```

## Output Format

The output is also in JSONL format, with each line containing a JSON object that extends the original data with model-specific answer fields:

```json
{
  "id": "image_001",
  "image": "/path/to/image.jpg",
  "vqa_pairs": [...],
  "vqa_pairs_with_model_name": [
    {
      "question": "What is in this image?",
      "answer": "A cat",
      "model_name_answer": "The image shows an orange cat sitting on a windowsill."
    },
    ...
  ]
}
```

## Advanced Configuration

### Device and Precision Settings

Each script automatically selects available hardware acceleration:

- NVIDIA GPU (CUDA) - Uses float16/bfloat16 precision
- Apple Silicon (MPS) - Uses float16/float32 precision
- CPU - Uses float32 precision

For manual configuration, modify the `DEVICE` and `TORCH_DTYPE` variables in the script.

### Memory Optimization

For large batch processing tasks, optimize memory usage by:

1. Adjusting `MIN_PIXELS` and `MAX_PIXELS` to control image processing resolution
2. Modifying the `max_new_tokens` parameter in the answer generation function to limit text generation length
3. Enabling `flash_attention_2` when using CUDA to improve performance and reduce memory usage

### Checkpoint Resumption

If processing is interrupted, the script creates a checkpoint file. When restarted, it automatically continues from the interruption point.

## Troubleshooting

- **Out of memory errors**: Try reducing batch size or adjusting `MIN_PIXELS` and `MAX_PIXELS`
- **Model loading errors**: Ensure you have the latest version of transformers library
- **KeyError**: May need to install the latest transformers library from source
- **Image loading errors**: Check if image paths are correct

## Notes

- On first run, the script will download the model (typically 2-7GB)
- Processing large volumes of images requires significant time and computational resources
- High-resolution images may require more GPU memory 