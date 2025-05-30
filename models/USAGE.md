# Visual Language Models Usage Guide

This guide provides detailed instructions for using the visual language model (VL) inference tools in this repository. Each model is designed to process visual question answering tasks using different large multimodal models.

## Table of Contents
1. [Common Setup](#common-setup)
2. [Model-Specific Instructions](#model-specific-instructions)
   - [Qwen2.5-VL](#qwen25-vl)
   - [Gemma 3](#gemma-3)
   - [BLIP2](#blip2)
   - [InternVL3](#internvl3)
   - [LLaVA](#llava)
   - [OVIS2](#ovis2)
3. [Performance Optimization](#performance-optimization)
4. [Batch Processing](#batch-processing)
5. [Data Format Specifications](#data-format-specifications)
6. [Troubleshooting](#troubleshooting)

## Common Setup

### Requirements
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (optional but recommended)
- 16GB+ RAM (32GB+ recommended for larger models)

### Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install tqdm pillow requests
```

### Data Preparation

The input data should be in JSONL format with the following structure:

```json
{
  "id": "image_001",
  "image": "/absolute/path/to/image.jpg",
  "vqa_pairs": [
    {"question": "What is in this image?", "answer": "A cat"},
    {"question": "What color is the animal?", "answer": "Orange"}
  ]
}
```

## Model-Specific Instructions

### Qwen2.5-VL

The Qwen2.5-VL script uses the Qwen2.5-VL-3B-Instruct model for visual question answering.

#### Setup

```bash
# Install model-specific dependencies
pip install qwen-vl-utils[decord]==0.0.8
```

#### Configuration

Edit the `qwen_vl_inference.py` script to configure:

```python
# File paths
INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl"
OUTPUT_JSONL_FILE = "qwen_vl_VQA_test_results.jsonl"

# Image processing settings
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
```

#### Running

```bash
python qwen_vl_inference.py
```

### Gemma 3

The Gemma 3 script uses the Gemma-3-4b-it model for visual question answering.

#### Setup

```bash
# Install model-specific dependencies
pip install bitsandbytes  # Optional, for 8-bit quantization
```

#### Configuration

Edit the `gemma3_inference.py` script to configure:

```python
# Hugging Face access token (required for Gemma)
HF_TOKEN = "your_huggingface_token_here"

# File paths
INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl"
OUTPUT_JSONL_FILE = "gemma3_VQA_test_results.jsonl"
```

#### Running

```bash
python gemma3_inference.py
```

### BLIP2

The BLIP2 script uses BLIP2 models for visual question answering.

#### Configuration

Edit the `blip2_inference.py` script to configure:

```python
# Model configuration
MODEL_NAME = "Salesforce/blip2-opt-6.7b-coco"  # Or other available BLIP2 models

# File paths
INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl"
OUTPUT_JSONL_FILE = "blip2_VQA_test_results.jsonl"
```

#### Running

```bash
python blip2_inference.py
```

### InternVL3

The InternVL3 script uses the InternVL3 model for visual question answering.

#### Configuration

Edit the `internvl3_inference.py` script to configure:

```python
# Model configuration
MODEL_NAME = "OpenGVLab/InternVL-Chat-V1-5"  # Or other available InternVL models

# File paths
INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl"
OUTPUT_JSONL_FILE = "internvl3_VQA_test_results.jsonl"
```

#### Running

```bash
python internvl3_inference.py
```

### LLaVA

The LLaVA script uses LLaVA models for visual question answering.

#### Configuration

Edit the `llava_inference.py` script to configure:

```python
# Model configuration
MODEL_PATH = "liuhaotian/llava-v1.5-13b"  # Or other available LLaVA models

# File paths
INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl"
OUTPUT_JSONL_FILE = "llava_VQA_test_results.jsonl"
```

#### Running

```bash
python llava_inference.py
```

### OVIS2

The OVIS2 script uses the OVIS2 model for visual question answering.

#### Configuration

Edit the `ovis2_inference.py` script to configure:

```python
# Model configuration
MODEL_NAME = "BAAI/OVIS2-SFT-LLAMA3-8B"  # Or other available OVIS2 models

# File paths
INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl"
OUTPUT_JSONL_FILE = "ovis2_VQA_test_results.jsonl"
```

#### Running

```bash
python ovis2_inference.py
```

## Performance Optimization

### Memory Usage

To optimize memory usage:

1. **Quantization**: Enable 8-bit or 4-bit quantization when available
   ```python
   load_in_8bit = True  # or load_in_4bit = True
   ```

2. **Image Resizing**: Adjust image resolution parameters
   ```python
   MIN_PIXELS = 256 * 28 * 28  # Increase for better quality
   MAX_PIXELS = 1280 * 28 * 28  # Decrease for lower memory usage
   ```

3. **Generation Length**: Limit the output token generation
   ```python
   max_new_tokens = 100  # Default is often 200-300
   ```

4. **Batch Size**: Process fewer items at once
   ```python
   # Implemented inherently in scripts through sequential processing
   ```

### Speed Optimization

1. **Flash Attention**: Enable when using CUDA
   ```python
   model = Model.from_pretrained(
       MODEL_NAME,
       torch_dtype=TORCH_DTYPE,
       use_flash_attention_2=True
   )
   ```

2. **Mixed Precision**: Use lower precision for inference
   ```python
   TORCH_DTYPE = torch.float16  # or torch.bfloat16
   ```

## Batch Processing

All models support batch processing of JSONL files. The process:

1. **Automatic Resumption**: If processing is interrupted, it will resume from the last processed image
2. **Progress Tracking**: Shows progress at the dataset and question level
3. **Error Handling**: Automatically retries on memory issues and other recoverable errors

Example of batch processing status output:
```
Processing dataset entries: 56%|█████▌    | 56/100 [45:23<35:12, 50.31s/item]
Image ID: image_056
Image image_056 questions: 100%|██████████| 12/12 [01:52<00:00, 9.37s/item]
```

## Data Format Specifications

### Input Format

Each line in the input JSONL file should contain:

- `id`: Unique identifier for the image/entry
- `image`: Path to the image file (absolute path recommended)
- `vqa_pairs`: Array of question-answer pairs
  - `question`: The question to ask about the image
  - `answer`: The ground truth answer (optional for inference)
  - `question_type`: Type of question (optional)

### Output Format

Each line in the output JSONL file contains:

- All original fields from the input
- `vqa_pairs_with_[model_name]`: Array of processed QA pairs with model answers
  - Contains all original QA pair fields
  - Adds `[model_name]_answer`: The model's answer to the question

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `MAX_PIXELS` value
   - Enable quantization (`load_in_8bit=True`)
   - Use a smaller model variant
   - Process on a machine with more GPU memory

2. **Model Loading Errors**
   - Update transformers: `pip install -U transformers`
   - Check for model-specific dependencies
   - Verify Hugging Face login for gated models

3. **Slow Processing**
   - Use GPU acceleration if available
   - Enable flash attention for supported models
   - Reduce image resolution (lower `MIN_PIXELS` and `MAX_PIXELS`)

4. **Image Loading Errors**
   - Verify image paths are correct and accessible
   - Check image format support (PNG, JPEG, etc.)
   - Use absolute paths instead of relative paths

5. **JSON Parsing Errors**
   - Check input file formatting
   - Ensure UTF-8 encoding is used
   - Validate JSON structure with a JSON validator 