import torch
from PIL import Image
import requests
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import json
import os
import time
from tqdm.autonotebook import tqdm
import io
import base64
from huggingface_hub import login

# --- Configuration ---
MODEL_ID = "google/gemma-3-4b-it"
HF_TOKEN = ""  # Hugging Face access token

# Login to Hugging Face to access restricted models
print(f"Logging into Hugging Face with provided token...")
login(token=HF_TOKEN)

# Device configuration
if torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float32  # Use float32 on MPS due to potential bfloat16 compatibility issues
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32  # float32 is more stable on CPU

print(f"Using device: {DEVICE} with dtype: {TORCH_DTYPE}")

# Input/output file configuration
INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl"  # Your input file
OUTPUT_JSONL_FILE = "gemma3_VQA_test_results.jsonl"  # Results save location

# --- Model and processor ---
model = None
processor = None

def load_image(image_path):
    """Load an image"""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        tqdm.write(f"Error loading image {image_path}: {e}")
        return None

def load_model_and_processor():
    """Load model and processor"""
    global model, processor
    if model is None or processor is None:
        tqdm.write(f"Loading model: {MODEL_ID} to device: {DEVICE}")
        try:
            # Determine whether to use quantization based on device
            load_in_8bit = False  # Default: don't use 8-bit quantization
            if DEVICE == "cuda":
                # Only try 8-bit quantization on CUDA devices
                try:
                    import bitsandbytes as bnb
                    load_in_8bit = True
                    tqdm.write("Enabling 8-bit quantization to reduce GPU memory usage")
                except ImportError:
                    tqdm.write("bitsandbytes library not found, using full precision model")
            
            # Load model
            model = Gemma3ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                load_in_8bit=load_in_8bit,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
            
            # Load processor
            processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True
            )
            
            tqdm.write("Model and processor loaded successfully.")
        except Exception as e:
            tqdm.write(f"Error loading model: {e}")
            tqdm.write("Please ensure you have enough RAM and the correct PyTorch version.")
            exit(1)

def get_gemma3_answer(image_path, question, retries=2, delay=5, pbar_questions=None, max_new_tokens=200):
    """
    Get an answer from Gemma 3 model for a given image and question.
    Includes basic retry logic.
    """
    global model, processor

    if model is None or processor is None:
        load_model_and_processor()

    try:
        if not os.path.isabs(image_path):
            if pbar_questions: pbar_questions.write(f"Warning: Image path {image_path} appears to be relative. Assuming it is correct or accessible.")
            else: tqdm.write(f"Warning: Image path {image_path} appears to be relative. Assuming it is correct or accessible.")
        
        image = load_image(image_path)
        if image is None:
            return "Error: Unable to load image"
            
    except FileNotFoundError:
        if pbar_questions: pbar_questions.write(f"Error: Image not found {image_path}")
        else: tqdm.write(f"Error: Image not found {image_path}")
        return "Error: Image not found"
    except Exception as e:
        if pbar_questions: pbar_questions.write(f"Error opening image {image_path}: {e}")
        else: tqdm.write(f"Error opening image {image_path}: {e}")
        return f"Error opening image: {e}"

    # Format messages for Gemma 3
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful AI assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    for attempt in range(retries + 1):
        try:
            # Apply chat template using processor
            inputs = processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(model.device, dtype=TORCH_DTYPE)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate answer
            with torch.inference_mode():
                generation = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False
                )
                generation = generation[0][input_len:]
            
            # Decode answer
            response = processor.decode(generation, skip_special_tokens=True)
            return response
            
        except RuntimeError as e:
            error_msg_prefix = f"Runtime error (possibly {DEVICE} out of memory) attempt {attempt + 1}"
            if "out of memory" in str(e).lower() or "allocated tensor is too large" in str(e):
                if pbar_questions: pbar_questions.write(f"{error_msg_prefix}: {e}")
                else: tqdm.write(f"{error_msg_prefix}: {e}")

                if attempt < retries:
                    if pbar_questions: pbar_questions.write(f"Retrying in {delay} seconds...")
                    else: tqdm.write(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    if DEVICE == "cuda": torch.cuda.empty_cache()
                    elif DEVICE == "mps": torch.mps.empty_cache()
                else:
                    if pbar_questions: pbar_questions.write("Maximum retries reached for out of memory error.")
                    else: tqdm.write("Maximum retries reached for out of memory error.")
                    return f"Error: {DEVICE} out of memory, failed after retries"
            else:
                if pbar_questions: pbar_questions.write(f"Unexpected runtime error on attempt {attempt + 1}: {e}")
                else: tqdm.write(f"Unexpected runtime error on attempt {attempt + 1}: {e}")
                if attempt < retries: time.sleep(delay)
                else: return f"Error: Runtime error after retries: {e}"
        except Exception as e:
            if pbar_questions: pbar_questions.write(f"Error during Gemma 3 inference attempt {attempt + 1}: {e}")
            else: tqdm.write(f"Error during Gemma 3 inference attempt {attempt + 1}: {e}")
            if attempt < retries:
                if pbar_questions: pbar_questions.write(f"Retrying in {delay} seconds...")
                else: tqdm.write(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else: return f"Error: Gemma 3 inference failed after retries: {e}"
    return "Error: Maximum number of inference retries reached."

def process_dataset():
    """Process all images and questions in the dataset"""
    load_model_and_processor()

    processed_image_ids = set()
    if os.path.exists(OUTPUT_JSONL_FILE):
        try:
            with open(OUTPUT_JSONL_FILE, 'r', encoding='utf-8') as f_out_read:
                for line in f_out_read:
                    try:
                        data = json.loads(line)
                        if "id" in data: processed_image_ids.add(data["id"])
                    except json.JSONDecodeError:
                        tqdm.write(f"Warning: Could not decode line in existing output file: {line.strip()}")
            tqdm.write(f"Continuing processing. Found {len(processed_image_ids)} already processed image IDs in {OUTPUT_JSONL_FILE}")
        except Exception as e:
            tqdm.write(f"Error reading existing output file {OUTPUT_JSONL_FILE}: {e}. Starting from scratch or may overwrite.")

    # Get total number of lines for main progress bar
    try:
        with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f_count:
            num_total_lines = sum(1 for _ in f_count)
    except FileNotFoundError:
        tqdm.write(f"Error: Input file '{INPUT_JSONL_FILE}' not found.")
        return
    except Exception as e:
        tqdm.write(f"Error counting lines in '{INPUT_JSONL_FILE}': {e}")
        return

    if num_total_lines == 0:
        tqdm.write(f"Input file '{INPUT_JSONL_FILE}' is empty.")
        return

    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL_FILE, 'a', encoding='utf-8') as f_out:

        # Main progress bar for dataset entries
        dataset_pbar = tqdm(enumerate(f_in), total=num_total_lines, desc="Processing dataset entries")
        for line_number, line in dataset_pbar:
            try:
                entry = json.loads(line)
                image_id = entry["id"]
                original_image_path = entry["image"]

                if image_id in processed_image_ids:
                    # No need to print, the main progress bar will skip
                    continue

                current_image_path = original_image_path  # Assuming it's an absolute path

                dataset_pbar.set_postfix_str(f"Image ID: {image_id}", refresh=True)  # Update current item description

                if not os.path.exists(current_image_path):
                    tqdm.write(f"Image file not found: {current_image_path} ID {image_id}. Skipping line {line_number + 1}.")
                    error_entry = entry.copy()
                    error_entry["gemma3_processing_error"] = f"Image file not found at {current_image_path}"
                    error_entry["vqa_pairs_with_gemma3"] = [
                        {**qa_pair, "gemma3_answer": "Error: Image file not found"}
                        for qa_pair in entry.get("vqa_pairs", [])
                    ]
                    f_out.write(json.dumps(error_entry) + "\n")
                    f_out.flush()
                    processed_image_ids.add(image_id)
                    continue

                output_entry = entry.copy()
                output_entry["vqa_pairs_with_gemma3"] = []
                has_errors_for_this_image = False
                
                questions_list = entry.get("vqa_pairs", [])
                # Image-level question progress bar
                questions_pbar = tqdm(enumerate(questions_list), total=len(questions_list), desc=f"Image {image_id} questions", leave=False, position=1)
                for i, qa_pair in questions_pbar:
                    question = qa_pair.get("question", "Missing question")
                    questions_pbar.set_postfix_str(f"Q: {question[:30]}...", refresh=True)

                    # Pass questions_pbar to get_gemma3_answer so it can use tqdm.write when needed
                    gemma3_answer = get_gemma3_answer(current_image_path, question, pbar_questions=questions_pbar)

                    new_qa_pair = qa_pair.copy()
                    new_qa_pair["gemma3_answer"] = gemma3_answer
                    output_entry["vqa_pairs_with_gemma3"].append(new_qa_pair)

                    if "Error:" in gemma3_answer:
                        has_errors_for_this_image = True
                
                questions_pbar.close()  # Explicitly close internal progress bar

                f_out.write(json.dumps(output_entry) + "\n")
                f_out.flush()
                processed_image_ids.add(image_id)

            except json.JSONDecodeError:
                tqdm.write(f"Error decoding JSON from line {line_number + 1}: {line.strip()}")
                f_out.write(json.dumps({"error": "JSONDecodeError", "line_content": line.strip()}) + "\n")
                f_out.flush()
            except KeyError as e:
                tqdm.write(f"KeyError encountered in processing line {line_number + 1}: {e}. Line content: {line.strip()}")
                f_out.write(json.dumps({"error": f"KeyError: {e}", "line_content": line.strip()}) + "\n")
                f_out.flush()
            except Exception as e:
                tqdm.write(f"Unexpected error encountered in processing line {line_number + 1}: {e}, content: {line.strip()}")
                f_out.write(json.dumps({"error": "Unexpected processing error", "exception": str(e), "line_content": line.strip()}) + "\n")
                f_out.flush()
        dataset_pbar.close()  # Explicitly close main progress bar

if __name__ == "__main__":
    if not os.path.exists(INPUT_JSONL_FILE):
        print(f"Error: Input file '{INPUT_JSONL_FILE}' not found.")  # print before tqdm starts
    else:
        process_dataset()
        print(f"\nProcessing completed. Results saved to {OUTPUT_JSONL_FILE}")  # print after tqdm ends 