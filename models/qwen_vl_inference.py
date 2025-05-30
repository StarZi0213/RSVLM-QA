import os
import json
import torch
from PIL import Image
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from io import BytesIO
import argparse
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import base64
import re
from qwen_vl_utils import process_vision_info

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Device selection logic
if torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float16  # float16 is generally faster and uses less memory on MPS
elif torch.cuda.is_available():
    DEVICE = "cuda"  # For NVIDIA GPUs
    TORCH_DTYPE = torch.float16
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32  # float32 is more stable on CPU

print(f"Using device: {DEVICE} with dtype: {TORCH_DTYPE}")

INPUT_JSONL_FILE = "json/RSVL-VQA_test.jsonl"  # Input file
OUTPUT_JSONL_FILE = "json/qwen_vl_VQA_test_results.jsonl"  # Results save location
CHECKPOINT_FILE = "qwen_vl_inference_checkpoint.json"  # Checkpoint file

# Image min/max pixel settings (for performance)
MIN_PIXELS = 256 * 28 * 28  # Minimum pixel count
MAX_PIXELS = 1280 * 28 * 28  # Maximum pixel count

# --- Load Model and Processor ---
# Initialize global variables
model = None
processor = None
tokenizer = None

def load_model_and_processor():
    """Load Qwen2.5-VL model and processor"""
    global model, processor, tokenizer
    if model is None or processor is None:
        tqdm.write(f"Loading model: {MODEL_ID} to device: {DEVICE}")
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device_map="auto" if DEVICE == "cuda" else None,
                low_cpu_mem_usage=True,  # Reduce CPU memory peak during loading
            )
            if DEVICE != "cuda":  # If not using device_map="auto"
                model = model.to(DEVICE)
                
            processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            tqdm.write("Model and processor loaded successfully.")
        except Exception as e:
            tqdm.write(f"Error loading model: {e}")
            tqdm.write("Please ensure you have enough RAM and the correct PyTorch version.")
            if DEVICE == "mps":
                tqdm.write("If using MPS, ensure it's properly set up. Consider torch_dtype=torch.float32 if float16 causes issues.")
            exit()

def get_qwen_vl_answer(image_path, question, retries=2, delay=5, pbar_questions=None):
    """
    Generate an answer using Qwen2.5-VL model for a given image and question.
    Includes basic retry logic.
    pbar_questions is the tqdm instance for the inner loop (questions)
    """
    global model, processor, tokenizer

    if model is None or processor is None:
        load_model_and_processor()

    try:
        if not os.path.isabs(image_path):
            if pbar_questions: pbar_questions.write(f"Warning: Image path {image_path} seems relative. Assuming it's correct or accessible.")
            else: tqdm.write(f"Warning: Image path {image_path} seems relative. Assuming it's correct or accessible.")
            current_image_path = image_path
        else:
            current_image_path = image_path

        # Construct message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{current_image_path}"},
                    {"type": "text", "text": question}
                ]
            }
        ]
    except Exception as e:
        if pbar_questions: pbar_questions.write(f"Error preparing image {current_image_path}: {e}")
        else: tqdm.write(f"Error preparing image {current_image_path}: {e}")
        return f"Error preparing image: {e}"

    for attempt in range(retries + 1):
        try:
            # Prepare inputs
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(DEVICE)
            
            # Generate answer
            generated_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False, num_beams=5)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answer = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return answer
        except RuntimeError as e:
            error_msg_prefix = f"Runtime error (likely OOM on {DEVICE}) attempt {attempt + 1}"
            if "MPS backend out of memory" in str(e) or "allocated tensor is too large" in str(e) or "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in str(e) or "CUDA out of memory" in str(e):
                if pbar_questions: pbar_questions.write(f"{error_msg_prefix}: {e}")
                else: tqdm.write(f"{error_msg_prefix}: {e}")

                if attempt < retries:
                    if pbar_questions: pbar_questions.write(f"Retrying in {delay} seconds...")
                    else: tqdm.write(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    if DEVICE == "mps": torch.mps.empty_cache()
                    elif DEVICE == "cuda": torch.cuda.empty_cache()
                else:
                    if pbar_questions: pbar_questions.write("Maximum retries reached for OOM error.")
                    else: tqdm.write("Maximum retries reached for OOM error.")
                    return f"Error: {DEVICE} out of memory after retries"
            else:
                if pbar_questions: pbar_questions.write(f"Unexpected runtime error on attempt {attempt + 1}: {e}")
                else: tqdm.write(f"Unexpected runtime error on attempt {attempt + 1}: {e}")
                if attempt < retries: time.sleep(delay)
                else: return f"Error: Runtime error after retries: {e}"
        except Exception as e:
            if pbar_questions: pbar_questions.write(f"Error during Qwen-VL inference attempt {attempt + 1}: {e}")
            else: tqdm.write(f"Error during Qwen-VL inference attempt {attempt + 1}: {e}")
            if attempt < retries:
                if pbar_questions: pbar_questions.write(f"Retrying in {delay} seconds...")
                else: tqdm.write(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else: return f"Error: Qwen-VL inference failed after retries: {e}"
    return "Error: Maximum number of inference retries reached."

def process_dataset():
    """Process the entire dataset"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSONL_FILE), exist_ok=True)
    
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

    # Check if checkpoint exists and load processed IDs
    if os.path.exists(CHECKPOINT_FILE) and len(processed_image_ids) == 0:
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint_data = json.load(f)
                processed_ids_from_checkpoint = set(checkpoint_data.get('processed_ids', []))
                processed_image_ids.update(processed_ids_from_checkpoint)
            tqdm.write(f"Restored from checkpoint: {len(processed_ids_from_checkpoint)} records already processed")
        except Exception as e:
            tqdm.write(f"Error reading checkpoint file: {e}")

    # Get total line count for main progress bar
    try:
        with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f_count:
            num_total_lines = sum(1 for _ in f_count)
    except FileNotFoundError:
        tqdm.write(f"Error: Input file '{INPUT_JSONL_FILE}' not found for counting lines.")
        return
    except Exception as e:
        tqdm.write(f"Error counting lines in '{INPUT_JSONL_FILE}': {e}")
        return

    if num_total_lines == 0:
        tqdm.write(f"Input file '{INPUT_JSONL_FILE}' is empty.")
        return

    # Record start time
    start_time = time.time()

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
                    # No need to print here, the main progress bar will just skip
                    continue

                current_image_path = original_image_path  # Assuming absolute paths

                dataset_pbar.set_postfix_str(f"Image ID: {image_id}", refresh=True)  # Update description for current item

                if not os.path.exists(current_image_path):
                    tqdm.write(f"Image file not found: {current_image_path} for ID {image_id}. Skipping line {line_number + 1}.")
                    error_entry = entry.copy()
                    error_entry["qwen_vl_processing_error"] = f"Image file not found at {current_image_path}"
                    error_entry["vqa_pairs_with_qwen_vl"] = [
                        {**qa_pair, "qwen_vl_answer": "Error: Image file not found"}
                        for qa_pair in entry.get("vqa_pairs", [])
                    ]
                    f_out.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                    f_out.flush()
                    processed_image_ids.add(image_id)
                    continue

                output_entry = entry.copy()
                output_entry["vqa_pairs_with_qwen_vl"] = []
                has_errors_for_this_image = False
                
                questions_list = entry.get("vqa_pairs", [])
                # Inner progress bar for questions within an image
                questions_pbar = tqdm(enumerate(questions_list), total=len(questions_list), desc=f"Image {image_id} questions", leave=False, position=1)
                for i, qa_pair in questions_pbar:
                    question = qa_pair.get("question", "Missing question")
                    questions_pbar.set_postfix_str(f"Q: {question[:30]}...", refresh=True)

                    # Pass questions_pbar to get_qwen_vl_answer so it can use tqdm.write if needed
                    qwen_vl_answer = get_qwen_vl_answer(current_image_path, question, pbar_questions=questions_pbar)

                    new_qa_pair = qa_pair.copy()
                    new_qa_pair["qwen_vl_answer"] = qwen_vl_answer
                    output_entry["vqa_pairs_with_qwen_vl"].append(new_qa_pair)

                    if "Error:" in qwen_vl_answer:
                        has_errors_for_this_image = True
                
                questions_pbar.close()  # Explicitly close inner progress bar

                f_out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                f_out.flush()
                processed_image_ids.add(image_id)
                
                # Update checkpoint after each image
                try:
                    with open(CHECKPOINT_FILE, 'w') as f:
                        json.dump({'processed_ids': list(processed_image_ids)}, f)
                except Exception as e:
                    tqdm.write(f"Warning: Could not update checkpoint file: {e}")
                    
                # Calculate and show ETA
                elapsed_time = time.time() - start_time
                processed_count = len(processed_image_ids)
                if processed_count > 0:
                    avg_time_per_item = elapsed_time / processed_count
                    remaining_items = num_total_lines - line_number - 1
                    remaining_items_to_process = sum(1 for i in range(line_number + 1, num_total_lines) 
                                             if i < num_total_lines and json.loads(open(INPUT_JSONL_FILE).readlines()[i])["id"] not in processed_image_ids)
                    eta_seconds = avg_time_per_item * remaining_items_to_process
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    dataset_pbar.set_description(f"Processing dataset entries (ETA: {eta_str})")

            except json.JSONDecodeError:
                tqdm.write(f"Error decoding JSON from line {line_number + 1}: {line.strip()}")
                f_out.write(json.dumps({"error": "JSONDecodeError", "line_content": line.strip()}) + "\n")
                f_out.flush()
            except KeyError as e:
                tqdm.write(f"KeyError processing line {line_number + 1}: {e}. Line content: {line.strip()}")
                f_out.write(json.dumps({"error": f"KeyError: {e}", "line_content": line.strip()}) + "\n")
                f_out.flush()
            except Exception as e:
                tqdm.write(f"An unexpected error occurred while processing line {line_number + 1}: {e}, Content: {line.strip()}")
                f_out.write(json.dumps({"error": "Unexpected processing error", "exception": str(e), "line_content": line.strip()}) + "\n")
                f_out.flush()
        dataset_pbar.close()  # Explicitly close main progress bar

    # Calculate and display final statistics
    total_time = time.time() - start_time
    total_processed = len(processed_image_ids)
    avg_time = total_time / total_processed if total_processed > 0 else 0
    
    tqdm.write(f"\nProcessing complete!")
    tqdm.write(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    tqdm.write(f"Total images processed: {total_processed}")
    tqdm.write(f"Average time per image: {time.strftime('%H:%M:%S', time.gmtime(avg_time))}")
    tqdm.write(f"Results saved to: {OUTPUT_JSONL_FILE}")
    
    # Remove checkpoint file if processing completed successfully
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            tqdm.write("Checkpoint file removed as processing completed successfully.")
        except Exception as e:
            tqdm.write(f"Note: Could not remove checkpoint file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL file containing image paths and questions using Qwen2.5-VL.")
    parser.add_argument("--input", type=str, help="Input JSONL file path", default=INPUT_JSONL_FILE)
    parser.add_argument("--output", type=str, help="Output JSONL file path", default=OUTPUT_JSONL_FILE)
    parser.add_argument("--device", type=str, help="Device to use (cuda, mps, cpu)", default=DEVICE)
    args = parser.parse_args()
    
    # Update global variables if command line arguments provided
    if args.input != INPUT_JSONL_FILE:
        INPUT_JSONL_FILE = args.input
        print(f"Using input file: {INPUT_JSONL_FILE}")
    
    if args.output != OUTPUT_JSONL_FILE:
        OUTPUT_JSONL_FILE = args.output
        print(f"Using output file: {OUTPUT_JSONL_FILE}")
    
    if args.device != DEVICE and args.device in ["cuda", "mps", "cpu"]:
        DEVICE = args.device
        print(f"Using device: {DEVICE}")
        
        # Update dtype based on new device
        if DEVICE == "cpu":
            TORCH_DTYPE = torch.float32
        else:
            TORCH_DTYPE = torch.float16
    
    if not os.path.exists(INPUT_JSONL_FILE):
        print(f"Error: Input file '{INPUT_JSONL_FILE}' not found.")
    else:
        process_dataset() 