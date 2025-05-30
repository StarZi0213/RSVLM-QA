import torch
from PIL import Image
import requests # Not strictly needed for local files, but good for LLaVA examples
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
import os
import time
from tqdm.autonotebook import tqdm # tqdm can be used directly, no need for as tqdm

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# For M-series Macs, 'mps' should be prioritized.
# If 'mps' is not available or causes issues, fall back to 'cpu'.
if torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float16 # float16 is generally faster and uses less memory on MPS
elif torch.cuda.is_available():
    DEVICE = "cuda" # For NVIDIA GPUs
    TORCH_DTYPE = torch.float16
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32 # float32 is more stable on CPU

print(f"Using device: {DEVICE} with dtype: {TORCH_DTYPE}")

INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl" # Your input file
OUTPUT_JSONL_FILE = "llava_VQA_test_results.jsonl"    # Where results will be saved
# IMAGE_BASE_PATH is not strictly needed if your JSON contains absolute paths that are correct.
# If paths in JSON are relative or incorrect, you might use this:
# IMAGE_BASE_PATH = "/Users/j4-ai/AI_Research/Xing/Dataset/RSVL-VQA/RSVL-VQA/INRIA-Aerial-Image-Labeling/train/images/"

# --- Load Model and Processor ---
# Initialize model and processor as None globally or pass them around
model = None
processor = None

def load_model_and_processor():
    global model, processor
    if model is None or processor is None:
        tqdm.write(f"Loading model: {MODEL_ID} onto device: {DEVICE}") # Use tqdm.write for cleaner output with progress bars
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True, # Helps reduce CPU RAM peak during loading
            ).to(DEVICE)
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            tqdm.write("Model and processor loaded successfully.")
        except Exception as e:
            tqdm.write(f"Error loading model: {e}")
            tqdm.write("Please ensure you have sufficient RAM and the correct PyTorch version.")
            if DEVICE == "mps":
                tqdm.write("If using MPS, ensure it's properly set up. Consider torch_dtype=torch.float32 if float16 causes issues.")
            exit()

def get_llava_answer(image_path, question, retries=2, delay=5, pbar_questions=None):
    """
    Gets an answer from the LLaVA model for a given image and question.
    Includes basic retry logic.
    pbar_questions is the tqdm instance for the inner loop (questions)
    """
    global model, processor

    if model is None or processor is None:
        load_model_and_processor()

    try:
        if not os.path.isabs(image_path):
             if pbar_questions: pbar_questions.write(f"Warning: Image path {image_path} seems relative. Assuming it's correct or accessible.")
             else: tqdm.write(f"Warning: Image path {image_path} seems relative. Assuming it's correct or accessible.")
             current_image_path = image_path
        else:
            current_image_path = image_path

        raw_image = Image.open(current_image_path).convert('RGB')
    except FileNotFoundError:
        if pbar_questions: pbar_questions.write(f"Error: Image not found at {current_image_path}")
        else: tqdm.write(f"Error: Image not found at {current_image_path}")
        return "Error: Image not found"
    except Exception as e:
        if pbar_questions: pbar_questions.write(f"Error opening image {current_image_path}: {e}")
        else: tqdm.write(f"Error opening image {current_image_path}: {e}")
        return f"Error opening image: {e}"

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    for attempt in range(retries + 1):
        try:
            # For potentially long generations, it's good not to print too much *inside* this tight loop
            # The question progress bar will show activity.
            inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(DEVICE, dtype=TORCH_DTYPE if DEVICE != "cpu" else torch.float32)
            generate_ids = model.generate(**inputs, max_new_tokens=150)
            full_decoded_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            assistant_keyword = "ASSISTANT:"
            assistant_index = full_decoded_text.rfind(assistant_keyword)
            if assistant_index != -1:
                generated_text = full_decoded_text[assistant_index + len(assistant_keyword):].strip()
            else:
                temp_prompt_for_stripping = prompt.replace("<image>\n", "")
                if full_decoded_text.startswith(temp_prompt_for_stripping):
                     generated_text = full_decoded_text[len(temp_prompt_for_stripping):].strip()
                else:
                    if pbar_questions: pbar_questions.write(f"Warning: Could not cleanly separate answer from prompt. Full output: {full_decoded_text}")
                    else: tqdm.write(f"Warning: Could not cleanly separate answer from prompt. Full output: {full_decoded_text}")
                    generated_text = full_decoded_text
            # No print here for "LLaVA Answer" to keep progress bar clean during question processing
            return generated_text
        except RuntimeError as e:
            error_msg_prefix = f"RuntimeError (likely OOM on {DEVICE}) on attempt {attempt + 1}"
            if "MPS backend out of memory" in str(e) or "allocated tensor is too large" in str(e) or "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in str(e):
                if pbar_questions: pbar_questions.write(f"{error_msg_prefix}: {e}")
                else: tqdm.write(f"{error_msg_prefix}: {e}")

                if attempt < retries:
                    if pbar_questions: pbar_questions.write(f"Retrying in {delay} seconds...")
                    else: tqdm.write(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    if DEVICE == "mps": torch.mps.empty_cache()
                    elif DEVICE == "cuda": torch.cuda.empty_cache()
                else:
                    if pbar_questions: pbar_questions.write("Max retries reached for OOM error.")
                    else: tqdm.write("Max retries reached for OOM error.")
                    return f"Error: {DEVICE} out of memory after retries"
            else:
                if pbar_questions: pbar_questions.write(f"An unexpected RuntimeError occurred on attempt {attempt + 1}: {e}")
                else: tqdm.write(f"An unexpected RuntimeError occurred on attempt {attempt + 1}: {e}")
                if attempt < retries: time.sleep(delay)
                else: return f"Error: Runtime error after retries: {e}"
        except Exception as e:
            if pbar_questions: pbar_questions.write(f"Error during LLaVA inference on attempt {attempt + 1}: {e}")
            else: tqdm.write(f"Error during LLaVA inference on attempt {attempt + 1}: {e}")
            if attempt < retries:
                if pbar_questions: pbar_questions.write(f"Retrying in {delay} seconds...")
                else: tqdm.write(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else: return f"Error: LLaVA inference failed after retries: {e}"
    return "Error: Max retries reached for inference."


def process_dataset():
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
            tqdm.write(f"Resuming. Found {len(processed_image_ids)} already processed image IDs in {OUTPUT_JSONL_FILE}")
        except Exception as e:
            tqdm.write(f"Error reading existing output file {OUTPUT_JSONL_FILE}: {e}. Starting fresh or may overwrite.")

    # Get total number of lines for the main progress bar
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

    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL_FILE, 'a', encoding='utf-8') as f_out:

        # Main progress bar for processing entries in the dataset
        dataset_pbar = tqdm(enumerate(f_in), total=num_total_lines, desc="Processing Dataset Entries")
        for line_number, line in dataset_pbar:
            try:
                entry = json.loads(line)
                image_id = entry["id"]
                original_image_path = entry["image"]

                if image_id in processed_image_ids:
                    # No need to print here, the main progress bar will just skip
                    continue

                current_image_path = original_image_path # Assuming absolute paths

                dataset_pbar.set_postfix_str(f"Img ID: {image_id}", refresh=True) # Update description for current item

                if not os.path.exists(current_image_path):
                    tqdm.write(f"Image file not found: {current_image_path} for ID {image_id}. Skipping line {line_number + 1}.")
                    error_entry = entry.copy()
                    error_entry["llava_processing_error"] = f"Image file not found at {current_image_path}"
                    error_entry["vqa_pairs_with_llava"] = [
                        {**qa_pair, "llava_answer": "Error: Image file not found"}
                        for qa_pair in entry.get("vqa_pairs", [])
                    ]
                    f_out.write(json.dumps(error_entry) + "\n")
                    f_out.flush()
                    processed_image_ids.add(image_id)
                    continue

                output_entry = entry.copy()
                output_entry["vqa_pairs_with_llava"] = []
                has_errors_for_this_image = False
                
                questions_list = entry.get("vqa_pairs", [])
                # Inner progress bar for questions within an image
                questions_pbar = tqdm(enumerate(questions_list), total=len(questions_list), desc=f"Img {image_id} Qs", leave=False, position=1)
                for i, qa_pair in questions_pbar:
                    question = qa_pair.get("question", "Missing question")
                    questions_pbar.set_postfix_str(f"Q: {question[:30]}...", refresh=True)

                    # Pass the questions_pbar to get_llava_answer so it can use tqdm.write if needed
                    llava_answer = get_llava_answer(current_image_path, question, pbar_questions=questions_pbar)

                    new_qa_pair = qa_pair.copy()
                    new_qa_pair["llava_answer"] = llava_answer
                    output_entry["vqa_pairs_with_llava"].append(new_qa_pair)

                    if "Error:" in llava_answer:
                        has_errors_for_this_image = True
                
                questions_pbar.close() # Explicitly close inner progress bar

                f_out.write(json.dumps(output_entry) + "\n")
                f_out.flush()
                processed_image_ids.add(image_id)
                
                # Update main progress bar description after processing an item
                # dataset_pbar.set_description_str(f"Processing Dataset (Last ID: {image_id})")


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
        dataset_pbar.close() # Explicitly close main progress bar

if __name__ == "__main__":
    if not os.path.exists(INPUT_JSONL_FILE):
        print(f"Error: Input file '{INPUT_JSONL_FILE}' not found.")  # print is fine before tqdm starts
    else:
        process_dataset()
        print(f"\nProcessing complete. Results saved to {OUTPUT_JSONL_FILE}")  # print is fine after tqdm ends