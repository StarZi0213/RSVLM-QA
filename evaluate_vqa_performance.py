import os
import json
import time
import argparse
import traceback
from tqdm import tqdm
import requests
import threading
import queue
import random
import backoff
import signal
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# API key from environment variable or default
DEFAULT_API_KEY = ''

# API Rate limits for GPT-4.1
MAX_REQUESTS_PER_MINUTE = 500  # 500 requests per minute
MAX_TOKENS_PER_MINUTE = 200000  # 200k tokens per minute

# Tracking processed items to prevent duplicates
processed_lock = threading.Lock()
currently_processing = set()

# Global variables for tracking progress and handling signals
checkpoint_lock = threading.Lock()
current_processed_ids = set()  # Global set of processed IDs for signal handler

def signal_handler(sig, frame):
    """Handle interrupt signals (CTRL+C) by saving checkpoint before exiting"""
    print("\nProgram received interrupt signal, saving checkpoint...")
    # Use a copy of the current processed IDs to avoid race conditions
    with checkpoint_lock:
        processed_ids_to_save = current_processed_ids.copy()
    
    # Get checkpoint file from command line args
    checkpoint_file = None
    for i, arg in enumerate(sys.argv):
        if arg == "--checkpoint" and i + 1 < len(sys.argv):
            checkpoint_file = sys.argv[i + 1]
            break
        elif arg.startswith("--checkpoint="):
            checkpoint_file = arg.split("=", 1)[1]
            break
    
    # Default checkpoint file if not specified
    if not checkpoint_file:
        checkpoint_file = "./vqa_evaluation_checkpoint.json"
    
    # Save checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "processed_ids": list(processed_ids_to_save),
            "timestamp": time.time(),
            "interrupted": True
        }, f)
    
    print(f"Saved progress of {len(processed_ids_to_save)} records to {checkpoint_file}")
    print("You can restart the program with the same command, and it will continue from where it left off.")
    sys.exit(0)

# Register signal handler for SIGINT (CTRL+C)
signal.signal(signal.SIGINT, signal_handler)

# RateLimiter class to handle API rate limiting
class RateLimiter:
    def __init__(self, max_rpm, max_tpm):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.request_timestamps = []
        self.token_counts = []
        self.lock = threading.Lock()
        
    def wait_if_needed(self, estimated_tokens=400):
        """Wait if we're approaching rate limits"""
        with self.lock:
            now = time.time()
            
            # Clean up old timestamps (older than 1 minute)
            one_minute_ago = now - 60
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > one_minute_ago]
            self.token_counts = [tc for i, tc in enumerate(self.token_counts) 
                               if i < len(self.request_timestamps) and self.request_timestamps[i] > one_minute_ago]
            
            # Check if we're approaching request rate limit
            if len(self.request_timestamps) >= self.max_rpm * 0.95:
                # Wait until oldest request is more than a minute old
                wait_time = 60 - (now - self.request_timestamps[0]) + 0.1
                if wait_time > 0:
                    return wait_time
            
            # Check if we're approaching token rate limit
            current_token_count = sum(self.token_counts)
            if current_token_count + estimated_tokens >= self.max_tpm * 0.95:
                # Wait until oldest token count is more than a minute old
                wait_time = 60 - (now - self.request_timestamps[0]) + 0.1
                if wait_time > 0:
                    return wait_time
            
            return 0
    
    def add_request(self, token_count):
        """Record a new request with its token count"""
        with self.lock:
            now = time.time()
            self.request_timestamps.append(now)
            self.token_counts.append(token_count)

# Global rate limiter
rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE)

# Statistics collector
class StatsCollector:
    def __init__(self):
        self.processed_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_tokens = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def add_result(self, success, token_count=0):
        with self.lock:
            self.processed_count += 1
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            self.total_tokens += token_count
    
    def get_stats(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            elapsed_minutes = elapsed / 60
            
            return {
                "processed": self.processed_count,
                "success": self.success_count,
                "failure": self.failure_count,
                "total_tokens": self.total_tokens,
                "elapsed_seconds": elapsed,
                "requests_per_minute": self.processed_count / elapsed_minutes if elapsed_minutes > 0 else 0,
                "tokens_per_minute": self.total_tokens / elapsed_minutes if elapsed_minutes > 0 else 0,
            }

# Global stats collector
stats = StatsCollector()

# Backoff decorator for API calls with exponential retry
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError),
    max_tries=5,
    jitter=backoff.full_jitter
)
def evaluate_answer_with_gpt(item, api_key):
    """Evaluate VQA answer using GPT-4.1 API"""
    try:
        # Verify API key is provided
        if not api_key:
            print("❌ Error: No API key provided. Set the OPENAI_API_KEY environment variable or use --api-key.")
            return None
        
        # Extract data from the record
        question = item.get("question", "")
        ground_truth = item.get("answer", "")
        model_answer = item.get("model_answer", "")
        question_type = item.get("question_type", "")
        
        # Different prompts for caption vs. other question types
        if question_type == "caption":
            prompt = f"""You are an expert evaluator for image captioning tasks. Please evaluate the quality of the following caption on a scale of 0-100, where 0 is completely wrong and 100 is perfect.

Task: {question}
Reference caption: {ground_truth}
Generated caption: {model_answer}

IMPORTANT: Your response MUST FOLLOW this EXACT format without any extra text:
<score>X</score><reason>Detailed explanation of your score with specific observations</reason>

Where X is a number between 0 and 100. Do not include any text outside these tags.
Focus on content accuracy, completeness, detail, and language quality. Be fair but critical in your evaluation."""
        else:
            prompt = f"""You are an expert evaluator for visual question answering tasks. Please evaluate if the model's answer properly addresses the question asked, based on the ground truth answer.

Question: {question}
Ground truth answer: {ground_truth}
Model's answer: {model_answer}

IMPORTANT: Your response MUST FOLLOW this EXACT format without any extra text:
<judge>Correct</judge><reason>Detailed explanation of your judgment with specific observations</reason>
OR
<judge>Wrong</judge><reason>Detailed explanation of your judgment with specific observations</reason>

Do not include any text outside these tags. Be fair but critical in your evaluation. Consider the semantic meaning rather than exact wording."""
        
        # Estimate token count for this request
        estimated_token_count = len(prompt.split()) * 1.3 + 500  # Rough estimate
        
        # Wait if approaching rate limits
        wait_time = rate_limiter.wait_if_needed(estimated_token_count)
        if wait_time > 0:
            print(f"Approaching rate limit, waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Add small random delay to prevent burst
        time.sleep(random.uniform(0.0, 0.1))
        
        # Prepare the request
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in evaluating visual question answering and image captioning results. You MUST provide evaluations in the EXACT format specified, with no additional text."},
            {"role": "user", "content": prompt}
        ]
        
        # API request parameters
        payload = {
            "model": "gpt-4.1",
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.2,
        }
        
        # API request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Send the request
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        # Record the request with rate limiter
        rate_limiter.add_request(estimated_token_count)
        
        # Process the response
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            usage = response.json().get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            
            # Update statistics
            stats.add_result(True, total_tokens)
            
            # Return the raw response without extracting
            print(f"✓ Successfully evaluated ({total_tokens} tokens)")
            return {"raw_response": result}
            
        else:
            error_message = response.text
            print(f"✗ Request failed: {response.status_code}, Error: {error_message}")
            
            # Update statistics
            stats.add_result(False)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Waiting {retry_after} seconds")
                time.sleep(retry_after)
            
            return None
    
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        traceback.print_exc()
        stats.add_result(False)
        return None

def load_jsonl_data(jsonl_file):
    """Load JSONL data and extract the model answers for evaluation"""
    data = []
    total_count = 0
    model_answers_count = 0
    model_types = set()
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    record_id = record.get("id", "")
                    image_path = record.get("image", "")
                    
                    # For new datasets, id might not be present
                    if not record_id and image_path:
                        # Use image path as identifier
                        record_id = image_path.split("/")[-1].split(".")[0]
                        record["id"] = record_id
                    
                    total_count += 1
                    
                    # Find all vqa_pairs_with_* keys
                    model_keys = [k for k in record.keys() if k.startswith("vqa_pairs_with_")]
                    
                    if model_keys:
                        model_answers_count += 1
                        
                        for model_key in model_keys:
                            model_name = model_key.replace("vqa_pairs_with_", "")
                            model_types.add(model_name)
                            
                            # Process each question-answer pair
                            for qa_pair in record.get(model_key, []):
                                # Create an item with all necessary information
                                item = {
                                    "record_id": record_id,
                                    "model_name": model_name,
                                    "question_id": qa_pair.get("question_id", ""),
                                    "question_type": qa_pair.get("question_type", ""),
                                    "question": qa_pair.get("question", ""),
                                    "answer": qa_pair.get("answer", ""),
                                    "model_answer": qa_pair.get(f"{model_name}_answer", ""),
                                    "image": image_path
                                }
                                
                                # Make sure the model answer exists
                                if not item["model_answer"] and model_name in qa_pair:
                                    item["model_answer"] = qa_pair.get(model_name, "")
                                
                                if item["model_answer"]:
                                    data.append(item)
                    
                except json.JSONDecodeError:
                    continue
    
    print(f"Dataset loading summary:")
    print(f"- Total records: {total_count}")
    print(f"- Records with model answers: {model_answers_count}")
    print(f"- Models found: {', '.join(model_types)}")
    print(f"- Total QA pairs to evaluate: {len(data)}")
    return data

def save_checkpoint(checkpoint_file, processed_ids):
    """Save a checkpoint with processed item IDs"""
    with checkpoint_lock:
        # Update global tracking variable for signal handler
        current_processed_ids.update(processed_ids)
        
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "processed_ids": list(processed_ids),
                "timestamp": time.time(),
                "interrupted": False
            }, f)
    print(f"Checkpoint saved: {len(processed_ids)} processed items")

def load_checkpoint(checkpoint_file):
    """Load a checkpoint with processed item IDs"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            processed_ids = set(data.get("processed_ids", []))
            timestamp = data.get("timestamp", 0)
            interrupted = data.get("interrupted", False)
            age_hours = (time.time() - timestamp) / 3600
            
            # Update global tracking variable for signal handler
            with checkpoint_lock:
                current_processed_ids.update(processed_ids)
            
            if interrupted:
                print(f"Detected previous program interruption. Resuming from checkpoint...")
            
            print(f"Loaded checkpoint: {len(processed_ids)} processed items (Age: {age_hours:.1f} hours)")
            return processed_ids
    return set()

def process_single_item(item, output_file, result_queue, api_key=DEFAULT_API_KEY):
    """Process a single QA pair to evaluate the model answer"""
    try:
        item_id = f"{item['record_id']}_{item['model_name']}_{item['question_id']}"
        
        # Check if this item is already being processed by another thread
        with processed_lock:
            if item_id in currently_processing:
                print(f"Skipping duplicate task: {item_id}")
                return {"id": item_id, "status": "skipped_duplicate"}
            currently_processing.add(item_id)
        
        print(f"\n-------- Processing item: {item_id} --------")
        
        try:
            # Get evaluation from gpt-4.1
            evaluation = evaluate_answer_with_gpt(item, api_key)
            
            if evaluation is not None:
                # Create result item with evaluation
                result = item.copy()
                
                # Add the raw response directly
                if "raw_response" in evaluation:
                    result["evaluation"] = evaluation["raw_response"]
                
                # Write to output file using a thread-safe method
                result_queue.put(result)
                
                print(f"✓ Result queued: {item_id}")
                return {
                    "id": item_id,
                    "status": "success"
                }
            
            print(f"✗ Failed to evaluate item {item_id}")
            return {
                "id": item_id,
                "status": "failed"
            }
        finally:
            # Always remove from currently processing set, even if an error occurs
            with processed_lock:
                currently_processing.discard(item_id)
    
    except Exception as e:
        print(f"✗ Error processing item: {e}")
        traceback.print_exc()
        # Make sure to remove from currently processing if there's an exception
        with processed_lock:
            currently_processing.discard(item_id)
        return {
            "id": f"{item.get('record_id', 'unknown')}_{item.get('model_name', 'unknown')}_{item.get('question_id', 'unknown')}",
            "status": "error",
            "error": str(e)
        }

def writer_thread(output_file, result_queue, stop_event, checkpoint_file=None):
    """Thread to handle writing results to file in real-time"""
    count = 0
    last_report_time = time.time()
    last_checkpoint_time = time.time()
    last_write_time = time.time()
    checkpoint_interval = 60  # Save checkpoint every 60 seconds
    write_interval = 30  # Write to file every 30 seconds
    
    print(f"Writer thread started, results will be written to {output_file}")
    
    # Dictionary to store results by image
    results_by_image = {}
    
    # Create a set to track processed item IDs for checkpoint
    processed_ids = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        processed_ids = load_checkpoint(checkpoint_file)
    
    # Update global tracking variable for signal handler
    with checkpoint_lock:
        current_processed_ids.update(processed_ids)
    
    # Load existing results if available
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        rec_id = record.get("id", "")
                        if rec_id:
                            results_by_image[rec_id] = record
                    except json.JSONDecodeError:
                        continue
        print(f"Loaded {len(results_by_image)} existing records from output file")
    else:
        # Create output file if it doesn't exist
        with open(output_file, 'w') as f:
            # Initialize empty file
            pass
    
    def write_results_to_file():
        """Helper function to write current results to file"""
        try:
            records_to_write = list(results_by_image.values())
            
            # Create a temporary file
            temp_file = f"{output_file}.temp"
            with open(temp_file, 'w') as f:
                for record in records_to_write:
                    f.write(json.dumps(record) + '\n')
            
            # Replace the original file with the temporary file
            os.replace(temp_file, output_file)
            
            print(f"Wrote {len(records_to_write)} records to output file")
            return True
        except Exception as e:
            print(f"Error writing results to file: {e}")
            traceback.print_exc()
            return False
    
    while not stop_event.is_set() or not result_queue.empty():
        try:
            # Get result from queue, wait up to 1 second
            try:
                result = result_queue.get(timeout=1)
            except queue.Empty:
                # If queue is empty and we need to report progress, do it
                if time.time() - last_report_time > 10:  # Report every 10 seconds
                    stats_data = stats.get_stats()
                    print(f"\nProgress: {stats_data['processed']} items, "
                          f"{stats_data['requests_per_minute']:.1f} rpm, "
                          f"{stats_data['tokens_per_minute']:.1f} tpm")
                    last_report_time = time.time()
                
                # Save checkpoint periodically
                if checkpoint_file and time.time() - last_checkpoint_time > checkpoint_interval:
                    save_checkpoint(checkpoint_file, processed_ids)
                    last_checkpoint_time = time.time()
                
                # Write results to file periodically
                if time.time() - last_write_time > write_interval:
                    if write_results_to_file():
                        last_write_time = time.time()
                
                continue
            
            # Get record and model information
            record_id = result.get("record_id", "unknown")
            model_name = result.get("model_name", "unknown")
            question_id = result.get("question_id", "unknown")
            
            # Create a unique ID for this evaluation
            item_id = f"{record_id}_{model_name}_{question_id}"
            
            # Organize results by image
            if record_id not in results_by_image:
                results_by_image[record_id] = {
                    "id": record_id,
                    "image": result.get("image", ""),
                    "questions": []
                }
            
            # Extract evaluation result
            evaluation = result.get("evaluation", "")
            
            # Create simplified question entry
            question_entry = {
                "question_id": question_id,
                "question_type": result.get("question_type", ""),
                "question": result.get("question", ""),
                "answer": result.get("answer", ""),
                "model_answer": result.get("model_answer", ""),
                "model": model_name,
                "judgment": evaluation
            }
            
            # Check if this question already exists
            existing_questions = results_by_image[record_id]["questions"]
            existing_idx = None
            for i, q in enumerate(existing_questions):
                if q["question_id"] == question_id and q["model"] == model_name:
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                # Update existing question
                existing_questions[existing_idx] = question_entry
            else:
                # Add as new question
                results_by_image[record_id]["questions"].append(question_entry)
            
            # Add to processed IDs for checkpoint
            processed_ids.add(item_id)
            
            # Update global tracking variable for signal handler
            with checkpoint_lock:
                current_processed_ids.add(item_id)
            
            count += 1
            result_queue.task_done()
            
            # Periodically report progress
            if count % 10 == 0 or time.time() - last_report_time > 10:
                stats_data = stats.get_stats()
                print(f"\nProgress: {count} processed, "
                      f"{stats_data['requests_per_minute']:.1f} rpm, "
                      f"{stats_data['tokens_per_minute']:.1f} tpm")
                last_report_time = time.time()
            
            # Save checkpoint periodically
            if checkpoint_file and time.time() - last_checkpoint_time > checkpoint_interval:
                save_checkpoint(checkpoint_file, processed_ids)
                last_checkpoint_time = time.time()
            
            # Write results to file periodically
            if time.time() - last_write_time > write_interval:
                if write_results_to_file():
                    last_write_time = time.time()
        
        except Exception as e:
            print(f"Writer thread error: {e}")
            traceback.print_exc()
    
    # Final checkpoint save
    if checkpoint_file:
        save_checkpoint(checkpoint_file, processed_ids)
    
    # Final write to output file
    write_results_to_file()
    
    print(f"All processing complete. Processed {count} results")

def process_dataset(input_file, output_file, num_threads=20, max_items=None, api_key=DEFAULT_API_KEY, checkpoint_file=None):
    """Process the dataset and evaluate model answers"""
    print("\n" + "="*50)
    print(f" Starting VQA Performance Evaluation ")
    print("="*50)
    
    # Verify API key is provided
    if not api_key:
        print("❌ Error: No API key provided. Set the OPENAI_API_KEY environment variable or use --api-key.")
        return
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")
    
    # Load checkpoint if available
    processed_ids = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        processed_ids = load_checkpoint(checkpoint_file)
        print(f"Will skip {len(processed_ids)} already processed items")
        
        # Update global tracking variable for signal handler
        with checkpoint_lock:
            current_processed_ids.update(processed_ids)
    
    # Load data with model answers
    data = load_jsonl_data(input_file)
    
    if max_items and max_items > 0:
        data = data[:max_items]
        print(f"Limited to processing {max_items} items for this run")
    
    # Filter out already processed items if checkpoint exists
    if processed_ids:
        original_count = len(data)
        data = [item for item in data if f"{item['record_id']}_{item['model_name']}_{item['question_id']}" not in processed_ids]
        filtered_count = original_count - len(data)
        print(f"Filtered out {filtered_count} already processed items, {len(data)} remaining")
    
    if not data:
        print("No items to process. Exiting.")
        return
    
    # Create a queue for results
    result_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Start writer thread with checkpoint
    writer = threading.Thread(
        target=writer_thread, 
        args=(output_file, result_queue, stop_event, checkpoint_file)
    )
    writer.daemon = True
    writer.start()
    
    # Calculate optimal number of threads - never use more threads than items
    optimal_threads = min(num_threads, len(data))
    print(f"\nStarting to process {len(data)} QA pairs using {optimal_threads} threads...")
    print("-"*50)
    
    # Create a list to ensure each item is processed only once
    items_to_process = list(data)  # Make a copy to avoid modifying the original list
    
    # Process items with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
        # Create list of futures
        futures = []
        
        # Submit each item only once
        for item in items_to_process:
            # Skip if already processed (double check)
            item_id = f"{item['record_id']}_{item['model_name']}_{item['question_id']}"
            if item_id in processed_ids:
                continue
                
            # Create a future for this item
            future = executor.submit(process_single_item, item, output_file, result_queue, api_key)
            futures.append(future)
        
        # Create a progress bar
        with tqdm(total=len(futures), desc="Processing items") as pbar:
            # Process results as they complete
            for future in futures:
                try:
                    result = future.result()
                    # Update progress bar
                    pbar.update(1)
                except Exception as e:
                    print(f"Task error: {e}")
                    traceback.print_exc()
    
    # Signal writer thread to stop and wait for it
    print("All processing tasks completed, waiting for writer thread to finish...")
    stop_event.set()
    writer.join()
    
    # Get final statistics
    final_stats = stats.get_stats()
    
    print("\n" + "="*50)
    print(" Processing Summary ")
    print("="*50)
    print(f"Successfully processed: {final_stats['success']} items")
    print(f"Failed: {final_stats['failure']} items")
    print(f"Total tokens used: {final_stats['total_tokens']}")
    print(f"Average tokens per item: {final_stats['total_tokens'] / final_stats['processed'] if final_stats['processed'] > 0 else 0:.1f}")
    print(f"Average processing rate: {final_stats['requests_per_minute']:.1f} requests/minute")
    print(f"Average token usage rate: {final_stats['tokens_per_minute']:.1f} tokens/minute")
    print(f"Total processing time: {final_stats['elapsed_seconds'] / 60:.1f} minutes")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA model performance using g p t")
    parser.add_argument("--input", default='保存/blip2_VQA_test_results.jsonl', 
                        help="Input JSONL file path with model answers")
    parser.add_argument("--output", default="保存/json/eval/blip2_VQA_test_results_GPT4.1Evaled.jsonl", 
                        help="Output JSONL file path for evaluation results")
    parser.add_argument("--threads", type=int, default=10, 
                        help="Number of processing threads to use")
    parser.add_argument("--max-items", type=int, default=None, 
                        help="Maximum number of items to process (for testing)")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, 
                        help="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.")
    parser.add_argument("--checkpoint", default="./vqa_evaluation_checkpoint.json", 
                        help="Checkpoint file path for resuming")
    parser.add_argument("--no-checkpoint", action="store_true", 
                        help="Disable checkpoint functionality")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last interruption (using checkpoint)")
    args = parser.parse_args()
    
    # Get API key from environment variable if not provided as argument
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if args.resume and args.no_checkpoint:
        print("Error: Cannot use both --resume and --no-checkpoint options")
        return
    
    if args.resume:
        # Check if checkpoint file exists
        if not os.path.exists(args.checkpoint):
            print(f"Error: Could not find checkpoint file {args.checkpoint}")
            return
        print(f"Resuming from previous interruption...")
    
    print(f"Starting with parameters:")
    print(f"- Input file: {args.input}")
    print(f"- Output file: {args.output}")
    print(f"- Threads: {args.threads}")
    print(f"- Max items: {args.max_items if args.max_items else 'unlimited'}")
    print(f"- API key: {'set' if api_key else 'not set'}")
    print(f"- Checkpoint: {'disabled' if args.no_checkpoint else args.checkpoint}")
    print(f"- Resume mode: {'enabled' if args.resume else 'disabled'}")
    
    # Process dataset with checkpoint if enabled
    checkpoint_file = None if args.no_checkpoint else args.checkpoint
    process_dataset(args.input, args.output, args.threads, args.max_items, api_key, checkpoint_file)
    
    # Remove interrupted flag from checkpoint after successful completion
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            data["interrupted"] = False
            with open(checkpoint_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error updating checkpoint status: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # This should be handled by our signal handler
        pass
    except Exception as e:
        print(f"Program execution error: {e}")
        traceback.print_exc()
        
        # Try to save checkpoint on unhandled exceptions too
        if current_processed_ids:
            checkpoint_file = "./vqa_evaluation_checkpoint.json"
            for i, arg in enumerate(sys.argv):
                if arg == "--checkpoint" and i + 1 < len(sys.argv):
                    checkpoint_file = sys.argv[i + 1]
                    break
                elif arg.startswith("--checkpoint="):
                    checkpoint_file = arg.split("=", 1)[1]
                    break
            
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        "processed_ids": list(current_processed_ids),
                        "timestamp": time.time(),
                        "interrupted": True
                    }, f)
                print(f"Saved progress of {len(current_processed_ids)} records to {checkpoint_file}")
            except Exception as save_err:
                print(f"Error saving checkpoint: {save_err}")
        
        sys.exit(1) 