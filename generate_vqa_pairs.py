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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# API key from environment variable or default to empty string
DEFAULT_API_KEY = ''

# API Rate limits for GPT-4.1-mini
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
    print("\nThe program received an interrupt signal and is saving checkpoints...")
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
        checkpoint_file = "./vqa_pairs_checkpoint.json"
    
    # Save checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "processed_ids": list(processed_ids_to_save),
            "timestamp": time.time(),
            "interrupted": True
        }, f)
    
    print(f"saved {len(processed_ids_to_save)}  progress of a record to {checkpoint_file}")
    print("You can restart the program with the same command and it will continue from the point of interruption.")
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
def generate_vqa_pairs_with_gpt(record, api_key):
    """Generate VQA question-answer pairs using GPT-4.1-mini API"""
    try:
        # Verify API key is provided
        if not api_key:
            print("❌ Error: API key not provided. Please set the OPENAI_API_KEY environment variable or use the --api-key parameter.")
            return None
        
        # Extract data from the record
        answer = record.get("answer", "")
        relations = record.get("relations", [])
        tags = record.get("tags", [])
        
        # Convert tags from string to list if necessary
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",")]
        
        # Create the optimized prompt
        prompt = (
            f"Given the following image description, spatial relationships, and object tags, create 5 different types of visual question-answer (VQA) pairs.\n\n"
            f"Image Description:\n{answer}\n\n"
            f"Spatial Relationships:\n"
        )
        
        # Add relations to prompt
        for i, rel in enumerate(relations):
            object1 = rel.get("object1", "")
            relation = rel.get("relation", "")
            object2 = rel.get("object2", "")
            prompt += f"{i+1}. {object1} {relation} {object2}\n"
        
        prompt += f"\nObject Tags:\n"
        for i, tag in enumerate(tags):
            prompt += f"{i+1}. {tag}\n"
        
        prompt += (
            f"\nPlease create 9 different types of VQA pairs, including:\n"
            f"1. At least 3 questions about spatial relationships (e.g., 'Where is the highway located?')\n"
            f"2. At least 2 question about objects in the image (e.g., 'What buildings are visible in the image?')\n"
            f"3. At least 2 question about overall image features (e.g., 'What type of area is shown?')\n"
            f"4. At least 2 question about object quantities or proportions (e.g., 'Is there more residential or industrial area?')\n\n"
            f"IMPORTANT: Only ask questions that can be answered based on the provided description, relationships, and tags. Do not make up information not present in the input.\n\n"
            f"Return in JSON format with question_id, question_type, question, and answer fields:\n"
            f"```json\n"
            f"[\n"
            f"  {{\n"
            f"    \"question_id\": \"q1\",\n"
            f"    \"question_type\": \"spatial\",\n"
            f"    \"question\": \"question text\",\n"
            f"    \"answer\": \"answer text\"\n"
            f"  }},\n"
            f"  ...\n"
            f"]\n"
            f"```"
        )
        
        # Estimate token count for this request
        estimated_token_count = len(prompt.split()) * 1.3 + 500  # Rough estimate
        
        # Wait if approaching rate limits
        wait_time = rate_limiter.wait_if_needed(estimated_token_count)
        if wait_time > 0:
            print(f"Approaching rate limit. Wait. {wait_time:.2f} 秒")
            time.sleep(wait_time)
        
        # Add small random delay to prevent burst
        time.sleep(random.uniform(0.0, 0.1))
        
        # Prepare the request
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in creating visual question-answer pairs. You create diverse, natural questions that are strictly based on the given image description, spatial relationships, and tags. Only generate questions that can be factually answered from the provided information, without inventing details not present in the input."},
            {"role": "user", "content": prompt}
        ]
        
        # API request parameters
        payload = {
            "model": "gpt-4.1",
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.7,
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
            
            # Extract questions from the response
            vqa_pairs = []
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'```json(.*?)```', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    vqa_pairs = json.loads(json_str)
                else:
                    json_match = re.search(r'(\[.*\])', result, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                        vqa_pairs = json.loads(json_str)
                    else:
                        # Try to find any JSON array in the response
                        json_match = re.search(r'\[\s*{.*}\s*\]', result, re.DOTALL)
                        if json_match:
                            vqa_pairs = json.loads(json_match.group(0))
                
                if vqa_pairs:
                    print(f"✓ Successfully generated {len(vqa_pairs)}  VQA questions have a significant impact on ({total_tokens} tokens)")
                else:
                    print(f"✗ Unable to parse JSON: {result[:200]}...")
                    # Try last resort parsing
                    if "question_id" in result and "question" in result:
                        print("Try alternative parsing methods...")
                        parts = result.split("\n")
                        current_qa = {}
                        for part in parts:
                            if "question_id" in part:
                                if current_qa and "question_id" in current_qa:
                                    vqa_pairs.append(current_qa)
                                    current_qa = {}
                                match = re.search(r'"question_id":\s*"([^"]+)"', part)
                                if match:
                                    current_qa["question_id"] = match.group(1)
                            if "question_type" in part:
                                match = re.search(r'"question_type":\s*"([^"]+)"', part)
                                if match:
                                    current_qa["question_type"] = match.group(1)
                            if "question" in part and "question_id" not in part and "question_type" not in part:
                                match = re.search(r'"question":\s*"([^"]+)"', part)
                                if match:
                                    current_qa["question"] = match.group(1)
                            if "answer" in part:
                                match = re.search(r'"answer":\s*"([^"]+)"', part)
                                if match:
                                    current_qa["answer"] = match.group(1)
                        if current_qa and "question_id" in current_qa:
                            vqa_pairs.append(current_qa)
                        
                        if vqa_pairs:
                            print(f"✓ Alternative parsing succeeds, generating {len(vqa_pairs)} The VQA questions have a significant impact on")
            except json.JSONDecodeError as je:
                print(f"✗ JSON parsing failure: {je}")
                print(f"Response content: {result[:200]}...")
                vqa_pairs = []
            
            return vqa_pairs
        else:
            error_message = response.text
            print(f"✗ Request Failed: {response.status_code}, incorrect: {error_message}")
            
            # Update statistics
            stats.add_result(False)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limiting. Waiting {retry_after} s")
                time.sleep(retry_after)
            
            return None
    
    except Exception as e:
        print(f"✗ Error generating VQA pairs: {e}")
        traceback.print_exc()
        stats.add_result(False)
        return None

def load_jsonl_data(jsonl_file, mode="missing_vqa"):
    """Load JSONL data based on the specified mode
    
    mode options:
    - "missing_vqa": only load records without 'vqa_pairs' field
    - "all": load all records
    """
    data = []
    total_count = 0
    missing_vqa_count = 0
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    record_id = record.get("id", "")
                    
                    # For new datasets, id might not be present
                    if not record_id:
                        record_id = str(total_count)
                        record["id"] = record_id
                    
                    total_count += 1
                    
                    # Handle based on mode
                    if mode == "missing_vqa":
                        has_vqa = "vqa_pairs" in record and isinstance(record["vqa_pairs"], list)
                        
                        # Check if vqa_pairs are missing
                        if not has_vqa:
                            missing_vqa_count += 1
                            data.append(record)
                    elif mode == "all":
                        data.append(record)
                    
                except json.JSONDecodeError:
                    continue
    
    print(f"Summary of dataset loading:")
    print(f"- Total number of records: {total_count}")
    if mode == "missing_vqa":
        print(f"- Missing records of VQA question pairs: {missing_vqa_count}")
    print(f"- Valid records to be processed: {len(data)}")
    return data

def load_existing_data(jsonl_file):
    """Load all data from existing JSONL file"""
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError:
                    continue
    return data

def save_checkpoint(checkpoint_file, processed_ids):
    """Save a checkpoint with processed record IDs"""
    with checkpoint_lock:
        # Update global tracking variable for signal handler
        current_processed_ids.update(processed_ids)
        
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "processed_ids": list(processed_ids),
                "timestamp": time.time(),
                "interrupted": False
            }, f)
    print(f"Checkpoint saved: {len(processed_ids)} Records processed")

def load_checkpoint(checkpoint_file):
    """Load a checkpoint with processed record IDs"""
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
                print(f"Detects that the last program was interrupted. Resuming from the interruption point...")
            
            print(f"Loaded checkpoints: {len(processed_ids)} Records processed (before the present: {age_hours:.1f} hourly)")
            return processed_ids
    return set()

def process_single_record(record, output_file, result_queue, api_key=DEFAULT_API_KEY):
    """Process a single record to generate VQA pairs"""
    try:
        record_id = record.get("id", "")
        
        # Check if this record is already being processed by another thread
        with processed_lock:
            if record_id in currently_processing:
                print(f"Skip Repeat Tasks: {record_id}")
                return {"id": record_id, "status": "skipped_duplicate"}
            currently_processing.add(record_id)
        
        print(f"\n-------- Records being processed: {record_id} --------")
        
        try:
            # Get VQA pairs from GPT-4.1-mini
            vqa_pairs = generate_vqa_pairs_with_gpt(record, api_key)
            
            if vqa_pairs is not None and len(vqa_pairs) > 0:
                # Create result record (copy original record and add vqa_pairs)
                result = record.copy()
                result["vqa_pairs"] = vqa_pairs
                
                # Write to output file using a thread-safe method
                result_queue.put(result)
                
                print(f"✓ The result has been added to the queue: {record_id}")
                return {
                    "id": record_id,
                    "status": "success"
                }
            
            print(f"✗ Failure to record {record_id} Generate VQA question pairs")
            return {
                "id": record_id,
                "status": "failed"
            }
        finally:
            # Always remove from currently processing set, even if an error occurs
            with processed_lock:
                currently_processing.discard(record_id)
    
    except Exception as e:
        print(f"✗ Error while processing records: {e}")
        traceback.print_exc()
        # Make sure to remove from currently processing if there's an exception
        with processed_lock:
            currently_processing.discard(record_id)
        return {
            "id": record.get("id", "unknown"),
            "status": "error",
            "error": str(e)
        }

def writer_thread(output_file, result_queue, stop_event, existing_data, checkpoint_file=None):
    """Thread to handle writing results to file in real-time"""
    count = 0
    last_report_time = time.time()
    last_checkpoint_time = time.time()
    checkpoint_interval = 60  # Save checkpoint every 60 seconds
    
    print(f"The write thread has been started and will update the results to the {output_file}")
    
    # Create a dictionary of existing data for quick lookup
    existing_data_dict = {item.get("id", i): i for i, item in enumerate(existing_data)}
    
    # Create a set to track which record IDs have already been written to the output file
    written_record_ids = set()
    
    # If the output file exists, load the IDs of records that are already in it
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        written_record_ids.add(record.get("id", ""))
                    except json.JSONDecodeError:
                        continue
    else:
        # Create output file if it doesn't exist
        with open(output_file, 'w') as f:
            # Initialize empty file
            pass
    
    # Track processed IDs for checkpoint
    processed_ids = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        processed_ids = load_checkpoint(checkpoint_file)
    
    # Update global tracking variable for signal handler
    with checkpoint_lock:
        current_processed_ids.update(processed_ids)
    
    while not stop_event.is_set() or not result_queue.empty():
        try:
            # Get result from queue, wait up to 1 second
            try:
                result = result_queue.get(timeout=1)
            except queue.Empty:
                # If queue is empty and we need to report progress, do it
                if time.time() - last_report_time > 10:  # Report every 10 seconds
                    stats_data = stats.get_stats()
                    print(f"\nProgress: {stats_data['processed']} entry, "
                          f"{stats_data['requests_per_minute']:.1f} rpm, "
                          f"{stats_data['tokens_per_minute']:.1f} tpm")
                    last_report_time = time.time()
                
                # Save checkpoint periodically
                if checkpoint_file and time.time() - last_checkpoint_time > checkpoint_interval:
                    save_checkpoint(checkpoint_file, processed_ids)
                    last_checkpoint_time = time.time()
                
                continue
            
            # Get record ID
            record_id = result.get("id", "")
            
            # Update existing data dictionary
            if record_id in existing_data_dict:
                idx = existing_data_dict[record_id]
                existing_data[idx] = result
            else:
                existing_data.append(result)
                existing_data_dict[record_id] = len(existing_data) - 1
            
            # Write the result to file - either append or update
            if record_id in written_record_ids:
                # This record already exists in the file, need to update it
                # This is more complex as we need to rewrite the file
                # We'll defer updates to a batch process later to avoid constant file rewrites
                pass
            else:
                # New record, just append to the file
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
                written_record_ids.add(record_id)
            
            # Add to processed IDs for checkpoint
            processed_ids.add(record_id)
            
            # Update global tracking variable for signal handler
            with checkpoint_lock:
                current_processed_ids.add(record_id)
            
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
                
                # Every checkpoint interval, also do any pending file updates
                # for records that needed to be updated rather than appended
                rewrite_needed = False
                current_records = []
                
                with open(output_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                record_id = record.get("id", "")
                                # If this record has been updated in existing_data, use the new version
                                if record_id in existing_data_dict:
                                    idx = existing_data_dict[record_id]
                                    updated_record = existing_data[idx]
                                    if updated_record != record:
                                        rewrite_needed = True
                                        current_records.append(updated_record)
                                    else:
                                        current_records.append(record)
                                else:
                                    current_records.append(record)
                            except json.JSONDecodeError:
                                continue
                
                if rewrite_needed:
                    # Write all records back to file
                    temp_file = f"{output_file}.temp"
                    with open(temp_file, 'w') as f:
                        for rec in current_records:
                            f.write(json.dumps(rec) + '\n')
                    
                    # Replace original file with updated one
                    os.replace(temp_file, output_file)
                    print(f"Output files have been updated with the latest changes")
        
        except Exception as e:
            print(f"Error in write thread: {e}")
            traceback.print_exc()
    
    # Final checkpoint save
    if checkpoint_file:
        save_checkpoint(checkpoint_file, processed_ids)
    
    # Final file update to ensure all changes are written
    current_records = []
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    record_id = record.get("id", "")
                    # If this record has been updated in existing_data, use the new version
                    if record_id in existing_data_dict:
                        idx = existing_data_dict[record_id]
                        current_records.append(existing_data[idx])
                    else:
                        current_records.append(record)
                except json.JSONDecodeError:
                    continue
    
    # Write all records back to file
    temp_file = f"{output_file}.temp"
    with open(temp_file, 'w') as f:
        for rec in current_records:
            f.write(json.dumps(rec) + '\n')
    
    # Replace original file with updated one
    os.replace(temp_file, output_file)
    
    print(f"All processing completed. Total processing {count} Results")

def process_dataset(input_file, output_file, num_threads=20, max_records=None, mode="missing_vqa", api_key=DEFAULT_API_KEY, checkpoint_file=None):
    """Process the dataset, focusing on records with missing VQA pairs or processing all
    
    mode options:
    - "missing_vqa": process only records missing vqa_pairs
    - "all": process all records
    """
    print("\n" + "="*50)
    if mode == "missing_vqa":
        print(f" Start VQA question pair generation - Missing question pair fixing ")
    else:
        print(f" Start VQA Question Pair Generation - Full Dataset ")
    print("="*50)
    
    # Verify API key is provided
    if not api_key:
        print("❌ Error: API key not provided. Please set the OPENAI_API_KEY environment variable or use the --api-key parameter.")
        return
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured that the output directory exists: {output_dir}")
    
    # Load checkpoint if available
    processed_ids = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        processed_ids = load_checkpoint(checkpoint_file)
        print(f"will skip {len(processed_ids)} Records processed")
        
        # Update global tracking variable for signal handler
        with checkpoint_lock:
            current_processed_ids.update(processed_ids)
    
    # Load existing data for updating
    existing_data = []
    if os.path.exists(output_file):
        existing_data = load_existing_data(output_file)
        print(f"From {output_file} It's loaded. {len(existing_data)} Existing records")
    
    # Load data based on mode
    data = load_jsonl_data(input_file, mode)
    
    if max_records and max_records > 0:
        data = data[:max_records]
        print(f"Restricted processing for this run {max_records} entry")
    
    # Filter out already processed records if checkpoint exists
    if processed_ids:
        original_count = len(data)
        data = [record for record in data if record.get("id", "") not in processed_ids]
        filtered_count = original_count - len(data)
        print(f"filtered {filtered_count} Records processed, remaining {len(data)} Articles to be processed")
    
    if not data:
        if mode == "missing_vqa":
            print("There are no missing VQA question pairs of records to process. Exit.")
        else:
            print("没有记录需要处理。退出。")
        return
    
    # Create a queue for results
    result_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Start writer thread with checkpoint
    writer = threading.Thread(
        target=writer_thread, 
        args=(output_file, result_queue, stop_event, existing_data, checkpoint_file)
    )
    writer.daemon = True
    writer.start()
    
    # Calculate optimal number of threads - never use more threads than records
    optimal_threads = min(num_threads, len(data))
    if mode == "missing_vqa":
        print(f"\nStart using {optimal_threads} threading {len(data)} Records of missing VQA question pairs...")
    else:
        print(f"\nStart using {optimal_threads} threading {len(data)} entry...")
    print("-"*50)
    
    # Create a list to ensure each record is processed only once
    records_to_process = list(data)  # Make a copy to avoid modifying the original list
    
    # Process records with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
        # Create list of futures
        futures = []
        
        # Submit each record only once
        for record in records_to_process:
            # Skip if already processed (double check)
            if record.get("id", "") in processed_ids:
                continue
                
            # Create a future for this record
            future = executor.submit(process_single_record, record, output_file, result_queue, api_key)
            futures.append(future)
        
        # Create a progress bar
        with tqdm(total=len(futures), desc="Processing records") as pbar:
            # Process results as they complete
            for future in futures:
                try:
                    result = future.result()
                    # Update progress bar
                    pbar.update(1)
                except Exception as e:
                    print(f"Something is wrong with the mission.: {e}")
                    traceback.print_exc()
    
    # Signal writer thread to stop and wait for it
    print("All processing tasks have been completed, waiting for the write thread to finish...")
    stop_event.set()
    writer.join()
    
    # Get final statistics
    final_stats = stats.get_stats()
    
    print("\n" + "="*50)
    print(" Processing summary ")
    print("="*50)
    print(f"success story: {final_stats['success']} entry")
    print(f"success story: {final_stats['failure']} entry")
    print(f"Total number of tokens used: {final_stats['total_tokens']}")
    print(f"Average number of tokens per record: {final_stats['total_tokens'] / final_stats['processed'] if final_stats['processed'] > 0 else 0:.1f}")
    print(f"Average processing rate: {final_stats['requests_per_minute']:.1f} Requests/minute")
    print(f"Average Token Utilization Rate: {final_stats['tokens_per_minute']:.1f} Tokens/minute")
    print(f"Total processing time: {final_stats['elapsed_seconds'] / 60:.1f} minutes")
    print("="*50)
    
    # Count records still missing VQA pairs
    if mode == "missing_vqa":
        missing_count = 0
        with open(output_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                if "vqa_pairs" not in record or not record["vqa_pairs"]:
                    missing_count += 1
        
        if missing_count > 0:
            print(f"⚠️ warnings: {missing_count} Records still missing VQA questions on!")
        else:
            print("✅ successes: All records now have VQA questions on the!")

def main():
    parser = argparse.ArgumentParser(description="使用GPT-4.1")
    parser.add_argument("--input", default="/mnt/d/Storage/OneDrive/UTS/PhD/Paper/ACM-MM/RSVL-QA/spatial_relations_with_relations.jsonl", 
                        help="Input JSONL file path")
    parser.add_argument("--output", default="./vqa_dataset.jsonl", 
                        help="Output JSONL file path")
    parser.add_argument("--threads", type=int, default=10, 
                        help="Number of processing threads to use")
    parser.add_argument("--max-records", type=int, default=None, 
                        help="Maximum number of records to be processed (for testing)")
    parser.add_argument("--mode", choices=["missing_vqa", "all"], default="missing_vqa", 
                        help="operating mode: 'missing_vqa'Fix the missing VQA issue on, 'all'Processing of all data")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, 
                        help="OpenAI API key. It can also be set via the OPENAI_API_KEY environment variable.")
    parser.add_argument("--test", action="store_true", 
                        help="Run a single test example")
    parser.add_argument("--checkpoint", default="./vqa_pairs_checkpoint.json", 
                        help="Checkpoint file paths for breakpoint transfers")
    parser.add_argument("--no-checkpoint", action="store_true", 
                        help="Not using the checkpoint function")
    parser.add_argument("--resume", action="store_true",
                       help="Continuation from last break (using checkpoints)")
    args = parser.parse_args()
    
    # Get API key from environment variable if not provided as argument
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if args.test:
        # Run a test on a single example
        test_record = {
            "id": "test",
            "image": "/path/to/image.jpg",
            "answer": "The landscape is characterized by a prominent institutional or educational campus in the center-left, identifiable by its multiple large buildings, sports fields, and running track. Surrounding this campus are dense residential neighborhoods with individual houses and tree-lined streets.",
            "relations": [
                {"object1": "educational campus", "relation": "in the center-left", "object2": "landscape"},
                {"object1": "residential neighborhoods", "relation": "surrounding", "object2": "campus"}
            ],
            "tags": ["urban area", "institutional buildings", "sports fields", "residential neighborhoods"]
        }
        print(f"\nTesting a single example:")
        print(f"Input Record ID: {test_record['id']}")
        vqa_pairs = generate_vqa_pairs_with_gpt(test_record, api_key)
        if vqa_pairs:
            print(f"The generated VQA questions are useful for:")
            for pair in vqa_pairs:
                print(f"- [{pair.get('question_type', '')}] Q: {pair.get('question', '')}")
                print(f"  A: {pair.get('answer', '')}")
        else:
            print("Failure to generate VQA question pair")
        return
    
    if args.resume and args.no_checkpoint:
        print("Error: Cannot be used at the same time --resume and --no-checkpoint options (as in settings)")
        return
    
    if args.resume:
        # Check if checkpoint file exists
        if not os.path.exists(args.checkpoint):
            print(f"错误：找不到检查点文件 {args.checkpoint}")
            return
        print(f"正在从上次中断处继续执行...")
    
    print(f"Use the following parameters to start:")
    print(f"- input file: {args.input}")
    print(f"- output file: {args.output}")
    print(f"- Number of threads: {args.threads}")
    print(f"- Maximum number of records: {args.max_records if args.max_records else 'limitless'}")
    print(f"- mode: {args.mode}")
    print(f"- API key: {'configured' if api_key else 'as yet unsettled'}")
    print(f"- checkpoints: {'prohibit' if args.no_checkpoint else args.checkpoint}")
    print(f"- recovery mode: {'enable' if args.resume else 'prohibit'}")
    
    # Process dataset with checkpoint if enabled
    checkpoint_file = None if args.no_checkpoint else args.checkpoint
    process_dataset(args.input, args.output, args.threads, args.max_records, args.mode, api_key, checkpoint_file)
    
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
            checkpoint_file = "./vqa_pairs_checkpoint.json"
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
                print(f"saved {len(current_processed_ids)} The progress of a record to {checkpoint_file}")
            except Exception as save_err:
                print(f"Error saving checkpoint: {save_err}")
        
        sys.exit(1) 