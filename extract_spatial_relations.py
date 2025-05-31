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

# API Rate limits for gpt-4.1
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
        checkpoint_file = "./spatial_relations_checkpoint.json"
    
    # Save checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "processed_ids": list(processed_ids_to_save),
            "timestamp": time.time(),
            "interrupted": True
        }, f)
    
    print(f"Saved progress of {len(processed_ids_to_save)} records to {checkpoint_file}")
    print("You can restart the program with the same command, and it will continue from where it was interrupted.")
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
def extract_relations_with_gpt(answer, api_key):
    """Extract spatial relations from text using gpt-4.1 API"""
    try:
        # Verify API key is provided
        if not api_key:
            print("❌ Error: No API key provided. Please set OPENAI_API_KEY environment variable or use --api-key parameter.")
            return None
        
        # Create the optimized prompt
        prompt = (
            f"Analyze the following text and extract the spatial relationships described in it. Each spatial relationship consists of three parts: object1, directional relationship, and object2. "
            f"For example, in the sentence 'The house is next to the lake', object1 is 'house', relation is 'next to', and object2 is 'lake'.\n\n"
            f"Text: \"{answer}\"\n\n"
            f"Return the extracted relationships in JSON format as follows:\n"
            f"<relations>\n"
            f"[\n"
            f"  {{\"object1\": \"object1\", \"relation\": \"relation\", \"object2\": \"object2\"}},\n"
            f"  ...\n"
            f"]\n"
            f"</relations>"
        )
        
        # Estimate token count for this request
        estimated_token_count = len(prompt.split()) * 1.3 + 200  # Rough estimate
        
        # Wait if approaching rate limits
        wait_time = rate_limiter.wait_if_needed(estimated_token_count)
        if wait_time > 0:
            print(f"Rate limit approaching, waiting for {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Add small random delay to prevent burst
        time.sleep(random.uniform(0.0, 0.1))
        
        # Prepare the request
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in extracting spatial relationships from text. You precisely identify spatial relationships between objects and output them in the specified format."},
            {"role": "user", "content": prompt}
        ]
        
        # API request parameters
        payload = {
            "model": "gpt-4.1",
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.0,
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
            
            # Extract relations from the response
            relations = []
            if "<relations>" in result and "</relations>" in result:
                relations_text = result.split("<relations>")[1].split("</relations>")[0].strip()
                try:
                    relations = json.loads(relations_text)
                    print(f"✓ Successfully extracted spatial relations: {len(relations)} relationships ({total_tokens} tokens)")
                except json.JSONDecodeError:
                    print(f"✗ JSON parsing failed for spatial relations: {relations_text}")
                    relations = []
            else:
                # Fallback: try to find any JSON array in the response
                try:
                    # Try to find a JSON array in the response
                    import re
                    json_match = re.search(r'\[\s*{.*}\s*\]', result, re.DOTALL)
                    if json_match:
                        relations = json.loads(json_match.group(0))
                        print(f"✓ Extracted {len(relations)} relationships using alternative method")
                    else:
                        print(f"✗ No relation tags found in response: {result[:100]}...")
                except Exception:
                    print(f"✗ Alternative extraction method failed")
            
            return relations
        else:
            error_message = response.text
            print(f"✗ Request failed: {response.status_code}, Error: {error_message}")
            
            # Update statistics
            stats.add_result(False)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Waiting for {retry_after} seconds")
                time.sleep(retry_after)
            
            return None
    
    except Exception as e:
        print(f"✗ Error extracting relations: {e}")
        traceback.print_exc()
        stats.add_result(False)
        return None

def load_jsonl_data(jsonl_file, mode="missing_relations"):
    """Load JSONL data based on the specified mode
    
    mode options:
    - "missing_relations": only load records without 'relations' field
    - "all": load all records
    """
    data = []
    total_count = 0
    missing_relations_count = 0
    
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
                    if mode == "missing_relations":
                        has_relations = "relations" in record and isinstance(record["relations"], list)
                        
                        # Check if relations are missing
                        if not has_relations:
                            missing_relations_count += 1
                            data.append(record)
                    elif mode == "all":
                        data.append(record)
                    
                except json.JSONDecodeError:
                    continue
    
    print(f"Dataset Loading Summary:")
    print(f"- Total records: {total_count}")
    if mode == "missing_relations":
        print(f"- Records missing spatial relations: {missing_relations_count}")
    print(f"- Valid records to process: {len(data)}")
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

def process_single_record(record, output_file, result_queue, api_key=DEFAULT_API_KEY):
    """Process a single record to extract spatial relations"""
    try:
        record_id = record.get("id", "")
        answer = record.get("answer", "")
        
        # Check if this record is already being processed by another thread
        with processed_lock:
            if record_id in currently_processing:
                print(f"Skipping duplicate task: {record_id}")
                return {"id": record_id, "status": "skipped_duplicate"}
            currently_processing.add(record_id)
        
        print(f"\n-------- Processing record: {record_id} --------")
        
        try:
            # Get the relations from g p t
            relations = extract_relations_with_gpt(answer, api_key)
            
            if relations is not None:
                # Create result record (copy original record and add relations)
                result = record.copy()
                result["relations"] = relations
                
                # Write to output file using a thread-safe method
                result_queue.put(result)
                
                print(f"✓ Result added to queue: {record_id}")
                return {
                    "id": record_id,
                    "status": "success"
                }
            
            print(f"✗ Failed to get spatial relations for record {record_id}")
            return {
                "id": record_id,
                "status": "failed"
            }
        finally:
            # Always remove from currently processing set, even if an error occurs
            with processed_lock:
                currently_processing.discard(record_id)
    
    except Exception as e:
        print(f"✗ Error processing record: {e}")
        traceback.print_exc()
        # Make sure to remove from currently processing if there's an exception
        with processed_lock:
            currently_processing.discard(record_id)
        return {
            "id": record.get("id", "unknown"),
            "status": "error",
            "error": str(e)
        }

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
    print(f"Checkpoint saved: {len(processed_ids)} processed records")

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
                print(f"Detected interruption from previous run. Resuming from the interruption point...")
            
            print(f"Loaded checkpoint: {len(processed_ids)} processed records (age: {age_hours:.1f} hours)")
            return processed_ids
    return set()

def writer_thread(output_file, result_queue, stop_event, existing_data, checkpoint_file=None):
    """Thread to handle writing results to file in real-time"""
    count = 0
    last_report_time = time.time()
    last_checkpoint_time = time.time()
    checkpoint_interval = 60  # Save checkpoint every 60 seconds
    
    print(f"Writer thread started, will update results in {output_file}")
    
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
                    print(f"\nProgress: {stats_data['processed']} records, "
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
                    print(f"Updated output file with recent changes")
        
        except Exception as e:
            print(f"Writer thread error: {e}")
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
    
    print(f"All processing complete. Total of {count} results processed")

def process_dataset(input_file, output_file, num_threads=20, max_records=None, mode="missing_relations", api_key=DEFAULT_API_KEY, checkpoint_file=None):
    """Process the dataset, focusing on records with missing relations or processing all
    
    mode options:
    - "missing_relations": process only records missing relations
    - "all": process all records
    """
    print("\n" + "="*50)
    if mode == "missing_relations":
        print(f" Starting Spatial Relation Extraction - Missing Relations Repair ")
    else:
        print(f" Starting Spatial Relation Extraction - Full Dataset ")
    print("="*50)
    
    # Verify API key is provided
    if not api_key:
        print("❌ Error: No API key provided. Please set OPENAI_API_KEY environment variable or use --api-key parameter.")
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
        print(f"Will skip {len(processed_ids)} already processed records")
        
        # Update global tracking variable for signal handler
        with checkpoint_lock:
            current_processed_ids.update(processed_ids)
    
    # Load existing data for updating
    existing_data = []
    if os.path.exists(output_file):
        existing_data = load_existing_data(output_file)
        print(f"Loaded {len(existing_data)} existing records from {output_file}")
    
    # Load data based on mode
    data = load_jsonl_data(input_file, mode)
    
    if max_records and max_records > 0:
        data = data[:max_records]
        print(f"Limited to processing {max_records} records for this run")
    
    # Filter out already processed records if checkpoint exists
    if processed_ids:
        original_count = len(data)
        data = [record for record in data if record.get("id", "") not in processed_ids]
        filtered_count = original_count - len(data)
        print(f"Filtered out {filtered_count} already processed records, {len(data)} remaining to process")
    
    if not data:
        if mode == "missing_relations":
            print("No records missing relations to process. Exiting.")
        else:
            print("No records to process. Exiting.")
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
    if mode == "missing_relations":
        print(f"\nStarting to process {len(data)} records missing relations using {optimal_threads} threads...")
    else:
        print(f"\nStarting to process {len(data)} records using {optimal_threads} threads...")
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
    print(f"Successfully processed: {final_stats['success']} records")
    print(f"Failed: {final_stats['failure']} records")
    print(f"Total tokens used: {final_stats['total_tokens']}")
    print(f"Average tokens per record: {final_stats['total_tokens'] / final_stats['processed'] if final_stats['processed'] > 0 else 0:.1f}")
    print(f"Average processing rate: {final_stats['requests_per_minute']:.1f} requests/minute")
    print(f"Average token usage rate: {final_stats['tokens_per_minute']:.1f} tokens/minute")
    print(f"Total processing time: {final_stats['elapsed_seconds'] / 60:.1f} minutes")
    print("="*50)
    
    # Count records still missing relations
    if mode == "missing_relations":
        missing_count = 0
        with open(output_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                if "relations" not in record or not record["relations"]:
                    missing_count += 1
        
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count} records still missing relations!")
        else:
            print("✅ Success: All records now have relations!")

def main():
    parser = argparse.ArgumentParser(description="Extract spatial relations using gpt-4.1")
    parser.add_argument("--input", default="/mnt/d/Storage/OneDrive/UTS/PhD/Paper/ACM-MM/RSVL-QA/image_query_answer_with_tags.jsonl", help="Input JSONL file path")
    parser.add_argument("--output", default="./spatial_relations_with_relations.jsonl", help="Output JSONL file path")
    parser.add_argument("--threads", type=int, default=10, help="Number of processing threads to use")
    parser.add_argument("--max-records", type=int, default=None, help="Maximum number of records to process (for testing)")
    parser.add_argument("--mode", choices=["missing_relations", "all"], default="missing_relations", 
                        help="Operation mode: 'missing_relations' to fix missing relations, 'all' to process all data")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, 
                        help="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.")
    parser.add_argument("--test", action="store_true", help="Run a single test example")
    parser.add_argument("--checkpoint", default="./spatial_relations_checkpoint.json", 
                        help="Checkpoint file path for resuming interrupted runs")
    parser.add_argument("--no-checkpoint", action="store_true", 
                        help="Disable checkpoint functionality")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from the last interruption point (using checkpoint)")
    args = parser.parse_args()
    
    if args.test:
        # Run a test on a single example
        test_answer = "The highway is located above the residential area. The school is to the northeast of the parking lot. Trees are scattered to the right of the houses."
        print(f"\nTesting a single example:")
        print(f"Input text: {test_answer}")
        relations = extract_relations_with_gpt(test_answer, args.api_key)
        if relations:
            print(f"Extracted relations:")
            for rel in relations:
                print(f"- {rel.get('object1', '')} {rel.get('relation', '')} {rel.get('object2', '')}")
        else:
            print("Failed to extract relations")
        return
    
    if args.resume and args.no_checkpoint:
        print("Error: Cannot use both --resume and --no-checkpoint options")
        return
    
    if args.resume:
        # Check if checkpoint file exists
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint file {args.checkpoint} not found")
            return
        print(f"Resuming from last interruption point...")
    
    print(f"Starting with the following parameters:")
    print(f"- Input file: {args.input}")
    print(f"- Output file: {args.output}")
    print(f"- Threads: {args.threads}")
    print(f"- Max records: {args.max_records if args.max_records else 'unlimited'}")
    print(f"- Mode: {args.mode}")
    print(f"- API key: {'set via parameter' if args.api_key != DEFAULT_API_KEY else ('set via environment variable' if DEFAULT_API_KEY else 'not set')}")
    print(f"- Checkpoint: {'disabled' if args.no_checkpoint else args.checkpoint}")
    print(f"- Resume mode: {'enabled' if args.resume else 'disabled'}")
    
    # Process dataset with checkpoint if enabled
    checkpoint_file = None if args.no_checkpoint else args.checkpoint
    process_dataset(args.input, args.output, args.threads, args.max_records, args.mode, args.api_key, checkpoint_file)
    
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
            checkpoint_file = "./spatial_relations_checkpoint.json"
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