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
from collections import Counter, defaultdict

# API key from environment variable or default to empty string
DEFAULT_API_KEY = os.environ.get('OPENAI_API_KEY', '')

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
    print("\nReceived interrupt signal, saving checkpoint...")
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
        checkpoint_file = "./count_vqa_pairs_gpt4_checkpoint.json"
    
    # Save checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "processed_ids": list(processed_ids_to_save),
            "timestamp": time.time(),
            "interrupted": True
        }, f)
    
    print(f"Saved progress with {len(processed_ids_to_save)} records to {checkpoint_file}")
    print("You can restart the program with the same command and it will continue from this point.")
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

def load_jsonl_dataset(jsonl_file, mode="basic"):
    """
    Load data from the JSONL file
    
    mode options:
    - "basic": load all records
    - "missing_gpt4_refinements": only load records without GPT-4.1 refinements
    """
    print(f"Loading dataset: {jsonl_file}")
    
    try:
        records = []
        total_count = 0
        missing_refinements = 0
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        total_count += 1
                        
                        # Add an ID if not present
                        if "id" not in record:
                            if "image_path" in record:
                                record["id"] = Path(record["image_path"]).stem
                            else:
                                record["id"] = f"record_{total_count}"
                        
                        # Check if count_vqa_pairs exist
                        has_count_vqa = "count_vqa_pairs" in record and isinstance(record["count_vqa_pairs"], list)
                        
                        # Check if gpt4_refined_vqa_pairs exist
                        has_gpt4_refinements = "gpt4_refined_vqa_pairs" in record and isinstance(record["gpt4_refined_vqa_pairs"], list)
                        
                        if mode == "basic" and has_count_vqa:
                            records.append(record)
                        elif mode == "missing_gpt4_refinements" and has_count_vqa and not has_gpt4_refinements:
                            missing_refinements += 1
                            records.append(record)
                            
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON line: {line[:100]}...")
                        continue
        
        print(f"Dataset loading summary:")
        print(f"- Total records: {total_count}")
        if mode == "missing_gpt4_refinements":
            print(f"- Records missing GPT-4.1 refinements: {missing_refinements}")
        print(f"- Valid records to process: {len(records)}")
        return records
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return []

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
                print(f"Detected previous interruption. Resuming from checkpoint...")
            
            print(f"Loaded checkpoint: {len(processed_ids)} processed records (Age: {age_hours:.1f} hours)")
            return processed_ids
    return set()

def load_existing_data(jsonl_file):
    """Load all data from existing JSONL file"""
    data = []
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        data.append(record)
                    except json.JSONDecodeError:
                        continue
    return data

@backoff.on_exception(backoff.expo, 
                     (requests.exceptions.RequestException, 
                      requests.exceptions.Timeout, 
                      requests.exceptions.ConnectionError),
                     max_tries=5)
def enhance_single_vqa_pair(vqa_pair, image_description, api_key):
    """
    Enhance a single VQA pair using GPT-4.1
    
    Args:
        vqa_pair: Dictionary containing a single question-answer pair
        image_description: Description of the image (if available)
        api_key: OpenAI API key
        
    Returns:
        dict: Enhanced VQA pair or None if enhancement failed
    """
    try:
        question = vqa_pair.get("question", "")
        answer = vqa_pair.get("answer", "")
        question_id = vqa_pair.get("question_id", "unknown")
        question_type = vqa_pair.get("question_type", "unknown")
        
        # Create the prompt for GPT-4.1
        prompt = f"""You are an expert in enhancing Visual Question Answering (VQA) pairs related to object counting in images.

Below is an automatically generated counting-based VQA pair for an image. 
This pair was generated based on object detection results without seeing the actual image.

Image description: {image_description}

Question: {question}
Answer: {answer}

Please enhance this VQA pair by:
1. Improving the language quality and naturalness of both the question and answer
2. Making the question more diverse in structure and complexity
3. Ensuring the answer is informative, complete, and grammatically correct
4. Keeping the same factual content but making the language more human-like
5. Maintaining the same objects and counts as in the original pair

Return the enhanced VQA pair in JSON format as follows:
{{
  "question_id": "{question_id}",
  "question_type": "{question_type}",
  "question": "enhanced question",
  "answer": "enhanced answer",
  "original_question": "{question}",
  "original_answer": "{answer}"
}}

DO NOT add any explanation before or after the JSON object.
"""

        # Estimate token count for this request
        estimated_token_count = len(prompt.split()) * 1.3 + 300  # Rough estimate
        
        # Wait if approaching rate limits
        wait_time = rate_limiter.wait_if_needed(estimated_token_count)
        if wait_time > 0:
            print(f"Rate limit approaching, waiting for {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        
        # Make the API request
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        # Record the request with rate limiter
        rate_limiter.add_request(estimated_token_count)
        
        # Process the response
        if response.status_code == 200:
            result = response.json()
            usage = result.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            
            # Get the content from the response
            content = result["choices"][0]["message"]["content"]
            
            # Extract JSON content from the response
            try:
                # Find JSON object in the response (might be wrapped in ```json or other formatting)
                if "{" in content and "}" in content:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    json_content = content[json_start:json_end]
                    enhanced_pair = json.loads(json_content)
                else:
                    # Try parsing the entire content as JSON
                    enhanced_pair = json.loads(content)
                
                # Update statistics
                stats.add_result(True, total_tokens)
                print(f"✅ Successfully enhanced VQA pair ({total_tokens} tokens)")
                
                return enhanced_pair, total_tokens
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON from GPT-4.1 response: {e}")
                print(f"Response content: {content[:300]}...")
                stats.add_result(False)
                return None, 0
        else:
            error_message = response.text
            print(f"❌ Request failed: {response.status_code}, Error: {error_message}")
            
            # Update statistics
            stats.add_result(False)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Waiting for {retry_after} seconds")
                time.sleep(retry_after)
            
            return None, 0
    
    except Exception as e:
        print(f"❌ Error enhancing VQA pair: {e}")
        traceback.print_exc()
        stats.add_result(False)
        return None, 0

def enhance_vqa_pairs_with_gpt4(record, api_key):
    """
    Enhance VQA pairs using GPT-4.1, one pair at a time
    
    Args:
        record: Dictionary containing record data with count_vqa_pairs
        api_key: OpenAI API key
        
    Returns:
        list: Enhanced VQA pairs
    """
    try:
        count_vqa_pairs = record.get("count_vqa_pairs", [])
        
        if not count_vqa_pairs:
            return []
        
        # Extract image description if available
        image_description = record.get("answer", "")
        
        # Process each VQA pair individually
        enhanced_pairs = []
        total_tokens_used = 0
        
        for i, pair in enumerate(count_vqa_pairs):
            print(f"Processing VQA pair {i+1}/{len(count_vqa_pairs)}...")
            
            # Enhance the individual VQA pair
            enhanced_pair, tokens_used = enhance_single_vqa_pair(pair, image_description, api_key)
            total_tokens_used += tokens_used
            
            if enhanced_pair:
                enhanced_pairs.append(enhanced_pair)
                print(f"  Q: {enhanced_pair.get('question', '')[:50]}...")
            else:
                # If enhancement failed, include the original pair with a flag
                fallback_pair = pair.copy()
                fallback_pair["original_question"] = pair.get("question", "")
                fallback_pair["original_answer"] = pair.get("answer", "")
                fallback_pair["enhancement_failed"] = True
                enhanced_pairs.append(fallback_pair)
                print(f"  Failed to enhance pair {i+1}")
        
        print(f"Enhanced {len(enhanced_pairs)}/{len(count_vqa_pairs)} VQA pairs, using {total_tokens_used} tokens total")
        return enhanced_pairs
        
    except Exception as e:
        print(f"❌ Error in enhance_vqa_pairs_with_gpt4: {e}")
        traceback.print_exc()
        return []

def process_single_record(record, output_file, result_queue, api_key=DEFAULT_API_KEY):
    """
    Process a single record to enhance VQA pairs using GPT-4.1
    
    Args:
        record: Dictionary containing record data
        output_file: Path to the output JSONL file
        result_queue: Queue for sending results to writer thread
        api_key: API key for OpenAI
    """
    record_id = record.get("id", "unknown")
    
    try:
        # Check if this record is already being processed
        with processed_lock:
            if record_id in currently_processing:
                print(f"Skipping duplicate processing: {record_id}")
                return
            currently_processing.add(record_id)
        
        print(f"Processing record: {record_id}")
        
        # Enhance VQA pairs using GPT-4.1
        enhanced_vqa_pairs = enhance_vqa_pairs_with_gpt4(record, api_key)
        
        if enhanced_vqa_pairs:
            # Create result record for writer thread
            result = record.copy()
            result["gpt4_refined_vqa_pairs"] = enhanced_vqa_pairs
            
            # Put result in the queue for writer thread
            result_queue.put(result)
            
            print(f"✅ Successfully enhanced {len(enhanced_vqa_pairs)} VQA pairs for record {record_id}")
        else:
            # Update statistics for failure
            print(f"❌ Failed to enhance VQA pairs for record {record_id}")
        
    except Exception as e:
        print(f"❌ Error processing record {record_id}: {e}")
        traceback.print_exc()
    finally:
        # Remove from currently processing set
        with processed_lock:
            currently_processing.discard(record_id)

def writer_thread(output_file, result_queue, stop_event, existing_data, checkpoint_file=None):
    """
    Thread to write results to output file as they become available
    
    Args:
        output_file: Path to the output JSONL file
        result_queue: Queue containing results to write
        stop_event: Event to signal thread to stop
        existing_data: List of existing records from output file
        checkpoint_file: Path to checkpoint file for saving progress
    """
    print(f"Writer thread started, writing results to {output_file}")
    
    # Create lookup for existing records
    existing_records = {record.get("id"): record for record in existing_data if "id" in record}
    processed_ids = set(existing_records.keys())
    print(f"Loaded {len(existing_records)} existing records from output file")
    
    # Load processed IDs from checkpoint if available
    if checkpoint_file and os.path.exists(checkpoint_file):
        checkpoint_processed_ids = load_checkpoint(checkpoint_file)
        processed_ids.update(checkpoint_processed_ids)
    
    # Create/open output file for appending
    with open(output_file, 'a') as f:
        checkpoint_counter = 0
        last_checkpoint_time = time.time()
        
        while not stop_event.is_set() or not result_queue.empty():
            try:
                # Get result with timeout to periodically check stop_event
                result = result_queue.get(timeout=1.0)
                
                # Get record ID
                record_id = result.get("id", "unknown")
                
                # Determine if this is an update to an existing record
                is_update = record_id in existing_records
                
                if is_update:
                    # Update in-memory copy of existing records
                    existing_records[record_id] = result
                    
                    # We'll rewrite the entire file at the end
                    pass
                else:
                    # Write new result to file
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()  # Flush to disk immediately
                    
                    # Add to existing records
                    existing_records[record_id] = result
                
                # Update processed IDs
                processed_ids.add(record_id)
                
                # Increment checkpoint counter
                checkpoint_counter += 1
                
                # Save checkpoint periodically
                current_time = time.time()
                if checkpoint_file and (checkpoint_counter >= 10 or current_time - last_checkpoint_time >= 300):
                    save_checkpoint(checkpoint_file, processed_ids)
                    checkpoint_counter = 0
                    last_checkpoint_time = current_time
                
                # Print progress stats
                if checkpoint_counter % 5 == 0:
                    current_stats = stats.get_stats()
                    print(f"Progress update: Processed={current_stats['processed']}, "
                          f"Success={current_stats['success']}, "
                          f"Failure={current_stats['failure']}, "
                          f"Requests/min={current_stats['requests_per_minute']:.1f}")
                
            except queue.Empty:
                # Queue is empty, no results to process
                continue
            except Exception as e:
                print(f"Writer thread error: {e}")
                traceback.print_exc()
        
        # If there were any updates to existing records, rewrite the entire file
        if any(record_id in existing_records for record_id in processed_ids):
            print("Updates to existing records detected. Rewriting entire output file...")
            temp_file = f"{output_file}.temp"
            with open(temp_file, 'w') as temp_f:
                for record in existing_records.values():
                    temp_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # Replace original file with the temp file
            os.replace(temp_file, output_file)
            print(f"Output file rewritten with {len(existing_records)} records")
        
        # Save final checkpoint
        if checkpoint_file:
            save_checkpoint(checkpoint_file, processed_ids)
    
    print("Writer thread completed")

def process_dataset(input_file, output_file, num_threads=10, max_records=None, mode="missing_gpt4_refinements", api_key=DEFAULT_API_KEY, checkpoint_file=None):
    """
    Process the dataset to enhance VQA pairs using GPT-4.1
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSONL file
        num_threads: Number of worker threads to use
        max_records: Maximum number of records to process (for testing)
        mode: Processing mode ("missing_gpt4_refinements" or "basic")
        api_key: OpenAI API key
        checkpoint_file: Path to checkpoint file for saving/loading progress
    """
    print(f"Starting dataset processing...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Mode: {mode}")
    print(f"Thread count: {num_threads}")
    print(f"Checkpoint file: {checkpoint_file}")
    
    # Load processed IDs from checkpoint if available
    processed_ids = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        processed_ids = load_checkpoint(checkpoint_file)
    print(f"Loaded {len(processed_ids)} processed record IDs from checkpoint")
    
    # Load existing data from output file if it exists
    existing_data = load_existing_data(output_file)
    print(f"Loaded {len(existing_data)} records from output file")
    
    # Add IDs from existing data to processed IDs
    for record in existing_data:
        if "id" in record:
            processed_ids.add(record["id"])
    
    # Load dataset
    records = load_jsonl_dataset(input_file, mode)
    
    # Filter out already processed records
    unprocessed_records = [r for r in records if r.get("id") not in processed_ids]
    print(f"Records to process: {len(unprocessed_records)} / {len(records)}")
    
    # Limit records for testing if specified
    if max_records and max_records > 0:
        unprocessed_records = unprocessed_records[:max_records]
        print(f"Limited to processing first {max_records} records")
    
    # Skip processing if no records to process
    if not unprocessed_records:
        print("No records to process, exiting.")
        return
    
    # Create queue for results
    result_queue = queue.Queue()
    
    # Create stop event for writer thread
    stop_event = threading.Event()
    
    # Start writer thread
    writer = threading.Thread(
        target=writer_thread,
        args=(output_file, result_queue, stop_event, existing_data, checkpoint_file)
    )
    writer.start()
    
    try:
        # Process records using thread pool
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks to thread pool
            futures = []
            for record in unprocessed_records:
                future = executor.submit(
                    process_single_record,
                    record, output_file, result_queue, api_key
                )
                futures.append(future)
            
            # Use tqdm to show progress
            for _ in tqdm(ThreadPoolExecutor().map(lambda f: f.result(), futures), 
                         total=len(futures), desc="Processing records", unit="record"):
                pass
    
    except KeyboardInterrupt:
        print("\nInterrupted, stopping...")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        traceback.print_exc()
    finally:
        # Signal writer thread to stop
        stop_event.set()
        
        # Wait for writer thread to finish
        writer.join()
    
    # Print final statistics
    final_stats = stats.get_stats()
    print("\nProcessing complete! Final statistics:")
    print(f"Records processed: {final_stats['processed']}")
    print(f"Successful enhancements: {final_stats['success']}")
    print(f"Failed enhancements: {final_stats['failure']}")
    print(f"Total runtime: {final_stats['elapsed_seconds'] / 60:.1f} minutes")
    print(f"Average processing rate: {final_stats['requests_per_minute']:.1f} records/minute")
    print(f"Total tokens used: {final_stats['total_tokens']}")

def main():
    """Main function to parse arguments and start processing"""
    parser = argparse.ArgumentParser(description="Enhance count-based VQA pairs using GPT-4.1")
    
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file path")
    parser.add_argument("-o", "--output", default="count_vqa_gpt4_enhanced.jsonl", help="Output JSONL file path")
    parser.add_argument("-t", "--threads", type=int, default=10, help="Number of processing threads")
    parser.add_argument("-m", "--mode", choices=["missing_gpt4_refinements", "basic"], default="missing_gpt4_refinements", 
                        help="Processing mode: 'missing_gpt4_refinements' only process records missing GPT-4.1 refinements, 'basic' process all records")
    parser.add_argument("-k", "--api-key", default=DEFAULT_API_KEY, help="OpenAI API key")
    parser.add_argument("-c", "--checkpoint", default="./count_vqa_pairs_gpt4_checkpoint.json", help="Checkpoint file path")
    parser.add_argument("-n", "--max-records", type=int, default=None, help="Maximum number of records to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: No API key provided. Please set OPENAI_API_KEY environment variable or use --api-key parameter.")
        return
    
    # Start processing
    process_dataset(
        input_file=args.input,
        output_file=args.output,
        num_threads=args.threads,
        max_records=args.max_records,
        mode=args.mode,
        api_key=args.api_key,
        checkpoint_file=args.checkpoint
    )

if __name__ == "__main__":
    main() 
