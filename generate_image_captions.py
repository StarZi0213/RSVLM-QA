import os
import json
import base64
import time
import argparse
import traceback
from tqdm import tqdm
import requests
import threading
import queue
import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import io
import tempfile

# API key from environment variable or default to empty string
DEFAULT_API_KEY = 'sk-proj--0qkqsf9gWqeQhqOQQN3YDECGerqOvMsP6GD4nbdFaQRvFJWgLjl-5mMntZaC34-kZ4HWIMie2WgdhLQ8xC5I8bEV_Ki6xiIvz2dGbcXmxGZyvzQ17dFArpkA'

# API Rate limits
MAX_REQUESTS_PER_MINUTE = 10000  # 10k requests per minute
MAX_TOKENS_PER_MINUTE = 2000000  # 2M tokens per minute
MAX_TOKENS_PER_DAY = 200000000   # 200M tokens per day

# Tracking processed images to prevent duplicates
processed_image_lock = threading.Lock()
currently_processing = set()

# RateLimiter class to handle API rate limiting
class RateLimiter:
    def __init__(self, max_rpm, max_tpm):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.request_timestamps = []
        self.token_counts = []
        self.lock = threading.Lock()
        
    def wait_if_needed(self, estimated_tokens=1000):
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

def encode_image(image_path):
    """Encode image to base64 string, converting from TIF/TIFF to JPEG if needed"""
    try:
        # Check file extension
        file_extension = Path(image_path).suffix.lower()
        
        # If TIF/TIFF, convert to JPEG
        if file_extension in ['.tif', '.tiff']:
            print(f"Converting {file_extension} image to JPEG: {image_path}")
            with Image.open(image_path) as img:
                # Convert to RGB if needed (TIFFs can be in various modes)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPEG to a temporary buffer
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90)
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode("utf-8")
        else:
            # For regular supported formats
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        traceback.print_exc()
        return None

def analyze_image_with_gpt4(image_path, features, api_key):
    """Analyze an image using GPT-4.1 Vision API"""
    try:
        # Verify API key is provided
        if not api_key:
            print("❌ Error: No API key provided. Please set OPENAI_API_KEY environment variable or use --api-key parameter.")
            return None
            
        # Encode the image
        base64_image = encode_image(image_path)
        if not base64_image:
            print(f"Failed to encode image: {image_path}")
            return None
        
        # Convert features to a readable string format
        feature_str = ", ".join([f"{k}: {v:.1f}%" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in features.items() if v > 0])
        
        # Create the optimized prompt with stronger emphasis on summary generation
        # prompt = (
        #     f"I'm sending you a satellite/aerial image with these calculated land cover features: {feature_str}.\n\n"
        #     f"Please analyze this image thoroughly and verify if you can see these features. "
        #     f"Don't worry about exact percentages - use descriptive terms like 'large area', 'small portion', etc. "
        #     f"Focus on what's actually visible in the image rather than just repeating the feature list.\n\n"
        #     f"Respond in this exact format:\n"
        #     f"<caption>One comprehensive sentence describing the entire landscape image and its main features</caption>\n"
        #     f"<feature>List the features you can actually see in the image, confirming or contradicting the provided features. Be specific about locations (e.g., 'forest area in the northern section')</feature>\n"
        #     f"<summary>A detailed paragraph summarizing the image content by combining the visual elements and landscape features you identified. THIS SECTION IS REQUIRED AND MUST NOT BE LEFT EMPTY.</summary>"
        # )
        prompt = (
            f"I'm sending you a satellite/aerial image. Please analyze this image thoroughly. Provide directional information from an overall perspective. 
            Regarding the proportion of objects, please Don't give exact percentages.\\You should use descriptive terms like 'large area', 'small portion', etc. 
            Focus on what's actually visible in the image, . \\
            Respond in this exact format: "
            f"<One Sentence caption> describing the entire landscape image and its main features, Total-subtotal structure</One Sentence caption>"
            f"<feature>List the features you can actually see in the image, confirming or contradicting the provided features. Be specific about locations (e.g., 'forest area in the northern section')</feature>\n"
            f"<caption>A detailed paragraph summarizing the image content by combining the visual elements and landscape features you identified. THIS SECTION IS REQUIRED AND MUST NOT BE LEFT EMPTY. Total-subtotal structure</caption>"        
            )
        # Estimate token count for this request
        estimated_token_count = len(prompt.split()) * 1.3 + 1000  # Rough estimate
        
        # Wait if approaching rate limits
        wait_time = rate_limiter.wait_if_needed(estimated_token_count)
        if wait_time > 0:
            print(f"Rate limit approaching, waiting for {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        print(f"Sending image to GPT-4.1: {Path(image_path).name}")
        
        # Prepare the request
        user_content = [{"type": "text", "text": prompt}]
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        }
        user_content.append(image_content)
        
        messages = [{"role": "user", "content": user_content}]
        
        # API request parameters
        payload = {
            "model": "gpt-4.1",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.3,
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
            
            print(f"✓ Successfully analyzed image: {Path(image_path).name} ({total_tokens} tokens)")
            print(f"Response preview: {result[:100]}...")
            return result
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
        print(f"✗ Error analyzing image: {e}")
        traceback.print_exc()
        stats.add_result(False)
        return None

def load_jsonl_data(jsonl_file, mode="empty_summary"):
    """Load JSONL data based on the specified mode
    
    mode options:
    - "empty_summary": only load records with empty summaries (for fixing)
    - "new": load all records as new data to process
    """
    data = []
    total_count = 0
    empty_summary_count = 0
    new_count = 0
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    image_path = record.get("image", "")
                    image_id = record.get("image_id", "")
                    
                    # For new datasets, image_id might not be present
                    if not image_id and image_path:
                        image_id = Path(image_path).stem
                        record["image_id"] = image_id
                    
                    total_count += 1
                    
                    # Handle based on mode
                    if mode == "empty_summary":
                        description = record.get("Description", {})
                        summary = description.get("summary", "")
                        
                        # Check if summary is empty and the image file exists
                        if summary == "" and os.path.exists(image_path):
                            empty_summary_count += 1
                            data.append(record)
                    elif mode == "new":
                        # For new data, just check if the image exists
                        if os.path.exists(image_path):
                            new_count += 1
                            data.append(record)
                        else:
                            print(f"Warning: Image file not found: {image_path}")
                    
                except json.JSONDecodeError:
                    continue
    
    print(f"Dataset loading summary:")
    print(f"- Total records: {total_count}")
    if mode == "empty_summary":
        print(f"- Records with empty summaries: {empty_summary_count}")
    else:
        print(f"- New images to process: {new_count}")
    print(f"- Valid images to process: {len(data)}")
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

def process_single_image(record, output_file, result_queue, api_key=DEFAULT_API_KEY):
    """Process a single image record"""
    try:
        image_path = record.get("image", "")
        features = record.get("features", {})
        image_id = record.get("image_id", "")
        
        # Check if this image is already being processed by another thread
        with processed_image_lock:
            if image_id in currently_processing:
                print(f"Skipping duplicate task for image: {image_id}")
                return {"image_id": image_id, "status": "skipped_duplicate"}
            currently_processing.add(image_id)
        
        print(f"\n-------- Processing image: {image_id} --------")
        
        try:
            # Get the response from GPT-4.1
            response = analyze_image_with_gpt4(image_path, features, api_key)
            
            if response:
                # Parse the response to extract caption, feature, and summary
                caption = ""
                feature_text = ""
                summary = ""
                
                # Preserve existing data
                description = record.get("Description", {})
                caption = description.get("caption", "")
                feature_text = description.get("feature_analysis", "")
                
                if "<caption>" in response and "</caption>" in response:
                    new_caption = response.split("<caption>")[1].split("</caption>")[0].strip()
                    if caption == "":  # Only update if previous caption was empty
                        caption = new_caption
                    print(f"Caption: {caption}")
                else:
                    print("Warning: Caption tag not found in response")
                
                if "<feature>" in response and "</feature>" in response:
                    new_feature_text = response.split("<feature>")[1].split("</feature>")[0].strip()
                    if feature_text == "":  # Only update if previous feature was empty
                        feature_text = new_feature_text
                    print(f"Features: {feature_text[:100]}...")
                else:
                    print("Warning: Feature tag not found in response")
                
                if "<summary>" in response and "</summary>" in response:
                    summary = response.split("<summary>")[1].split("</summary>")[0].strip()
                    print(f"Summary: {summary[:100]}...")
                else:
                    print("Warning: Summary tag not found in response")
                    # If still no summary found, create a basic one
                    if not summary:
                        summary = "The image shows a landscape with various features that couldn't be fully analyzed."
                
                # Make sure summary is not empty
                if not summary.strip():
                    summary = "This image depicts a landscape that contains the features described above, though detailed analysis could not be completed."
                
                # Create result record
                result = {
                    "image": image_path,
                    "image_id": image_id,
                    "features": features,
                    "Description": {
                        "caption": caption,
                        "feature_analysis": feature_text,
                        "summary": summary,
                        "full_response": response
                    }
                }
                
                # Write to output file using a thread-safe method
                result_queue.put(result)
                
                print(f"✓ Result queued for saving: {image_id}")
                return {
                    "image_id": image_id,
                    "status": "success"
                }
            
            print(f"✗ Failed to get analysis for {image_id}")
            return {
                "image_id": image_id,
                "status": "failed"
            }
        finally:
            # Always remove from currently processing set, even if an error occurs
            with processed_image_lock:
                currently_processing.discard(image_id)
    
    except Exception as e:
        print(f"✗ Error processing record: {e}")
        traceback.print_exc()
        # Make sure to remove from currently processing if there's an exception
        with processed_image_lock:
            currently_processing.discard(image_id)
        return {
            "image_id": record.get("image_id", "unknown"),
            "status": "error",
            "error": str(e)
        }

def writer_thread(output_file, result_queue, stop_event, existing_data):
    """Thread to handle writing results to file"""
    count = 0
    last_report_time = time.time()
    
    print(f"Writer thread started, will update results in {output_file}")
    
    # Create a dictionary of existing data for quick lookup
    existing_data_dict = {item["image_id"]: idx for idx, item in enumerate(existing_data)}
    
    # Collect all processed results
    processed_results = []
    
    while not stop_event.is_set() or not result_queue.empty():
        try:
            # Get result from queue, wait up to 1 second
            try:
                result = result_queue.get(timeout=1)
            except queue.Empty:
                # If queue is empty and we need to report progress, do it
                if time.time() - last_report_time > 10:  # Report every 10 seconds
                    stats_data = stats.get_stats()
                    print(f"\nProgress: {stats_data['processed']} images, "
                          f"{stats_data['requests_per_minute']:.1f} rpm, "
                          f"{stats_data['tokens_per_minute']:.1f} tpm")
                    last_report_time = time.time()
                continue
            
            # Add to processed results
            processed_results.append(result)
            
            count += 1
            result_queue.task_done()
            
            # Periodically report progress
            if count % 10 == 0 or time.time() - last_report_time > 10:
                stats_data = stats.get_stats()
                print(f"\nProgress: {count} processed, "
                      f"{stats_data['requests_per_minute']:.1f} rpm, "
                      f"{stats_data['tokens_per_minute']:.1f} tpm")
                last_report_time = time.time()
        
        except Exception as e:
            print(f"Error in writer thread: {e}")
            traceback.print_exc()
    
    print(f"All processing complete. Updating output file with {count} new results...")
    
    # Update existing data with new results
    for result in processed_results:
        image_id = result["image_id"]
        if image_id in existing_data_dict:
            # Update existing record
            existing_data[existing_data_dict[image_id]] = result
        else:
            # Add new record
            existing_data.append(result)
    
    # Write updated data back to file
    temp_file = f"{output_file}.temp"
    with open(temp_file, 'w') as f:
        for item in existing_data:
            f.write(json.dumps(item) + '\n')
    
    # Replace original file with updated one
    os.replace(temp_file, output_file)
    
    print(f"Writer thread finished, updated {count} results in {output_file}")

def process_dataset(input_file, output_file, num_threads=20, max_images=None, mode="empty_summary", api_key=DEFAULT_API_KEY):
    """Process the dataset, focusing on images with empty summaries or processing new images
    
    mode options:
    - "empty_summary": process only images with empty summaries
    - "new": process new images for caption generation
    """
    print("\n" + "="*50)
    if mode == "empty_summary":
        print(f" Starting Image Caption Generation - Empty Summary Fix ")
    else:
        print(f" Starting Image Caption Generation - New Dataset ")
    print("="*50)
    
    # Verify API key is provided
    if not api_key:
        print("❌ Error: No API key provided. Please set OPENAI_API_KEY environment variable or use --api-key parameter.")
        return
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir and output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory ensured: {output_dir}")
    
    # Load existing data for updating if in empty_summary mode
    existing_data = []
    if mode == "empty_summary" and os.path.exists(output_file):
        existing_data = load_existing_data(output_file)
        print(f"Loaded {len(existing_data)} existing records from {output_file}")
    
    # Load data based on mode
    data = load_jsonl_data(input_file, mode)
    
    if max_images and max_images > 0:
        data = data[:max_images]
        print(f"Limited processing to {max_images} images for this run")
    
    if not data:
        if mode == "empty_summary":
            print("No images with empty summaries to process. Exiting.")
        else:
            print("No new images to process. Exiting.")
        return
    
    # Create a queue for results
    result_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Start writer thread with appropriate parameters
    if mode == "empty_summary":
        writer = threading.Thread(target=writer_thread, args=(output_file, result_queue, stop_event, existing_data))
    else:
        # For new data, start with empty existing_data
        writer = threading.Thread(target=writer_thread, args=(output_file, result_queue, stop_event, []))
    
    writer.daemon = True
    writer.start()
    
    # Calculate optimal number of threads - never use more threads than images
    optimal_threads = min(num_threads, len(data))
    if mode == "empty_summary":
        print(f"\nStarting to process {len(data)} images with empty summaries using {optimal_threads} threads...")
    else:
        print(f"\nStarting to process {len(data)} new images using {optimal_threads} threads...")
    print("-"*50)
    
    # Create a list to ensure each image is processed only once
    images_to_process = list(data)  # Make a copy to avoid modifying the original list
    
    # Process images with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
        # Create list of futures
        futures = []
        
        # Submit each image only once
        for record in images_to_process:
            # Create a future for this record
            future = executor.submit(process_single_image, record, output_file, result_queue, api_key)
            futures.append(future)
        
        # Create a progress bar
        with tqdm(total=len(futures), desc="Processing images") as pbar:
            # Process results as they complete
            for future in futures:
                try:
                    result = future.result()
                    # Update progress bar
                    pbar.update(1)
                except Exception as e:
                    print(f"Error in task: {e}")
                    traceback.print_exc()
    
    # Signal writer thread to stop and wait for it
    print("All processing tasks completed, waiting for writer to finish...")
    stop_event.set()
    writer.join()
    
    # Get final statistics
    final_stats = stats.get_stats()
    
    print("\n" + "="*50)
    print(" Processing Summary ")
    print("="*50)
    print(f"Successfully processed: {final_stats['success']} images")
    print(f"Failed: {final_stats['failure']} images")
    print(f"Total tokens used: {final_stats['total_tokens']}")
    print(f"Average tokens per image: {final_stats['total_tokens'] / final_stats['processed'] if final_stats['processed'] > 0 else 0:.1f}")
    print(f"Average processing rate: {final_stats['requests_per_minute']:.1f} requests/minute")
    print(f"Average token usage rate: {final_stats['tokens_per_minute']:.1f} tokens/minute")
    print(f"Total processing time: {final_stats['elapsed_seconds'] / 60:.1f} minutes")
    print("="*50)
    
    # Count if any empty summaries remain
    if mode == "empty_summary":
        empty_count = 0
        with open(output_file, 'r') as f:
            for line in f:
                if '"summary": ""' in line:
                    empty_count += 1
        
        if empty_count > 0:
            print(f"⚠️ WARNING: {empty_count} records still have empty summaries!")
        else:
            print("✅ SUCCESS: All records now have summaries!")

def main():
    parser = argparse.ArgumentParser(description="Generate image captions using GPT-4.1 Vision")
    parser.add_argument("--input", default="./image_captions.jsonl", help="Input JSONL file path")
    parser.add_argument("--output", default="./image_captions.jsonl", help="Output JSONL file path")
    parser.add_argument("--threads", type=int, default=20, help="Number of processing threads to use")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process (for testing)")
    parser.add_argument("--mode", choices=["empty_summary", "new"], default="empty_summary", 
                        help="Operation mode: 'empty_summary' to fix empty summaries, 'new' for new dataset")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, 
                        help="OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.")
    args = parser.parse_args()
    
    print(f"Starting with parameters:")
    print(f"- Input file: {args.input}")
    print(f"- Output file: {args.output}")
    print(f"- Threads: {args.threads}")
    print(f"- Max images: {args.max_images if args.max_images else 'No limit'}")
    print(f"- Mode: {args.mode}")
    print(f"- API key: {'Set via parameter' if args.api_key != DEFAULT_API_KEY else ('Set via environment variable' if DEFAULT_API_KEY else 'Not set')}")
    
    process_dataset(args.input, args.output, args.threads, args.max_images, args.mode, args.api_key)

if __name__ == "__main__":
    main() 