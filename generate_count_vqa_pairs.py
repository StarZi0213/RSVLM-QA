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
    print("\n程序接收到中断信号，正在保存检查点...")
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
        checkpoint_file = "./count_vqa_pairs_checkpoint.json"
    
    # Save checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "processed_ids": list(processed_ids_to_save),
            "timestamp": time.time(),
            "interrupted": True
        }, f)
    
    print(f"已保存 {len(processed_ids_to_save)} 条记录的进度到 {checkpoint_file}")
    print("您可以使用相同的命令重新启动程序，它将从中断点继续。")
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

def load_merged_dataset(json_file, mode="missing_count_vqa"):
    """
    Load data from the merged_dataset.json file
    
    mode options:
    - "missing_count_vqa": only load records without 'count_vqa_pairs' field
    - "all": load all records
    """
    print(f"正在加载数据集: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Convert to list of records if it's a dictionary
        if isinstance(data, dict):
            records = []
            for image_id, image_data in data.items():
                # Add image_id to the record
                record = image_data.copy()
                record["id"] = image_id
                records.append(record)
        else:
            records = data
            
        total_count = len(records)
        missing_count_vqa = 0
        valid_records = []
        
        # Process records based on mode
        for record in tqdm(records, desc="加载数据集"):
            # Add an ID if not present
            if "id" not in record:
                if "image_path" in record:
                    record["id"] = Path(record["image_path"]).stem
                else:
                    record["id"] = str(len(valid_records))
            
            # Check if record has the required 'Seg' field for object counting
            if "Seg" not in record or not record["Seg"]:
                continue
                
            # Handle based on mode
            if mode == "missing_count_vqa":
                has_count_vqa = "count_vqa_pairs" in record and isinstance(record["count_vqa_pairs"], list)
                
                # Check if count_vqa_pairs are missing
                if not has_count_vqa:
                    missing_count_vqa += 1
                    valid_records.append(record)
            elif mode == "all":
                valid_records.append(record)
        
        print(f"数据集加载摘要:")
        print(f"- 总记录数: {total_count}")
        if mode == "missing_count_vqa":
            print(f"- 缺少物体计数VQA问题对的记录: {missing_count_vqa}")
        print(f"- 要处理的有效记录: {len(valid_records)}")
        return valid_records
        
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        traceback.print_exc()
        return []

def extract_object_counts(record):
    """
    Extract object counts from the Seg field of a record
    
    Returns:
        dict: Dictionary mapping object types to their counts
    """
    object_counts = Counter()
    
    try:
        seg_data = record.get("Seg", {})
        
        # Count objects by category
        for category, instances in seg_data.items():
            # Skip if category is empty or not a string
            if not category or not isinstance(category, str):
                continue
                
            # Skip if instances is not a list
            if not isinstance(instances, list):
                continue
                
            # Count valid instances
            valid_instances = [inst for inst in instances if isinstance(inst, (dict, list))]
            if valid_instances:
                object_counts[category] = len(valid_instances)
        
        return object_counts
        
    except Exception as e:
        print(f"提取物体计数时出错: {e}")
        traceback.print_exc()
        return Counter()

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
    print(f"检查点已保存: {len(processed_ids)} 条已处理记录")

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
                print(f"检测到上次程序被中断。从中断点恢复...")
            
            print(f"已加载检查点: {len(processed_ids)} 条已处理记录 (距今: {age_hours:.1f} 小时)")
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

def generate_count_vqa_pairs(record):
    """
    Generate counting-based VQA pairs from a record
    
    Args:
        record: Dictionary containing image data with Seg field
        
    Returns:
        list: List of VQA pairs
    """
    vqa_pairs = []
    
    try:
        # Extract object counts from the record
        object_counts = extract_object_counts(record)
        
        # Skip if no objects to count
        if not object_counts:
            return []
            
        # Get image description if available
        image_description = record.get("answer", "")
        
        # Generate VQA pairs for each object type
        for obj_type, count in object_counts.items():
            # Clean up object type name for better readability
            clean_obj_type = obj_type.replace("_", " ").strip().lower()
            
            # Generate a count question
            question = f"How many {clean_obj_type}s are there in the image?"
            answer = f"There are {count} {clean_obj_type}" + ("s" if count > 1 else "") + " in the image."
            
            vqa_pairs.append({
                "question_id": f"count_{obj_type}_{len(vqa_pairs) + 1}",
                "question_type": "count",
                "question": question,
                "answer": answer
            })
            
            # For objects with count > 1, add a yes/no question about presence
            if count > 0:
                question = f"Are there any {clean_obj_type}s in the image?"
                answer = f"Yes, there are {count} {clean_obj_type}" + ("s" if count > 1 else "") + "."
                
                vqa_pairs.append({
                    "question_id": f"presence_{obj_type}_{len(vqa_pairs) + 1}",
                    "question_type": "presence",
                    "question": question,
                    "answer": answer
                })
                
            # For images with multiple object types, generate comparative questions
            if len(object_counts) > 1 and list(object_counts.keys()).index(obj_type) == 0:
                # Get a different object type for comparison
                other_types = [t for t in object_counts.keys() if t != obj_type]
                if other_types:
                    other_type = other_types[0]
                    clean_other_type = other_type.replace("_", " ").strip().lower()
                    other_count = object_counts[other_type]
                    
                    # Generate comparative question
                    question = f"Are there more {clean_obj_type}s or {clean_other_type}s in the image?"
                    
                    if count > other_count:
                        answer = f"There are more {clean_obj_type}s ({count}) than {clean_other_type}s ({other_count})."
                    elif count < other_count:
                        answer = f"There are more {clean_other_type}s ({other_count}) than {clean_obj_type}s ({count})."
                    else:
                        answer = f"There are equal numbers of {clean_obj_type}s and {clean_other_type}s ({count} each)."
                    
                    vqa_pairs.append({
                        "question_id": f"compare_{obj_type}_{other_type}_{len(vqa_pairs) + 1}",
                        "question_type": "comparison",
                        "question": question,
                        "answer": answer
                    })
        
        # Add overall count question if there are multiple object types
        if len(object_counts) > 1:
            total_count = sum(object_counts.values())
            obj_list = ", ".join([f"{count} {t.replace('_', ' ').lower()}" + ("s" if count > 1 else "") 
                                for t, count in object_counts.items()])
            
            question = "What is the total number of objects in the image?"
            answer = f"There are a total of {total_count} objects in the image, consisting of {obj_list}."
            
            vqa_pairs.append({
                "question_id": f"total_count_{len(vqa_pairs) + 1}",
                "question_type": "total_count",
                "question": question,
                "answer": answer
            })
            
        return vqa_pairs
        
    except Exception as e:
        print(f"生成计数VQA对时出错: {e}")
        traceback.print_exc()
        return []

def process_single_record(record, output_file, result_queue, api_key=DEFAULT_API_KEY):
    """
    Process a single record to generate counting-based VQA pairs
    
    Args:
        record: Dictionary containing image data
        output_file: Path to the output JSONL file
        result_queue: Queue for sending results to writer thread
        api_key: API key for OpenAI (not used in this function but kept for compatibility)
    """
    record_id = record.get("id", "unknown")
    
    try:
        # Check if this record is already being processed
        with processed_lock:
            if record_id in currently_processing:
                print(f"跳过重复处理: {record_id}")
                return
            currently_processing.add(record_id)
        
        print(f"正在处理记录: {record_id}")
        
        # Generate count-based VQA pairs
        count_vqa_pairs = generate_count_vqa_pairs(record)
        
        if count_vqa_pairs:
            # Create result record for writer thread
            result = {
                "id": record_id,
                "image_path": record.get("image_path", ""),
                "count_vqa_pairs": count_vqa_pairs
            }
            
            # Add original record data 
            for key in ["answer", "tags", "relations"]:
                if key in record:
                    result[key] = record[key]
            
            # Put result in the queue for writer thread
            result_queue.put(result)
            
            # Update statistics
            stats.add_result(True)
            print(f"✅ 成功为记录 {record_id} 生成 {len(count_vqa_pairs)} 个计数VQA对")
        else:
            # Update statistics for failure
            stats.add_result(False)
            print(f"❌ 无法为记录 {record_id} 生成计数VQA对")
        
    except Exception as e:
        # Update statistics for failure
        stats.add_result(False)
        print(f"❌ 处理记录 {record_id} 时出错: {e}")
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
    print(f"写入线程已启动，将结果写入 {output_file}")
    
    # Create lookup for existing records
    existing_records = {record.get("id"): record for record in existing_data if "id" in record}
    processed_ids = set(existing_records.keys())
    print(f"从现有输出文件中读取了 {len(existing_records)} 条记录")
    
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
                
                # Write result to file
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()  # Flush to disk immediately
                
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
                    print(f"进度更新: 已处理={current_stats['processed']}, "
                          f"成功={current_stats['success']}, "
                          f"失败={current_stats['failure']}, "
                          f"请求/分钟={current_stats['requests_per_minute']:.1f}")
                
            except queue.Empty:
                # Queue is empty, no results to process
                continue
            except Exception as e:
                print(f"写入线程错误: {e}")
                traceback.print_exc()
        
        # Save final checkpoint
        if checkpoint_file:
            save_checkpoint(checkpoint_file, processed_ids)
    
    print("写入线程已完成")

def process_dataset(input_file, output_file, num_threads=20, max_records=None, mode="missing_count_vqa", api_key=DEFAULT_API_KEY, checkpoint_file=None):
    """
    Process the dataset to generate counting-based VQA pairs
    
    Args:
        input_file: Path to the input JSON file (merged_dataset.json)
        output_file: Path to the output JSONL file
        num_threads: Number of worker threads to use
        max_records: Maximum number of records to process (for testing)
        mode: Processing mode ("missing_count_vqa" or "all")
        api_key: OpenAI API key
        checkpoint_file: Path to checkpoint file for saving/loading progress
    """
    print(f"开始处理数据集...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"模式: {mode}")
    print(f"线程数: {num_threads}")
    print(f"检查点文件: {checkpoint_file}")
    
    # Load processed IDs from checkpoint if available
    processed_ids = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        processed_ids = load_checkpoint(checkpoint_file)
    print(f"已从检查点加载 {len(processed_ids)} 条已处理记录的ID")
    
    # Load existing data from output file if it exists
    existing_data = load_existing_data(output_file)
    print(f"已从输出文件加载 {len(existing_data)} 条记录")
    
    # Load dataset
    records = load_merged_dataset(input_file, mode)
    
    # Filter out already processed records
    unprocessed_records = [r for r in records if r.get("id") not in processed_ids]
    print(f"待处理记录数: {len(unprocessed_records)} / {len(records)}")
    
    # Limit records for testing if specified
    if max_records and max_records > 0:
        unprocessed_records = unprocessed_records[:max_records]
        print(f"限制为处理前 {max_records} 条记录")
    
    # Skip processing if no records to process
    if not unprocessed_records:
        print("没有要处理的记录，退出程序。")
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
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                         desc="处理记录", unit="record"):
                pass
    
    except KeyboardInterrupt:
        print("\n收到中断，正在停止...")
    except Exception as e:
        print(f"处理数据集时出错: {e}")
        traceback.print_exc()
    finally:
        # Signal writer thread to stop
        stop_event.set()
        
        # Wait for writer thread to finish
        writer.join()
    
    # Print final statistics
    final_stats = stats.get_stats()
    print("\n处理完成! 最终统计:")
    print(f"已处理记录数: {final_stats['processed']}")
    print(f"成功记录数: {final_stats['success']}")
    print(f"失败记录数: {final_stats['failure']}")
    print(f"总运行时间: {final_stats['elapsed_seconds'] / 60:.1f} 分钟")
    print(f"平均处理速率: {final_stats['requests_per_minute']:.1f} 记录/分钟")

def main():
    """Main function to parse arguments and start processing"""
    parser = argparse.ArgumentParser(description="从现有数据集生成计数型VQA问题对")
    
    parser.add_argument("-i", "--input", required=True, help="输入数据集文件路径 (merged_dataset.json)")
    parser.add_argument("-o", "--output", default="count_vqa_dataset.jsonl", help="输出JSONL文件路径")
    parser.add_argument("-t", "--threads", type=int, default=10, help="处理线程数量")
    parser.add_argument("-m", "--mode", choices=["missing_count_vqa", "all"], default="missing_count_vqa", 
                        help="处理模式: 'missing_count_vqa' 仅处理缺少计数VQA对的记录, 'all' 处理所有记录")
    parser.add_argument("-k", "--api-key", default=DEFAULT_API_KEY, help="OpenAI API密钥")
    parser.add_argument("-c", "--checkpoint", default="./count_vqa_pairs_checkpoint.json", help="检查点文件路径")
    parser.add_argument("-n", "--max-records", type=int, default=None, help="最大处理记录数 (测试用)")
    
    args = parser.parse_args()
    
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