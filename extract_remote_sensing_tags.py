#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import time
import threading
import queue
import requests
from pathlib import Path
from tqdm import tqdm
import logging
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extract_tags.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TagExtractor:
    def __init__(self, input_file, output_file, api_key, num_threads=4, 
                batch_size=10, max_records=None, checkpoint_file="checkpoint.json"):
        self.input_file = input_file
        self.output_file = output_file
        self.api_key = api_key
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.max_records = max_records
        self.checkpoint_file = checkpoint_file
        
        # Data structures
        self.task_queue = queue.Queue()
        self.results = []
        self.processed_ids = set()
        self.lock = threading.Lock()
        self.total_records = 0
        self.pbar = None
        
        # Counters
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
    
    def load_checkpoint(self):
        """Load checkpoint data if exists"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    self.processed_ids = set(checkpoint_data.get('processed_ids', []))
                    self.results = checkpoint_data.get('results', [])
                    self.success_count = checkpoint_data.get('success_count', 0)
                    self.error_count = checkpoint_data.get('error_count', 0)
                    
                logger.info(f"Loaded checkpoint: {len(self.processed_ids)} records already processed")
                return True
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                return False
        return False
    
    def save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            checkpoint_data = {
                'processed_ids': list(self.processed_ids),
                'results': self.results,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
                
            logger.info(f"Checkpoint saved: {len(self.processed_ids)} records")
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def save_results(self):
        """Save results to output file"""
        try:
            # If the output file exists and we're appending, read existing content first
            existing_ids = set()
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r') as f:
                        for line in f:
                            try:
                                record = json.loads(line)
                                existing_ids.add(record.get('image_id', ''))
                            except:
                                continue
                except Exception as e:
                    logger.error(f"Error reading existing output file: {e}")
            
            # Write results, avoiding duplicates
            with open(self.output_file, 'a') as f:
                count = 0
                for result in self.results:
                    if result.get('image_id', '') not in existing_ids:
                        f.write(json.dumps(result) + '\n')
                        count += 1
                
            logger.info(f"Saved {count} new records to {self.output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def process_record(self, record):
        """Process a single record to extract remote sensing tags"""
        image_id = record.get('image_id', '')
        caption = record.get('Description', {}).get('caption', '')
        summary = record.get('Description', {}).get('summary', '')
        
        if not caption and not summary:
            return None
        
        try:
            # Prepare prompt for the API
            content = f"""
Extract key remote sensing objects, land cover types, and geographical elements from this aerial/satellite image description. 
Identify concrete entities (like buildings, roads, water bodies) and land cover types (like forest, urban, agricultural).
Return the result as a list of specific tags.

Image caption: {caption}

Image summary: {summary}

Expected output format:
{{"tags": ["tag1", "tag2", "tag3", ...]}}
"""
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "gpt-4.1",
                "messages": [
                    {"role": "system", "content": "You are a remote sensing expert who extracts object tags from image descriptions."},
                    {"role": "user", "content": content}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", 
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    message_content = response_data["choices"][0]["message"]["content"]
                    
                    # Try to parse the JSON from the response
                    # First check if there's a JSON block in markdown format
                    if "```json" in message_content and "```" in message_content:
                        json_str = message_content.split("```json")[1].split("```")[0].strip()
                        tags_data = json.loads(json_str)
                    else:
                        # Otherwise try to parse the whole content
                        tags_data = json.loads(message_content)
                    
                    # Create result record
                    result = {
                        "image_id": image_id,
                        "image": record.get("image", ""),
                        "tags": tags_data.get("tags", [])
                    }
                    
                    return result
                except Exception as e:
                    logger.error(f"Error parsing API response for {image_id}: {e}")
                    logger.error(f"Response content: {message_content}")
                    return None
            else:
                logger.error(f"API error for {image_id}: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Exception processing {image_id}: {e}")
            return None
    
    def worker(self):
        """Worker thread to process items from the queue"""
        while True:
            try:
                record = self.task_queue.get(block=False)
                if record is None:
                    break
                    
                image_id = record.get('image_id', '')
                
                # Skip if already processed
                if image_id in self.processed_ids:
                    self.task_queue.task_done()
                    continue
                
                # Process the record
                result = self.process_record(record)
                
                with self.lock:
                    if result:
                        self.results.append(result)
                        self.success_count += 1
                    else:
                        self.error_count += 1
                    
                    self.processed_ids.add(image_id)
                    if self.pbar:
                        self.pbar.update(1)
                    
                    # Periodically save checkpoint (every batch_size records)
                    if (self.success_count + self.error_count) % self.batch_size == 0:
                        self.save_checkpoint()
                
                self.task_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self.task_queue.task_done()
    
    def run(self):
        """Main processing function"""
        # Load checkpoint if exists
        self.load_checkpoint()
        
        # Count total records for progress bar
        try:
            with open(self.input_file, 'r') as f:
                for _ in f:
                    self.total_records += 1
            
            if self.max_records and self.max_records < self.total_records:
                self.total_records = self.max_records
        except Exception as e:
            logger.error(f"Error counting records: {e}")
            return False
        
        # Initialize progress bar
        self.pbar = tqdm(total=self.total_records, initial=len(self.processed_ids),
                         desc="Extracting tags", unit="record")
        
        # Load records into the queue
        try:
            count = 0
            with open(self.input_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            image_id = record.get('image_id', '')
                            
                            # Skip if already processed
                            if image_id in self.processed_ids:
                                continue
                                
                            self.task_queue.put(record)
                            count += 1
                            
                            # Check if we've reached max_records
                            if self.max_records and count >= self.max_records:
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in line: {line[:50]}...")
                            continue
            
            logger.info(f"Loaded {count} records for processing")
        except Exception as e:
            logger.error(f"Error loading records: {e}")
            return False
        
        # Create and start worker threads
        threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)
        
        # Wait for all work to complete
        try:
            # Add sentinel values to signal threads to exit
            for _ in range(self.num_threads):
                self.task_queue.put(None)
            
            # Wait for all threads to finish
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, saving progress...")
            self.save_checkpoint()
            self.save_results()
            raise
        
        # Close progress bar
        if self.pbar:
            self.pbar.close()
        
        # Save final results
        self.save_checkpoint()
        self.save_results()
        
        # Report statistics
        elapsed_time = time.time() - self.start_time
        logger.info(f"Processing complete!")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"Records processed: {len(self.processed_ids)}")
        logger.info(f"Success: {self.success_count}, Errors: {self.error_count}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Extract remote sensing tags from image captions using gpt-4.1")
    parser.add_argument("--input", default="./image_captions.jsonl", help="Input JSONL file path")
    parser.add_argument("--output", default="./image_tags.jsonl", help="Output JSONL file path")
    parser.add_argument("--api-key", help="OpenAI API key",default='')
    parser.add_argument("--threads", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for checkpoints")
    parser.add_argument("--max-records", type=int, default=None, help="Maximum number of records to process")
    parser.add_argument("--checkpoint", default="checkpoint.json", help="Checkpoint file path")
    args = parser.parse_args()
    
    # Print startup info
    logger.info(f"Starting tag extraction with parameters:")
    logger.info(f"- Input file: {args.input}")
    logger.info(f"- Output file: {args.output}")
    logger.info(f"- Worker threads: {args.threads}")
    logger.info(f"- Batch size: {args.batch_size}")
    logger.info(f"- Max records: {args.max_records if args.max_records else 'All'}")
    logger.info(f"- Checkpoint file: {args.checkpoint}")
    
    # Initialize extractor
    extractor = TagExtractor(
        input_file=args.input,
        output_file=args.output,
        api_key=args.api_key,
        num_threads=args.threads,
        batch_size=args.batch_size,
        max_records=args.max_records,
        checkpoint_file=args.checkpoint
    )
    
    # Run extraction
    extractor.run()

if __name__ == "__main__":
    main() 