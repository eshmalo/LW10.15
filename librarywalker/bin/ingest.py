#!/usr/bin/env python3

import os
import sys
import glob

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from librarywalker.ingest.ingest import ingest_document
from librarywalker.config import DATA_DIR, MAX_CHUNKS_PER_LEVEL

def process_all_input_files():
    """Process all files in the input directory."""
    
    # Get absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Get input directory path with absolute path
    input_dir = os.path.join(project_root, "librarywalker", "data", "input")
    
    # Get output directory path
    output_dir = os.path.join(project_root, "librarywalker", "data", "output")
    
    # Check if directories exist
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print("Creating input directory...")
        os.makedirs(input_dir, exist_ok=True)
        print(f"Please place your documents in {input_dir} and run this script again.")
        return
        
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Make sure sample document exists
    sample_doc_path = os.path.join(project_root, "librarywalker", "sample_document.txt")
    target_sample_path = os.path.join(input_dir, "sample_document.txt")
    
    if os.path.exists(sample_doc_path) and not os.path.exists(target_sample_path):
        print("Copying sample document to input directory...")
        with open(sample_doc_path, 'r', encoding='utf-8') as src:
            content = src.read()
            with open(target_sample_path, 'w', encoding='utf-8') as dest:
                dest.write(content)
    
    # Get all files in input directory
    input_files = glob.glob(os.path.join(input_dir, "*"))
    
    # Filter out directories
    input_files = [f for f in input_files if os.path.isfile(f)]
    
    if not input_files:
        print(f"No files found in {input_dir}")
        print(f"Please place your documents in {input_dir} and run this script again.")
        return
    
    print(f"Found {len(input_files)} files to process:")
    for f in input_files:
        print(f"  - {os.path.basename(f)}")
    
    print("\nStarting ingestion process...\n")
    
    # Updated DATA_DIR to use absolute path
    data_dir = os.path.join(project_root, "librarywalker", "data")
    
    # Process each file
    for file_path in input_files:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        try:
            # Optimized parameters for reliable embedding within rate limits
            ingest_document(
                file_path, 
                data_dir=data_dir, 
                max_chunks_per_level=MAX_CHUNKS_PER_LEVEL,
                max_workers=8,  # Processing power for chunking (balanced)
                batch_size=30,   # Smaller batch size to reduce token count per request
                max_concurrent_embedding_requests=4  # Very conservative concurrency for stable operation
            )
            print(f"Successfully processed {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
    
    print("\nAll files processed!")
    print(f"Vector database stored in: {output_dir}")

if __name__ == "__main__":
    process_all_input_files()