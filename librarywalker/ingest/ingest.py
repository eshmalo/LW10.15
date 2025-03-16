#!/usr/bin/env python3

import os
import argparse
import sys
from typing import List, Dict, Any

from ..config import DATA_DIR, BASE_CHUNK_SIZE, MAX_CHUNKS_PER_LEVEL, EMBEDDING_COSTS, OPENAI_EMBEDDING_MODEL
from ..vector_store import VectorStore
from .chunker import create_multi_level_chunks
from .embedder import embed_chunks, embed_chunks_parallel, estimate_embedding_cost

def ingest_document(document_path: str, 
                  data_dir: str = DATA_DIR,
                  chunk_mode: str = BASE_CHUNK_SIZE,
                  batch_size: int = 30,  # Smaller batches for more reliable processing
                  max_chunks_per_level: int = MAX_CHUNKS_PER_LEVEL,
                  max_workers: int = 8,   # Balanced worker count
                  max_concurrent_embedding_requests: int = 4) -> None:
    """
    Ingest a document: chunk, embed, and store.
    
    Args:
        document_path: Path to the document file
        data_dir: Directory to store vector database
        chunk_mode: Mode for base chunk splitting
        batch_size: Batch size for embedding API calls
        max_chunks_per_level: Maximum chunks to process per level
        max_workers: Maximum number of parallel processes to use for chunking
        max_concurrent_embedding_requests: Maximum number of concurrent embedding API requests
    """
    # Validate document path
    if not os.path.exists(document_path):
        print(f"Error: Document path '{document_path}' does not exist")
        return
        
    # Create input directory to copy document if needed
    input_dir = os.path.join(data_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    
    # If desired, copy the file to input directory for future reference
    document_filename = os.path.basename(document_path)
    input_copy_path = os.path.join(input_dir, document_filename)
    if document_path != input_copy_path and not os.path.exists(input_copy_path):
        if document_path.endswith('.pdf'):
            print(f"Copying PDF document to input directory: {input_copy_path}")
            import shutil
            try:
                shutil.copy2(document_path, input_copy_path)
            except Exception as e:
                print(f"Error copying PDF: {str(e)}")
        else:
            print(f"Copying document to input directory: {input_copy_path}")
            try:
                with open(document_path, 'r', encoding='utf-8') as src_file:
                    content = src_file.read()
                    with open(input_copy_path, 'w', encoding='utf-8') as dest_file:
                        dest_file.write(content)
            except Exception as e:
                print(f"Error copying text file: {str(e)}")
    
    # Create vector store
    vector_store = VectorStore(data_dir)
    
    # Generate chunks from document
    print(f"Creating chunks from '{document_path}'...")
    chunks = create_multi_level_chunks(document_path, chunk_mode, max_workers=max_workers)
    
    # For very large documents, we'll process in batches by level
    # This prevents out-of-memory issues and API rate limits
    
    # Group chunks by level
    chunks_by_level = {}
    for chunk in chunks:
        level = chunk["level"]
        if level not in chunks_by_level:
            chunks_by_level[level] = []
        chunks_by_level[level].append(chunk)
            
    # Verify we have chunks at all expected levels - should usually be at least 3 levels total
    # with higher levels having larger chunks up to the token limit
    num_levels = len(chunks_by_level.keys())
    if num_levels < 3:
        print(f"Warning: Only created chunks at {num_levels} levels (expected at least 3)")
        print("This may indicate an issue with the chunking process")
    
    # Count total chunks to process
    total_chunks = sum(len(chunks) for chunks in chunks_by_level.values())
    processed_chunks = 0
    
    # Calculate total tokens and estimated embedding cost
    # Get flattened list of all chunks
    all_chunks_flat = [chunk for level_chunks in chunks_by_level.values() for chunk in level_chunks]
    
    # Use the dedicated cost estimator function
    total_cost_details = estimate_embedding_cost(all_chunks_flat)
    total_cost = total_cost_details["total_cost"]
    total_tokens = total_cost_details["total_tokens"]
    cost_per_million = total_cost_details["cost_per_million_tokens"]
    
    # Get file-specific information
    file_name = os.path.basename(document_path)
    file_size_bytes = os.path.getsize(document_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Display cost estimates with file information
    print(f"\nEmbedding Cost Estimate for: {file_name}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Model: {OPENAI_EMBEDDING_MODEL} (${cost_per_million:.2f} per million tokens)")
    print(f"Total chunks to process: {total_chunks:,}")
    print(f"Total tokens to embed: {total_tokens:,}")
    
    # Format cost based on magnitude
    if total_cost < 0.001:
        cost_str = f"${total_cost:.8f}"
    elif total_cost < 0.01:
        cost_str = f"${total_cost:.6f}"
    elif total_cost < 0.1:
        cost_str = f"${total_cost:.4f}"
    elif total_cost < 1:
        cost_str = f"${total_cost:.3f}"
    else:
        cost_str = f"${total_cost:.2f}"
        
    print(f"Estimated embedding cost: {cost_str}")
    
    # Add tokens per MB metrics
    tokens_per_mb = total_tokens / file_size_mb if file_size_mb > 0 else 0
    cost_per_mb = total_cost / file_size_mb if file_size_mb > 0 else 0
    print(f"Tokens per MB: {tokens_per_mb:.0f}")
    print(f"Cost per MB: ${cost_per_mb:.6f}")
    
    # Provide per-level cost breakdown
    print("\nCost breakdown by level:")
    for level in sorted(chunks_by_level.keys()):
        level_chunks = chunks_by_level[level]
        level_cost_details = estimate_embedding_cost(level_chunks)
        level_cost = level_cost_details["total_cost"]
        level_tokens = level_cost_details["total_tokens"]
        level_chunks_count = len(level_chunks)
        
        # Calculate percentage of total
        level_percent_tokens = (level_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        level_percent_cost = (level_cost / total_cost) * 100 if total_cost > 0 else 0
        
        # Format level cost
        if level_cost < 0.001:
            level_cost_str = f"${level_cost:.8f}"
        elif level_cost < 0.01:
            level_cost_str = f"${level_cost:.6f}"
        elif level_cost < 0.1:
            level_cost_str = f"${level_cost:.4f}"
        elif level_cost < 1:
            level_cost_str = f"${level_cost:.3f}"
        else:
            level_cost_str = f"${level_cost:.2f}"
            
        print(f"  Level {level}: {level_chunks_count:,} chunks, {level_tokens:,} tokens ({level_cost_str})")
        print(f"    {level_percent_tokens:.1f}% of tokens, {level_percent_cost:.1f}% of total cost")
    
    # Just show the cost information without any warnings or constraints
    # The user can decide if they want to continue or abort
    print("\nNote: You can interrupt this process with Ctrl+C if needed before embedding begins.")
    
    # Process each level
    for level in sorted(chunks_by_level.keys()):
        level_chunks = chunks_by_level[level]
        print(f"Processing level {level} chunks ({len(level_chunks)} chunks, {len(level_chunks)/total_chunks*100:.1f}% of total)...")
        
        # For large chunk sets, show more detailed notice
        if len(level_chunks) > 1000:
            print(f"Processing all {len(level_chunks)} chunks at level {level}...")
            estimated_time = len(level_chunks) * 0.05  # Very rough estimate: ~0.05 seconds per chunk
            print(f"Estimated time: {estimated_time/60:.1f} minutes. This is processing ALL chunks as requested.")
        
        # Embed chunks for this level
        print(f"Embedding {len(level_chunks)} chunks at level {level} using parallel requests...")
        
        # Calculate optimal batch size based on chunk count and token density
        # Calculate average tokens per chunk for this level
        avg_tokens_per_chunk = sum(c["metadata"]["token_count"] for c in level_chunks) / len(level_chunks)
        token_density_factor = min(avg_tokens_per_chunk / 1000, 1)  # Normalize to 0-1 based on 1000 tokens
        
        # Adjust batch size dynamically based on token density and level size
        dynamic_batch_size = batch_size
        if avg_tokens_per_chunk > 1000:
            # Higher token counts need smaller batches
            dynamic_batch_size = max(10, int(batch_size * (1 - token_density_factor * 0.5)))
            print(f"High token density detected ({avg_tokens_per_chunk:.0f} tokens/chunk)")
            print(f"Using smaller batch size ({dynamic_batch_size}) for dense chunks")
        elif len(level_chunks) > 2000:
            # For very large levels, use smaller batches
            dynamic_batch_size = max(15, batch_size - 10)
            print(f"Using smaller batch size ({dynamic_batch_size}) for large level ({len(level_chunks)} chunks)")
        
        # Adjust concurrency based on level size and token density
        dynamic_concurrency = max_concurrent_embedding_requests
        if avg_tokens_per_chunk > 2000 or len(level_chunks) > 5000:
            # For extremely large or dense levels, reduce concurrency further
            reduction_factor = max(token_density_factor, min(len(level_chunks) / 10000, 1))
            dynamic_concurrency = max(2, int(max_concurrent_embedding_requests * (1 - reduction_factor * 0.5)))
            print(f"Using reduced concurrency ({dynamic_concurrency}) for better stability")
            
        # Use parallel or sequential embedding based on chunk count
        if len(level_chunks) > 100:
            embedded_chunks = embed_chunks_parallel(level_chunks, 
                                                  batch_size=dynamic_batch_size, 
                                                  max_concurrent_requests=dynamic_concurrency)
        else:
            embedded_chunks = embed_chunks(level_chunks, batch_size=dynamic_batch_size)
        
        # Prepare data for storage
        embeddings = [chunk["embedding"] for chunk in embedded_chunks]
        metadata = [chunk["metadata"] for chunk in embedded_chunks]
        texts = [chunk["text"] for chunk in embedded_chunks]
        
        # Storage optimization message - show only for higher levels
        if level > 0 and level == sorted(chunks_by_level.keys())[1]:  # First time we process a higher level
            print("\nStorage optimization: Only text content for level-0 chunks will be stored permanently.")
            print("Higher-level chunks (L1+) are represented only by their embeddings and metadata.")
            print("This significantly reduces storage requirements while maintaining search functionality.\n")
        
        # Store in vector database
        print(f"Storing level {level} chunks in vector database...")
        vector_store.add_embeddings(embeddings, metadata, texts)
        
        # Update progress
        processed_chunks += len(level_chunks)
        progress_percent = processed_chunks / total_chunks * 100
        print(f"Completed processing level {level} chunks ({progress_percent:.1f}% of total)")
    
    print(f"\nSummary for: {os.path.basename(document_path)}")
    print(f"Total chunks processed: {len(chunks):,}")
    print(f"Total tokens embedded: {total_tokens:,}")
    
    # Format the final cost
    if total_cost < 0.001:
        final_cost_str = f"${total_cost:.8f}"
    elif total_cost < 0.01:
        final_cost_str = f"${total_cost:.6f}"
    elif total_cost < 0.1:
        final_cost_str = f"${total_cost:.4f}"
    elif total_cost < 1:
        final_cost_str = f"${total_cost:.3f}"
    else:
        final_cost_str = f"${total_cost:.2f}"
        
    print(f"Total embedding cost: {final_cost_str}")
    
    # Calculate storage optimization statistics
    l0_chunks_count = len(chunks_by_level.get(0, []))
    higher_level_chunks_count = sum(len(chunks_by_level.get(level, [])) for level in chunks_by_level if level > 0)
    
    # Calculate average characters per chunk
    all_texts = " ".join([chunk["text"] for chunk in chunks])
    avg_chars_per_chunk = len(all_texts) / len(chunks) if chunks else 0
    
    # Estimate storage savings
    approx_bytes_saved = int(higher_level_chunks_count * avg_chars_per_chunk)
    
    # Convert to appropriate unit
    if approx_bytes_saved > 1024*1024:
        storage_saved = f"{approx_bytes_saved/(1024*1024):.2f} MB"
    elif approx_bytes_saved > 1024:
        storage_saved = f"{approx_bytes_saved/1024:.2f} KB"
    else:
        storage_saved = f"{approx_bytes_saved} bytes"
    
    print(f"Storage optimization: Only storing text for {l0_chunks_count} L0 chunks, not {higher_level_chunks_count} higher-level chunks")
    print(f"Estimated storage saved: {storage_saved}")
    print(f"Processing complete!")

def main():
    """Command-line interface for document ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents for Promptchan")
    parser.add_argument("document_path", help="Path to document file")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory for vector database")
    parser.add_argument("--chunk-mode", choices=["sentence", "paragraph", "line"], 
                      default=BASE_CHUNK_SIZE, help="Mode for base chunk splitting")
    parser.add_argument("--batch-size", type=int, default=20, 
                      help="Batch size for embedding API calls")
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS_PER_LEVEL,
                      help="Maximum chunks to process per level")
    parser.add_argument("--workers", type=int, default=4,
                      help="Number of parallel processes to use for chunking")
    
    args = parser.parse_args()
    
    ingest_document(
        args.document_path,
        data_dir=args.data_dir,
        chunk_mode=args.chunk_mode,
        batch_size=args.batch_size,
        max_chunks_per_level=args.max_chunks,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main()