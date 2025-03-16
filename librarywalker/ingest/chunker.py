#!/usr/bin/env python3

import re
import os
from typing import List, Dict, Any, Tuple, Optional

from ..utils import count_tokens
from ..config import MAX_EMBEDDING_TOKENS, BASE_CHUNK_SIZE

def split_into_base_chunks(text: str, mode: str = BASE_CHUNK_SIZE, 
                        max_tokens: int = 500) -> List[str]:
    """
    Split text into base chunks (Level 0) based on specified mode.
    Ensures chunks don't exceed the max_tokens limit.
    
    Args:
        text: The input text to split
        mode: Splitting mode - "sentence", "paragraph", or "line"
        max_tokens: Maximum tokens per chunk (default 500)
        
    Returns:
        List of base chunk strings
    """
    if not text:
        return []
    
    # For extremely large texts, use a generator-based approach to save memory
    if len(text) > 10_000_000:  # For texts larger than ~10MB
        return _split_large_text(text, mode)
        
    # First split according to the specified mode
    if mode == "sentence":
        # Basic sentence splitting (could be improved with NLP)
        raw_chunks = re.split(r'(?<=[.!?])\s+', text)
        raw_chunks = [s.strip() for s in raw_chunks if s.strip()]
    
    elif mode == "paragraph":
        raw_chunks = re.split(r'\n\s*\n', text)
        raw_chunks = [p.strip() for p in raw_chunks if p.strip()]
    
    elif mode == "line":
        raw_chunks = text.split('\n')
        raw_chunks = [l.strip() for l in raw_chunks if l.strip()]
    
    else:
        # Default to sentence
        raw_chunks = re.split(r'(?<=[.!?])\s+', text)
        raw_chunks = [s.strip() for s in raw_chunks if s.strip()]
    
    # Now ensure each chunk is within token limits
    final_chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for chunk in raw_chunks:
        chunk_tokens = count_tokens(chunk)
        
        # If a single chunk is already too large, split it further
        if chunk_tokens > max_tokens:
            if current_chunk:
                final_chunks.append(current_chunk)
                current_chunk = ""
                current_tokens = 0
                
            # Split large chunks by sentences if possible
            if mode != "sentence":
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                sub_chunk = ""
                sub_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = count_tokens(sentence)
                    
                    # If even a single sentence is too large, we need to split it by words
                    if sentence_tokens > max_tokens:
                        if sub_chunk:
                            final_chunks.append(sub_chunk)
                            sub_chunk = ""
                            sub_tokens = 0
                            
                        # Split by a fixed number of words
                        words = sentence.split()
                        for i in range(0, len(words), 50):  # 50 words per chunk
                            word_chunk = " ".join(words[i:i+50])
                            if word_chunk:
                                final_chunks.append(word_chunk)
                    
                    # If this sentence can be added to the current sub-chunk
                    elif sub_tokens + sentence_tokens <= max_tokens:
                        if sub_chunk:
                            sub_chunk += " " + sentence
                        else:
                            sub_chunk = sentence
                        sub_tokens += sentence_tokens
                    
                    # Start a new sub-chunk
                    else:
                        if sub_chunk:
                            final_chunks.append(sub_chunk)
                        sub_chunk = sentence
                        sub_tokens = sentence_tokens
                
                # Add any remaining sub-chunk
                if sub_chunk:
                    final_chunks.append(sub_chunk)
            else:
                # If we're already in sentence mode, split by words
                words = chunk.split()
                for i in range(0, len(words), 50):  # 50 words per chunk 
                    word_chunk = " ".join(words[i:i+50])
                    if word_chunk:
                        final_chunks.append(word_chunk)
        
        # If adding this chunk would exceed the token limit, start a new chunk
        elif current_tokens + chunk_tokens > max_tokens:
            if current_chunk:
                final_chunks.append(current_chunk)
            current_chunk = chunk
            current_tokens = chunk_tokens
        
        # Add to the current chunk
        else:
            if current_chunk:
                current_chunk += " " + chunk
            else:
                current_chunk = chunk
            current_tokens += chunk_tokens
    
    # Add the last chunk if there's any
    if current_chunk:
        final_chunks.append(current_chunk)
    
    return final_chunks
        
def _split_large_text(text: str, mode: str, max_tokens: int = 500) -> List[str]:
    """
    Memory-efficient splitting for very large texts.
    Also ensures chunks don't exceed the max_tokens limit.
    """
    print("Using memory-efficient text splitting for large document...")
    
    raw_chunks = []
    # Process text in smaller segments
    segment_size = 1_000_000  # Process ~1MB at a time
    
    for i in range(0, len(text), segment_size):
        segment = text[i:i+segment_size]
        
        if mode == "sentence":
            # Find complete sentences in this segment
            if i > 0:  # For segments after the first one, find the first sentence boundary
                first_boundary = re.search(r'(?<=[.!?])\s+', segment)
                if first_boundary:
                    # Start from the first complete sentence
                    start_pos = first_boundary.end()
                    segment = segment[start_pos:]
                    
            sentences = re.split(r'(?<=[.!?])\s+', segment)
            raw_chunks.extend([s.strip() for s in sentences if s.strip()])
            
        elif mode == "paragraph":
            paragraphs = re.split(r'\n\s*\n', segment)
            raw_chunks.extend([p.strip() for p in paragraphs if p.strip()])
            
        elif mode == "line":
            lines = segment.split('\n')
            raw_chunks.extend([l.strip() for l in lines if l.strip()])
            
        # Free memory
        del segment
        
        print(f"  Split progress: processed {min(i+segment_size, len(text))/len(text)*100:.1f}%")
    
    # Ensure chunks don't exceed token limits
    print("Ensuring all chunks are within token limits...")
    final_chunks = []
    current_chunk = ""
    current_tokens = 0
    
    chunk_batch_size = 1000  # Process chunks in batches to save memory
    for chunk_idx in range(0, len(raw_chunks), chunk_batch_size):
        chunk_batch = raw_chunks[chunk_idx:chunk_idx + chunk_batch_size]
        
        for chunk in chunk_batch:
            chunk_tokens = count_tokens(chunk)
            
            # If a single chunk is already too large, split it further
            if chunk_tokens > max_tokens:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0
                    
                # Split large chunks by sentences if possible
                if mode != "sentence":
                    sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    
                    sub_chunk = ""
                    sub_tokens = 0
                    
                    for sentence in sentences:
                        sentence_tokens = count_tokens(sentence)
                        
                        # If even a single sentence is too large, we need to split it by words
                        if sentence_tokens > max_tokens:
                            if sub_chunk:
                                final_chunks.append(sub_chunk)
                                sub_chunk = ""
                                sub_tokens = 0
                                
                            # Split by a fixed number of words
                            words = sentence.split()
                            for i in range(0, len(words), 50):  # 50 words per chunk
                                word_chunk = " ".join(words[i:i+50])
                                if word_chunk:
                                    final_chunks.append(word_chunk)
                        
                        # If this sentence can be added to the current sub-chunk
                        elif sub_tokens + sentence_tokens <= max_tokens:
                            if sub_chunk:
                                sub_chunk += " " + sentence
                            else:
                                sub_chunk = sentence
                            sub_tokens += sentence_tokens
                        
                        # Start a new sub-chunk
                        else:
                            if sub_chunk:
                                final_chunks.append(sub_chunk)
                            sub_chunk = sentence
                            sub_tokens = sentence_tokens
                    
                    # Add any remaining sub-chunk
                    if sub_chunk:
                        final_chunks.append(sub_chunk)
                else:
                    # If we're already in sentence mode, split by words
                    words = chunk.split()
                    for i in range(0, len(words), 50):  # 50 words per chunk
                        word_chunk = " ".join(words[i:i+50])
                        if word_chunk:
                            final_chunks.append(word_chunk)
            
            # If adding this chunk would exceed the token limit, start a new chunk
            elif current_tokens + chunk_tokens > max_tokens:
                if current_chunk:
                    final_chunks.append(current_chunk)
                current_chunk = chunk
                current_tokens = chunk_tokens
            
            # Add to the current chunk
            else:
                if current_chunk:
                    current_chunk += " " + chunk
                else:
                    current_chunk = chunk
                current_tokens += chunk_tokens
        
        # Free memory
        del chunk_batch
    
    # Add the last chunk if there's any
    if current_chunk:
        final_chunks.append(current_chunk)
    
    print(f"Finished processing. Created {len(final_chunks)} chunks within token limits.")
    return final_chunks

def make_chunk(text: str, level: int, index: int, 
              start_pos: int, end_pos: int, 
              document_id: str,
              source_chunks: List[Dict] = None) -> Dict[str, Any]:
    """
    Create a chunk with metadata.
    
    Args:
        text: The chunk text
        level: Chunk level (0, 1, 2, etc.)
        index: Chunk index within its level
        start_pos: Starting position in document
        end_pos: Ending position in document
        document_id: Document identifier
        source_chunks: For level >= 1, the list of level 0 chunks that make up this chunk
    
    Returns:
        Dictionary with chunk text and metadata
    """
    token_count = count_tokens(text)
    
    # Basic metadata
    metadata = {
        "document_id": document_id,
        "level": level,
        "chunk_index": f"L{level}-{index}",
        "range_in_document": (start_pos, end_pos),
        "token_count": token_count,
    }
    
    # For higher level chunks, track which level 0 chunks make up this chunk
    if source_chunks and level >= 1:
        source_indices = [c["index"] for c in source_chunks]
        source_ids = [c["metadata"]["chunk_index"] for c in source_chunks]
        
        # Store source chunk information
        metadata["source_chunk_indices"] = source_indices
        metadata["source_chunk_ids"] = source_ids
        
        # Count how many level 0 chunks are in this chunk
        metadata["l0_chunk_count"] = len(source_chunks)
        
        # Verify source chunks are consecutive for level >= 1
        is_consecutive = all(source_indices[i+1] == source_indices[i] + 1 
                            for i in range(len(source_indices)-1))
        
        if not is_consecutive:
            print(f"Warning: Non-consecutive source chunks in {metadata['chunk_index']}: {source_ids}")
    
    return {
        "text": text,
        "level": level,
        "index": index,
        "metadata": metadata
    }

def are_contiguous(chunk_a: Dict[str, Any], chunk_b: Dict[str, Any]) -> bool:
    """Check if two chunks are contiguous in the original document."""
    # Check if both chunks are from the same level
    if chunk_a["level"] != chunk_b["level"]:
        return False
        
    # Check if chunks are sequential based on index
    if chunk_b["index"] != chunk_a["index"] + 1:
        return False
    
    # Check if ranges are contiguous
    _, end_a = chunk_a["metadata"]["range_in_document"]
    start_b, _ = chunk_b["metadata"]["range_in_document"]
    
    # Debug if ranges don't match - should be rare
    if end_a != start_b:
        print(f"  Non-contiguous chunks found: {chunk_a['metadata']['chunk_index']} (end:{end_a}) and {chunk_b['metadata']['chunk_index']} (start:{start_b})")
    
    # At level 1 and above, we can be more lenient about contiguity
    # This allows better chunking at higher levels when chunks might not be perfectly contiguous
    if chunk_a["level"] >= 1:
        # Allow a small gap or overlap between chunks at higher levels
        return True
    
    return end_a == start_b

def merge_ranges(chunk_a: Dict[str, Any], chunk_b: Dict[str, Any]) -> Tuple[int, int]:
    """Merge document ranges of two chunks."""
    start_a, _ = chunk_a["metadata"]["range_in_document"]
    _, end_b = chunk_b["metadata"]["range_in_document"]
    return (start_a, end_b)

def extract_pdf_page(pdf_path, page_num):
    """Extract text from a single PDF page (for parallel processing)."""
    import PyPDF2
    
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            return text if text and text.strip() else ""
    except Exception as e:
        print(f"Error extracting page {page_num}: {str(e)}")
        return ""

def read_document(document_path: str, max_workers: int = 10) -> str:
    """
    Read document and extract text based on file type.
    Supports text files and PDFs with parallel processing.
    
    Args:
        document_path: Path to the document file
        max_workers: Maximum number of parallel workers for PDF extraction
        
    Returns:
        Extracted text content
    """
    file_extension = os.path.splitext(document_path)[1].lower()
    
    if file_extension == '.pdf':
        try:
            # Import PyPDF2 here to avoid dependency for those who don't need PDF support
            import PyPDF2
            
            # Get total page count
            with open(document_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)
            
            # Log progress for large PDFs
            if total_pages > 100:
                print(f"Extracting text from {total_pages} pages using parallel processing...")
            
            # For small PDFs, don't use multiprocessing
            if total_pages < 20:
                text_parts = []
                with open(document_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page_num in range(total_pages):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                return "\n\n".join(text_parts)
            
            # For large PDFs, use multiprocessing
            # Process pages in parallel using multiprocessing
            page_nums = list(range(total_pages))
            
            # Use a process pool to extract text from pages in parallel
            with multiprocessing.Pool(processes=min(max_workers, os.cpu_count())) as pool:
                # Create a partial function with the PDF path
                extract_fn = partial(extract_pdf_page, document_path)
                
                # Process pages in chunks to show progress
                batch_size = 50
                all_text_parts = []
                
                for i in range(0, len(page_nums), batch_size):
                    batch_nums = page_nums[i:i+batch_size]
                    
                    # Process this batch
                    text_parts = pool.map(extract_fn, batch_nums)
                    
                    # Filter out empty strings
                    valid_parts = [text for text in text_parts if text]
                    all_text_parts.extend(valid_parts)
                    
                    # Show progress
                    if total_pages > 100:
                        end_idx = min(i + batch_size, total_pages)
                        print(f"  Progress: {end_idx}/{total_pages} pages ({end_idx/total_pages*100:.1f}%)")
            
            # Join all text parts
            result = "\n\n".join(all_text_parts)
            
            if total_pages > 100:
                print(f"Finished extracting text from {total_pages} pages.")
                print(f"Extracted approximately {len(result) // 5} words.")
                
            return result
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return f"Error extracting text from {os.path.basename(document_path)}: {str(e)}"
    else:
        # Default text file handling
        with open(document_path, 'r', encoding='utf-8') as f:
            return f.read()

import multiprocessing
from functools import partial

def process_chunk_subset(subset_data):
    """
    Process a subset of chunks in parallel, creating multi-level chunks.
    
    Args:
        subset_data: Tuple of (level_chunks, start_idx, end_idx, new_level, document_id, index_offset)
        
    Returns:
        List of newly created chunks and the last index used
    """
    level_chunks, start_idx, end_idx, new_level, document_id, index_offset = subset_data
    subset = level_chunks[start_idx:end_idx]
    newly_created = []
    index_counter = index_offset
    
    # Track chunks we've already created to avoid duplicates across window sizes
    created_ranges = set()
    
    # Try to create multi-size chunks (not just pairs)
    # Loop from largest window size down to pairs
    for window_size in range(min(8, len(subset)), 1, -1):
        # For each possible starting position
        for i in range(len(subset) - window_size + 1):
            window_chunks = subset[i:i+window_size]
            
            # Check if chunks in window are contiguous
            is_contiguous = True
            for j in range(len(window_chunks) - 1):
                if not are_contiguous(window_chunks[j], window_chunks[j+1]):
                    is_contiguous = False
                    break
                    
            if not is_contiguous:
                continue
                
            # Get the range key for this window
            start_pos = window_chunks[0]["metadata"]["range_in_document"][0]
            end_pos = window_chunks[-1]["metadata"]["range_in_document"][1]
            range_key = (start_pos, end_pos)
            
            # Skip if we've already created a chunk covering this exact range
            if range_key in created_ranges:
                continue
                
            # Try combining all chunks in the window
            combined_text = " ".join([c["text"] for c in window_chunks])
            
            # Check if it fits in the embedding limit
            token_count = count_tokens(combined_text)
            if token_count <= MAX_EMBEDDING_TOKENS:
                # Create new chunk
                merged_chunk = make_chunk(
                    text=combined_text,
                    level=new_level,
                    index=index_counter,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    document_id=document_id
                )
                newly_created.append(merged_chunk)
                index_counter += 1
                
                # Mark this range as created
                created_ranges.add(range_key)
    
    # Also add simple adjacent pairs if we didn't catch them in the window logic
    for i in range(len(subset) - 1):
        c1 = subset[i]
        c2 = subset[i+1]
        
        if are_contiguous(c1, c2):
            # Get the range key for this pair
            start_pos = c1["metadata"]["range_in_document"][0]
            end_pos = c2["metadata"]["range_in_document"][1]
            range_key = (start_pos, end_pos)
            
            # Skip if we've already created a chunk covering this exact range
            if range_key in created_ranges:
                continue
                
            # Try combining
            combined_text = f"{c1['text']} {c2['text']}"
            
            # Check if combined text fits in embedding token limit
            token_count = count_tokens(combined_text)
            if token_count <= MAX_EMBEDDING_TOKENS:
                # Create new chunk
                merged_chunk = make_chunk(
                    text=combined_text,
                    level=new_level,
                    index=index_counter,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    document_id=document_id
                )
                newly_created.append(merged_chunk)
                index_counter += 1
                
                # Mark this range as created
                created_ranges.add(range_key)
    
    return newly_created, index_counter


def create_multi_level_chunks(document_path: str, 
                             chunk_mode: str = BASE_CHUNK_SIZE,
                             max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Create multi-level chunks from a document using parallelization.
    
    Each level N contains all possible contiguous combinations of N+1 level 0 chunks
    that fit within the token limit:
    - Level 0: Base chunks (sentences, paragraphs, etc.)
    - Level 1: Every possible pair of contiguous level 0 chunks
    - Level 2: Every possible triplet of contiguous level 0 chunks
    - Level 3: Every possible group of 4 contiguous level 0 chunks
    - And so on until reaching the token limit
    
    Args:
        document_path: Path to the document file
        chunk_mode: Mode for base chunk splitting
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of chunks with metadata at all levels
    """
    # Get document contents and ID (filename)
    text = read_document(document_path, max_workers=max_workers)
    document_id = os.path.basename(document_path)
    
    # Split into base chunks (Level 0)
    base_chunks_text = split_into_base_chunks(text, chunk_mode)
    
    all_chunks = []
    
    # Create Level 0 chunks with metadata
    print(f"Creating {len(base_chunks_text)} base chunks...")
    
    # Process base chunks (Level 0)
    level0_chunks = []
    for i, chunk_text in enumerate(base_chunks_text):
        # Skip if chunk is already too large for embedding
        if count_tokens(chunk_text) > MAX_EMBEDDING_TOKENS:
            print(f"Warning: Base chunk {i} exceeds token limit and will be skipped.")
            continue
            
        chunk = make_chunk(
            text=chunk_text,
            level=0,
            index=i,
            start_pos=i,  # Using index as position for simplicity
            end_pos=i+1,
            document_id=document_id
        )
        level0_chunks.append(chunk)
    
    # Add all level 0 chunks to our main collection
    all_chunks.extend(level0_chunks)
    print(f"  Processed {len(level0_chunks)} base chunks")
    
    # Now create higher levels directly from level 0 chunks
    index_counter = len(level0_chunks)  # Start index counter after level 0 chunks
    max_level = 20  # Set a high limit, but we might not reach it
    
    # Store created range pairs to avoid duplicates
    created_ranges = set()
    
    # For each level N, we're creating chunks with N+1 consecutive level 0 chunks
    for level in range(1, max_level):
        # For level N, the window size is N+1 level 0 chunks
        window_size = level + 1
        
        print(f"Building level {level} chunks (window size: {window_size})...")
        
        # If we don't have enough level 0 chunks for this window size, we're done
        if window_size > len(level0_chunks):
            print(f"  Not enough level 0 chunks for level {level} (need {window_size}, have {len(level0_chunks)})")
            break
        
        newly_created = []
        
        # For each possible starting position in level 0
        for i in range(len(level0_chunks) - window_size + 1):
            window_chunks = level0_chunks[i:i+window_size]
            
            # Check if all chunks in window are contiguous
            is_contiguous = True
            for j in range(len(window_chunks) - 1):
                # For level 0 chunks, they're contiguous if their indices are sequential
                if window_chunks[j+1]["index"] != window_chunks[j]["index"] + 1:
                    is_contiguous = False
                    break
            
            if not is_contiguous:
                continue
                
            # Get the range key for this window
            start_pos = window_chunks[0]["metadata"]["range_in_document"][0]
            end_pos = window_chunks[-1]["metadata"]["range_in_document"][1]
            range_key = (start_pos, end_pos)
            
            # Skip if we've already created a chunk with this exact range
            if range_key in created_ranges:
                continue
                
            # Try combining all chunks in the window
            combined_text = " ".join([c["text"] for c in window_chunks])
            
            # Check if it fits in the embedding limit
            token_count = count_tokens(combined_text)
            if token_count <= MAX_EMBEDDING_TOKENS:
                # Create new chunk with source tracking
                merged_chunk = make_chunk(
                    text=combined_text,
                    level=level,
                    index=index_counter,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    document_id=document_id,
                    source_chunks=window_chunks  # Track the level 0 chunks that make up this chunk
                )
                newly_created.append(merged_chunk)
                created_ranges.add(range_key)
                index_counter += 1
        
        # If no new chunks were created at this level, we're done
        if not newly_created:
            print(f"  No new chunks created at level {level}, stopping")
            break
            
        # Add new chunks to our collection
        all_chunks.extend(newly_created)
        
        # Calculate and display stats
        token_counts = [chunk["metadata"]["token_count"] for chunk in newly_created]
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        
        print(f"  Created {len(newly_created)} chunks at level {level}")
        print(f"  Level {level} stats: Min={min_tokens}, Avg={avg_tokens:.1f}, Max={max_tokens} tokens")
        print(f"  {(max_tokens/MAX_EMBEDDING_TOKENS)*100:.1f}% of max embedding capacity used")
        
        # Print more details about the longest chunks
        longest_chunks = sorted(newly_created, key=lambda x: x["metadata"]["token_count"], reverse=True)[:3]
        print(f"  Top 3 longest chunks at level {level}:")
        for i, chunk in enumerate(longest_chunks):
            print(f"    #{i+1}: {chunk['metadata']['token_count']} tokens, range: {chunk['metadata']['range_in_document']}")
        
        # Check if we're approaching the token limit
        if max_tokens >= 0.95 * MAX_EMBEDDING_TOKENS:
            print(f"  Reached {max_tokens}/{MAX_EMBEDDING_TOKENS} tokens at level {level}, next level would exceed limit")
            break
    
    print(f"Created {len(all_chunks)} chunks across {level+1} levels from '{document_id}'")
    return all_chunks