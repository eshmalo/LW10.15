#!/usr/bin/env python3

import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple

from ..config import DATA_DIR, MAX_CONTEXT_TOKENS
from ..vector_store import VectorStore
from .retriever import retrieve_relevant_chunks
from .prompter import format_context_from_chunks, create_claude_prompt, call_claude_api

def log_full_run(context: str, query: str, chunks: List[Tuple[str, Dict, float]] = None, run_id: str = None) -> str:
    """
    Save the full run context to a log file with detailed chunk breakdown.
    
    Args:
        context: The full context being sent to Claude
        query: The user's query
        chunks: The retrieved chunks with metadata
        run_id: Optional run identifier
        
    Returns:
        Path to the log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate a unique filename with timestamp
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{run_id}.txt"
    filepath = os.path.join(logs_dir, filename)
    
    # Write the context and query to the log file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"QUERY: {query}\n\n")
        
        # Add chunk breakdown if available
        if chunks:
            f.write("CHUNK BREAKDOWN:\n")
            f.write("-" * 50 + "\n")
            
            # Organize chunks by document/file
            chunks_by_file = {}
            l0_chunks = []
            
            # Identify all L0 chunks
            for _, metadata, score in chunks:
                if metadata.get('level', -1) == 0:
                    doc_id = metadata.get('document_id', 'Unknown')
                    chunk_id = metadata.get('chunk_index', 'Unknown')
                    token_count = metadata.get('token_count', 0)
                    
                    # Extract the chunk number from the ID (e.g. "L0-123" -> 123)
                    try:
                        chunk_num = int(chunk_id.split('-')[1]) if '-' in chunk_id else 0
                    except ValueError:
                        chunk_num = 0
                    
                    if doc_id not in chunks_by_file:
                        chunks_by_file[doc_id] = []
                    
                    chunks_by_file[doc_id].append({
                        'chunk_id': chunk_id,
                        'token_count': token_count,
                        'chunk_num': chunk_num,
                        'score': score
                    })
                    l0_chunks.append(chunk_id)
            
            # Calculate summary statistics
            total_l0_chunks = len(l0_chunks)
            total_files = len(chunks_by_file)
            total_tokens = sum(sum(chunk['token_count'] for chunk in file_chunks) 
                              for file_chunks in chunks_by_file.values())
            
            # Calculate the number of sequences across all files
            all_sequences = 0
            for doc_chunks in chunks_by_file.values():
                # Sort chunks by their numerical order
                doc_chunks.sort(key=lambda x: x['chunk_num'])
                
                # Count sequences
                current_seq_length = 1
                for i in range(1, len(doc_chunks)):
                    if doc_chunks[i]['chunk_num'] != doc_chunks[i-1]['chunk_num'] + 1:
                        all_sequences += 1
                        current_seq_length = 1
                    else:
                        current_seq_length += 1
                
                # Count the last sequence
                if doc_chunks:
                    all_sequences += 1
            
            # Write summary statistics
            f.write(f"RETRIEVAL SUMMARY:\n")
            f.write(f"Total L0 chunks: {total_l0_chunks} from {total_files} file(s)\n")
            f.write(f"Total tokens: {total_tokens}\n")
            f.write(f"Total distinct sequences: {all_sequences}\n")
            
            # Calculate average chunks per sequence
            if all_sequences > 0:
                avg_chunks_per_seq = total_l0_chunks / all_sequences
                f.write(f"Average chunks per sequence: {avg_chunks_per_seq:.1f}\n")
            
            if total_l0_chunks > 1:
                # Calculate contiguity percentage
                contiguity = (total_l0_chunks - all_sequences) / (total_l0_chunks - 1) * 100
                f.write(f"Chunk contiguity: {contiguity:.1f}%\n")
            
            f.write("\n")
            
            # Write breakdown by file
            for doc_id, doc_chunks in chunks_by_file.items():
                f.write(f"Document: {doc_id}\n")
                f.write(f"  L0 chunks: {len(doc_chunks)}\n")
                
                # Calculate total tokens from this file
                total_doc_tokens = sum(chunk['token_count'] for chunk in doc_chunks)
                f.write(f"  Total tokens: {total_doc_tokens}\n")
                
                # Sort chunks by their numerical order in the document
                doc_chunks.sort(key=lambda x: x['chunk_num'])
                
                # Identify sequences of consecutive chunks
                sequences = []
                current_seq = [doc_chunks[0]]
                
                for i in range(1, len(doc_chunks)):
                    if doc_chunks[i]['chunk_num'] == doc_chunks[i-1]['chunk_num'] + 1:
                        # This chunk continues the sequence
                        current_seq.append(doc_chunks[i])
                    else:
                        # This chunk starts a new sequence
                        sequences.append(current_seq)
                        current_seq = [doc_chunks[i]]
                
                # Add the last sequence
                if current_seq:
                    sequences.append(current_seq)
                
                # Report sequences
                f.write(f"  Sequences: {len(sequences)}\n")
                
                for i, seq in enumerate(sequences):
                    start_chunk = seq[0]['chunk_id']
                    end_chunk = seq[-1]['chunk_id']
                    seq_length = len(seq)
                    seq_tokens = sum(chunk['token_count'] for chunk in seq)
                    
                    if seq_length == 1:
                        f.write(f"    Sequence {i+1}: Single chunk {start_chunk} ({seq_tokens} tokens)\n")
                    else:
                        f.write(f"    Sequence {i+1}: {start_chunk} to {end_chunk} ({seq_length} chunks, {seq_tokens} tokens)\n")
                
                # List all chunk IDs from this file
                chunk_ids = [chunk['chunk_id'] for chunk in doc_chunks]
                chunks_str = ", ".join(chunk_ids)
                f.write(f"  All chunks: {chunks_str}\n\n")
            
            f.write("-" * 50 + "\n\n")
        
        # Write full context
        f.write(f"CONTEXT:\n{context}\n")
    
    return filepath

def chat_loop(data_dir: str = DATA_DIR,
             max_tokens: int = MAX_CONTEXT_TOKENS,
             debug: bool = False,
             show_context: bool = False) -> None:
    """
    Main chat loop for Promptchan.
    
    Args:
        data_dir: Directory with vector database
        max_tokens: Maximum tokens for context
        debug: Enable debug output
        show_context: Show the context sent to Claude
    """
    # Initialize vector store
    try:
        vector_store = VectorStore(data_dir)
        print(f"Loaded vector store from {data_dir}")
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        print("You may need to ingest some documents first.")
        return
    
    print("\nPromptchan Chat Interface")
    print("Type 'exit' or 'quit' to end the conversation")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
            
        # Skip empty queries
        if not user_input.strip():
            continue
            
        try:
            # Retrieve relevant chunks
            print("Searching knowledge base...")
            chunks = retrieve_relevant_chunks(
                user_input, 
                vector_store, 
                max_tokens=max_tokens
            )
            
            # Format context
            context = format_context_from_chunks(chunks)
            
            # Log the full context to a file with chunk breakdown
            log_file = log_full_run(context, user_input, chunks)
            
            # Show context preview (beginning and end if too long)
            max_display_chars = 5000  # Reasonable terminal display limit
            print(f"\nContext saved to: {log_file}")
            print("\nContext sent to Claude (preview):")
            print("-" * 50)
            if len(context) > max_display_chars:
                # Show the beginning and end of context
                beginning = context[:max_display_chars//2]
                ending = context[-max_display_chars//2:]
                print(f"{beginning}\n\n[...TRUNCATED FOR DISPLAY - FULL CONTEXT IN LOG FILE...]\n\n{ending}")
            else:
                print(context)
            print("-" * 50)
            
            # Create prompt
            prompt = create_claude_prompt(user_input, context)
            
            # Show raw prompt if debug mode is enabled
            if debug:
                print("\nRaw prompt sent to Claude API:")
                print("-" * 50)
                import json
                print(json.dumps(prompt, indent=2))
                print("-" * 50)
            
            # Call Claude API
            print("Asking Claude...")
            response = call_claude_api(prompt)
            
            # Show response
            print("\nPromptchan:", response)
            
            # Show debug info if requested
            if debug:
                print("\nDebug Information:")
                print(f"Retrieved {len(chunks)} chunks")
                for i, (_, metadata, score) in enumerate(chunks):
                    print(f"  Chunk {i+1}: {metadata.get('document_id')} | " 
                         f"Level {metadata.get('level')} | "
                         f"Index {metadata.get('chunk_index')} | "
                         f"Score {score:.3f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Command-line interface for Promptchan chat."""
    parser = argparse.ArgumentParser(description="Promptchan Chat Interface")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory with vector database")
    parser.add_argument("--max-tokens", type=int, default=MAX_CONTEXT_TOKENS, 
                      help="Maximum tokens for context")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--show-context", action="store_true", help="Show the context sent to Claude")
    
    args = parser.parse_args()
    
    chat_loop(
        data_dir=args.data_dir,
        max_tokens=args.max_tokens,
        debug=args.debug,
        show_context=args.show_context
    )

if __name__ == "__main__":
    main()