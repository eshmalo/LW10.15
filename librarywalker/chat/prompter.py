#!/usr/bin/env python3

import requests
import json
import time
from typing import List, Dict, Any, Tuple, Optional

from ..config import (
    ANTHROPIC_API_KEY, 
    ANTHROPIC_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    USE_OPENAI_FALLBACK
)

def format_context_from_chunks(chunks: List[Tuple[str, Dict, float]]) -> str:
    """
    Format retrieved chunks into a context section for Claude.
    
    The chunks are expected to be L0 chunks in document order,
    as produced by the coverage-based retrieval system.
    
    Args:
        chunks: List of (chunk_text, metadata, similarity_score) tuples
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant context found."
    
    context_parts = []
    
    # Get document ID from the first chunk (assuming all chunks are from the same document)
    doc_id = chunks[0][1].get('document_id', 'Unknown') if chunks else 'Unknown'
    
    # Add a header explaining the context format
    header = f"""RETRIEVED CONTEXT (from {doc_id})
The following text chunks are consecutive level-0 (base) chunks arranged in document order:
"""
    context_parts.append(header)
    
    # Include all chunks in sequence
    for i, (chunk_text, metadata, score) in enumerate(chunks):
        # Extract metadata fields
        chunk_index = metadata.get('chunk_index', 'unknown')
        doc_range = metadata.get('range_in_document', ('?', '?'))
        tokens = metadata.get('token_count', metadata.get('tokens', '?'))
        
        # Format the chunk header to be more concise
        chunk_header = f"[Chunk {i+1}: {chunk_index}]"
        
        # Add the chunk with minimal metadata
        context_parts.append(f"{chunk_header}\n{chunk_text}")
    
    # Add a separator at the end
    context_parts.append("END OF RETRIEVED CONTEXT")
    
    return "\n\n".join(context_parts)

def create_claude_prompt(query: str, context: str) -> Dict[str, Any]:
    """
    Create a complete prompt for Claude API with appropriate byte limit handling.
    
    Args:
        query: User's question
        context: Formatted context from chunks
        
    Returns:
        Claude API message format
    """
    system_prompt = """You are Promptchan, an AI assistant with access to special context. 
Answer questions based ONLY on the context provided below. 
If the context doesn't contain the information needed, say "I don't have enough information about that in my context." 
Be concise and accurate in your responses."""

    # Format the prompt according to Claude's API requirements
    # System message is a top-level parameter in Claude's API, not a message role
    user_message = f"""Here is the relevant context to help answer my question:

{context}

My question is: {query}"""

    prompt = {
        "model": ANTHROPIC_MODEL,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_message
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.0
    }
    
    # Check if the prompt exceeds Claude's byte limit (9MB)
    MAX_BYTES = 9 * 1024 * 1024  # 9MB
    import json
    prompt_bytes = len(json.dumps(prompt).encode('utf-8'))
    
    if prompt_bytes > MAX_BYTES:
        print(f"Warning: Prompt exceeds Claude's byte limit ({prompt_bytes} > {MAX_BYTES})")
        
        # Calculate how much we need to reduce 
        # Use a 10% safety margin (0.9) to ensure we're under the limit
        reduction_ratio = 0.9 * MAX_BYTES / prompt_bytes
        
        # Identify context portions between chunk headers
        import re
        
        # Split the context into chunks based on the chunk headers
        chunk_pattern = r'(\[Chunk \d+: L\d+-\d+\]\n)'
        chunk_parts = re.split(chunk_pattern, context)
        
        # Regroup into header-content pairs
        chunks = []
        for i in range(1, len(chunk_parts), 2):
            if i+1 < len(chunk_parts):
                header = chunk_parts[i]
                content = chunk_parts[i+1]
                chunks.append((header, content))
        
        # If splitting didn't work, fall back to simple truncation
        if not chunks:
            max_context_chars = int(len(context) * reduction_ratio)
            truncated_context = context[:max_context_chars] + "\n[Content truncated to fit API limits]"
            
            # Rebuild the user message with truncated context
            user_message = f"""Here is the relevant context to help answer my question:

{truncated_context}

My question is: {query}"""
            
            # Update the prompt
            prompt["messages"][0]["content"] = user_message
            
            # Log that we truncated
            print(f"Truncated context from {len(context)} to {len(truncated_context)} characters")
        else:
            # Calculate how many chunks to keep
            chunks_to_keep = max(1, int(len(chunks) * reduction_ratio))
            
            # Select chunks to keep - prefer keeping earlier chunks
            kept_chunks = chunks[:chunks_to_keep]
            
            # Rebuild context
            header_parts = context.split("\n\n", 2)  # Split to get the main header
            main_header = header_parts[0] + "\n\n" + header_parts[1] if len(header_parts) > 1 else ""
            
            rebuilt_context = main_header + "\n\n"
            for header, content in kept_chunks:
                rebuilt_context += header + content + "\n\n"
            
            # Add truncation note and footer
            rebuilt_context += f"[{len(chunks) - chunks_to_keep} chunks truncated to fit API limits]\n\n"
            rebuilt_context += "END OF RETRIEVED CONTEXT"
            
            # Rebuild the user message with truncated context
            user_message = f"""Here is the relevant context to help answer my question:

{rebuilt_context}

My question is: {query}"""
            
            # Update the prompt
            prompt["messages"][0]["content"] = user_message
            
            # Log what we did
            print(f"Truncated context from {len(chunks)} chunks to {chunks_to_keep} chunks")
        
        # Verify we're now under the limit
        new_prompt_bytes = len(json.dumps(prompt).encode('utf-8'))
        print(f"Updated prompt size: {new_prompt_bytes} bytes ({new_prompt_bytes/MAX_BYTES*100:.1f}% of limit)")
        
        # If we're still over the limit, do a more aggressive truncation
        if new_prompt_bytes > MAX_BYTES:
            print("Warning: Still exceeding byte limit after chunk truncation, applying additional reduction...")
            
            # Calculate a more aggressive reduction
            further_reduction = 0.85 * MAX_BYTES / new_prompt_bytes
            max_context_chars = int(len(prompt["messages"][0]["content"]) * further_reduction)
            
            # Simple truncation of the entire message
            truncated_msg = prompt["messages"][0]["content"][:max_context_chars] + "\n[Additional content truncated]"
            prompt["messages"][0]["content"] = truncated_msg
            
            # Final size check
            final_bytes = len(json.dumps(prompt).encode('utf-8'))
            print(f"Final prompt size after additional truncation: {final_bytes} bytes ({final_bytes/MAX_BYTES*100:.1f}% of limit)")
    
    return prompt

def call_openai_api(system_prompt: str, user_message: str) -> str:
    """
    Call OpenAI API as a fallback when Claude is unavailable.
    
    Args:
        system_prompt: The system prompt
        user_message: The user message
        
    Returns:
        OpenAI's response text
    """
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key not set for fallback"
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        payload = {
            "model": OPENAI_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 4000,
            "temperature": 0.0
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "Error: Unexpected response format from OpenAI API"
            
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return f"Error: OpenAI fallback failed - {str(e)}"

def call_claude_api(prompt: Dict[str, Any], max_retries: int = 3, retry_delay: int = 3) -> str:
    """
    Call Claude API with prompt and retry logic.
    
    Args:
        prompt: Claude API message format
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds (will increase with each retry)
        
    Returns:
        Claude's response text
    """
    # Debug output
    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not set")
        print("Please add your Anthropic API key to the .env file")
        if USE_OPENAI_FALLBACK and OPENAI_API_KEY:
            print("Falling back to OpenAI...")
            system_prompt = prompt.get("system", "")
            user_message = prompt.get("messages", [{}])[0].get("content", "")
            return call_openai_api(system_prompt, user_message)
        return "Error: No API keys available"
    
    # The API version for Claude 3.5 Sonnet
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"  # This version works for Claude 3.5
    }
    
    retries = 0
    while retries <= max_retries:
        try:
            if retries > 0:
                print(f"Retry attempt {retries} of {max_retries}...")
                
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=prompt,
                timeout=60  # Add timeout to prevent hanging
            )
            
            # Handle different response codes
            if response.status_code == 200:
                # Success
                result = response.json()
                
                # The Claude API returns content as a list of content blocks
                if "content" in result and isinstance(result["content"], list):
                    # Extract text from the first text block
                    for content_block in result["content"]:
                        if content_block.get("type") == "text":
                            return content_block.get("text", "Error: No text in response")
                    
                # Fallback in case the structure changes
                return "Error: Unexpected response format from Claude API"
                
            elif response.status_code == 529:
                # Overloaded error - retry after delay
                print(f"API Overloaded, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 2
                retries += 1
                continue
                
            else:
                # Other errors - show details
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()
                
        except Exception as e:
            if retries < max_retries:
                print(f"Error: {str(e)}, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 2
                retries += 1
                continue
            else:
                print(f"Error calling Claude API after {max_retries} retries: {str(e)}")
                if USE_OPENAI_FALLBACK and OPENAI_API_KEY:
                    print("Falling back to OpenAI...")
                    system_prompt = prompt.get("system", "")
                    user_message = prompt.get("messages", [{}])[0].get("content", "")
                    return call_openai_api(system_prompt, user_message)
                return f"Error: {str(e)}"
    
    # If we reach here, all retries failed
    if USE_OPENAI_FALLBACK and OPENAI_API_KEY:
        print("Maximum retries exceeded, falling back to OpenAI...")
        system_prompt = prompt.get("system", "")
        user_message = prompt.get("messages", [{}])[0].get("content", "")
        return call_openai_api(system_prompt, user_message)
    
    return "Error: Maximum retries exceeded, Claude API is currently unavailable."