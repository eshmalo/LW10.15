#!/usr/bin/env python3

import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
import threading
import queue

from ..config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBEDDING_COSTS

class RateLimiter:
    """
    Robust rate limiter for the OpenAI API with built-in time-based tracking and backoff.
    
    Using Tier 4 limits:
    - 10,000 RPM (Requests Per Minute)
    - 5,000,000 TPM (Tokens Per Minute)
    - 500,000,000 TPD (Tokens Per Day)
    
    For all embedding models (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
    """
    
    def __init__(self, tokens_per_minute: int = 5_000_000, requests_per_minute: int = 10_000):
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute
        
        # Use timestamp tracking for more accurate rate limiting
        self.token_timestamps = []
        self.request_timestamps = []
        
        # Track rate limit failures to implement adaptive backoff
        self.recent_failures = 0
        self.last_backoff_time = 0
        self.backoff_duration = 0
        
        # Use threading lock for thread-safety
        self.lock = threading.Lock()
    
    def _clear_expired_entries(self, current_time):
        """Clear expired entries from timestamp lists."""
        # Keep only entries from the last minute
        one_minute_ago = current_time - 60
        
        # Clear expired token timestamps
        self.token_timestamps = [t for t in self.token_timestamps if t > one_minute_ago]
        
        # Clear expired request timestamps
        self.request_timestamps = [t for t in self.request_timestamps if t > one_minute_ago]
    
    def _calculate_adaptive_wait_time(self, tokens_needed):
        """Calculate a dynamic wait time based on recent failures."""
        # Base wait time in seconds - just over a minute is sufficient
        base_wait = 61  # Just over a standard minute
        
        # If we've had recent failures, increase the backoff
        if self.recent_failures > 0:
            # Less aggressive exponential backoff with a maximum of 4 minutes
            backoff_factor = min(1.5 ** self.recent_failures, 4)
            wait_time = base_wait * backoff_factor
            
            # Smaller additional buffer proportional to tokens needed
            token_factor = tokens_needed / (self.tokens_per_minute * 0.2)  # Less conservative
            extra_buffer = min(token_factor * 5, 60)  # Cap at 1 minute extra
            
            return wait_time + extra_buffer
        
        return base_wait
    
    def register_failure(self):
        """Register a rate limit failure to increase future backoff."""
        with self.lock:
            self.recent_failures += 1
            
            # Cap at 5 for reasonable wait times
            if self.recent_failures > 5:
                self.recent_failures = 5
                
            # Record backoff
            self.last_backoff_time = time.time()
            self.backoff_duration = self._calculate_adaptive_wait_time(self.tokens_per_minute)
            
            # Return the backoff duration
            return self.backoff_duration
    
    def check_and_wait(self, token_count: int = 0):
        """
        Check if we're within rate limits and wait if necessary.
        
        Args:
            token_count: Number of tokens in the current request
        """
        current_time = time.time()
        
        # Use reasonable buffer
        # Add 20% buffer to token count and use 90% of capacity
        buffered_token_count = int(token_count * 1.2)
        max_tokens_safe = int(self.tokens_per_minute * 0.9)
        max_requests_safe = int(self.requests_per_minute * 0.9)
        
        with self.lock:
            # Clear expired entries
            self._clear_expired_entries(current_time)
            
            # Check if we're in backoff mode from a recent failure
            if self.backoff_duration > 0 and (current_time - self.last_backoff_time) < self.backoff_duration:
                remaining_backoff = self.backoff_duration - (current_time - self.last_backoff_time)
                if remaining_backoff > 1:  # Only report if significant time left
                    print(f"In backoff mode: waiting {remaining_backoff:.1f}s more after recent rate limit failure")
                    time.sleep(remaining_backoff)
                    
                    # Reset timestamps and backoff after waiting
                    self.token_timestamps = []
                    self.request_timestamps = []
                    self.backoff_duration = 0
                    current_time = time.time()
            
            # Count current usage
            current_tokens = len(self.token_timestamps)
            current_requests = len(self.request_timestamps)
            
            # Check token limit with a very conservative approach
            if current_tokens + buffered_token_count > max_tokens_safe:
                # Calculate dynamic wait time based on usage and history
                wait_time = self._calculate_adaptive_wait_time(buffered_token_count)
                
                print(f"Rate limit approaching: waiting {wait_time:.1f}s for token limit reset... " +
                      f"({current_tokens}/{self.tokens_per_minute} tokens used)")
                
                time.sleep(wait_time)
                
                # Reset after waiting
                self.token_timestamps = []
                self.request_timestamps = []
                current_time = time.time()
                
                # Reduce failure count after successful wait
                if self.recent_failures > 0:
                    self.recent_failures -= 1
            
            # Check request limit
            if current_requests >= max_requests_safe:
                wait_time = self._calculate_adaptive_wait_time(0)  # Token count doesn't matter here
                
                print(f"Request limit approaching: waiting {wait_time:.1f}s for request limit reset... " +
                      f"({current_requests}/{self.requests_per_minute} requests used)")
                
                time.sleep(wait_time)
                
                # Reset after waiting
                self.token_timestamps = []
                self.request_timestamps = []
                current_time = time.time()
                
                # Reduce failure count after successful wait
                if self.recent_failures > 0:
                    self.recent_failures -= 1
            
            # Record this request
            self.request_timestamps.append(current_time)
            
            # Record tokens - use the actual count for record keeping
            for _ in range(token_count):
                self.token_timestamps.append(current_time)

# Create a global rate limiter instance
rate_limiter = RateLimiter()

# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str], 
               batch_size: int = 30,  # Reduced for more reliable processing
               retry_limit: int = 5,  # Increased retry limit for resilience
               retry_delay: float = 2.0) -> List[List[float]]:
    """
    Embed texts using OpenAI API with batching, retries, and rate limiting.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to embed in a single API call
        retry_limit: Maximum number of retries on failure
        retry_delay: Initial delay between retries in seconds (will be doubled on each retry)
        
    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # More accurate token count for rate limiting
        # Using a conservative estimate (1 token per 3 chars to be safe)
        approx_token_count = sum(len(text) // 3 for text in batch)
        
        # Add a minimum token count per text to account for short texts
        min_tokens_per_text = 8
        approx_token_count += len(batch) * min_tokens_per_text
        
        # Wait if needed to respect rate limits
        rate_limiter.check_and_wait(approx_token_count)
        
        # Try to embed with retries
        for attempt in range(retry_limit):
            try:
                # Use the new OpenAI client interface
                response = client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=batch
                )
                
                # Extract embeddings from response (new format)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Log completion (only for large batches)
                if len(texts) > 100:
                    progress = min(i + batch_size, len(texts))
                    print(f"Embedding progress: {progress}/{len(texts)} texts ({progress/len(texts)*100:.1f}%)")
                
                # Success, break retry loop
                break
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error embedding batch {i//batch_size + 1}: {error_msg}")
                
                # Special handling for rate limit errors
                if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    # Register the failure with the rate limiter to trigger longer backoffs for future requests
                    wait_time = rate_limiter.register_failure()
                    
                    # Add additional backoff for this specific retry attempt - less aggressive
                    attempt_backoff = 60 if attempt == 0 else 90 * (attempt + 1)  # 60s, 180s, 270s, 360s, 450s
                    total_wait = wait_time + attempt_backoff
                    
                    print(f"Rate limit exceeded. Waiting {total_wait} seconds (attempt {attempt+1}/{retry_limit})...")
                    print(f"Future requests will use more conservative rate limiting.")
                    
                    # Reset the rate limiter's internal state
                    with rate_limiter.lock:
                        rate_limiter.token_timestamps = []
                        rate_limiter.request_timestamps = []
                    
                    time.sleep(total_wait)
                elif attempt < retry_limit - 1:
                    # Calculate delay with exponential backoff for other errors
                    delay = retry_delay * (2 ** attempt)
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to embed batch after {retry_limit} attempts")
                    # Return empty embeddings for this batch
                    all_embeddings.extend([[] for _ in range(len(batch))])
    
    return all_embeddings

import asyncio
import httpx
import time
from concurrent.futures import ThreadPoolExecutor

def estimate_embedding_cost(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Estimate the cost of embedding a list of chunks based on their token counts.
    
    Args:
        chunks: List of chunk objects with token counts in metadata
        
    Returns:
        Dictionary with cost details (total_cost, total_tokens, etc.)
    """
    # Get the cost per token for the current model
    # These are actual USD costs per token (not per million)
    cost_per_token = EMBEDDING_COSTS.get(OPENAI_EMBEDDING_MODEL, 0.00000010)  # Default if model not found
    
    # Calculate cost per million tokens for display purposes
    cost_per_million = cost_per_token * 1_000_000  # Cost per million tokens
    
    # Sum up all tokens
    total_tokens = sum(chunk["metadata"]["token_count"] for chunk in chunks)
    
    # Calculate cost (tokens * cost per token)
    estimated_cost = total_tokens * cost_per_token
    
    # Create dictionary with cost details
    cost_details = {
        "total_cost": estimated_cost,
        "total_tokens": total_tokens,
        "cost_per_million_tokens": cost_per_million,
        "model": OPENAI_EMBEDDING_MODEL,
        "num_chunks": len(chunks)
    }
    
    return cost_details

def embed_chunks_parallel(chunks: List[Dict[str, Any]], 
                        batch_size: int = 30,  # Smaller batch size for reliability
                        max_concurrent_requests: int = 4) -> List[Dict[str, Any]]:  # Very conservative for stable embedding
    """
    Embed a list of chunks with async parallel requests for faster embedding.
    
    Args:
        chunks: List of chunk objects with text field
        batch_size: Batch size for embedding API calls
        max_concurrent_requests: Maximum number of concurrent API requests
        
    Returns:
        Same chunks with embedding vectors added
    """
    try:
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Split texts into batches
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        # Create thread pool for concurrent embedding
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            # Submit tasks
            future_to_batch = {
                executor.submit(
                    embed_texts, 
                    batch,
                    batch_size=batch_size
                ): i for i, batch in enumerate(batches)
            }
            
            # Process results as they complete
            all_embeddings = [[] for _ in range(len(batches))]
            for future in future_to_batch:
                batch_idx = future_to_batch[future]
                try:
                    result = future.result()
                    all_embeddings[batch_idx] = result
                    print(f"Completed batch {batch_idx+1}/{len(batches)}")
                except Exception as e:
                    print(f"Error in batch {batch_idx+1}: {str(e)}")
                    # Return empty embeddings for this batch
                    all_embeddings[batch_idx] = [[] for _ in range(len(batches[batch_idx]))]
            
            # Flatten the embeddings list
            flattened_embeddings = []
            for batch in all_embeddings:
                flattened_embeddings.extend(batch)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, flattened_embeddings):
                chunk["embedding"] = embedding
            
            return chunks
    except Exception as e:
        print(f"Error in parallel embedding: {str(e)}")
        # Fallback to normal embedding
        return embed_chunks(chunks, batch_size)

def embed_chunks(chunks: List[Dict[str, Any]], 
                batch_size: int = 20) -> List[Dict[str, Any]]:
    """
    Embed a list of chunks and add the embeddings to each chunk.
    
    Args:
        chunks: List of chunk objects with text field
        batch_size: Batch size for embedding API calls
        
    Returns:
        Same chunks with embedding vectors added
    """
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Get embeddings
    embeddings = embed_texts(texts, batch_size=batch_size)
    
    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    return chunks