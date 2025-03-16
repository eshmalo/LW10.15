#!/usr/bin/env python3

from openai import OpenAI
from typing import List, Dict, Any, Tuple
import difflib

from ..config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, MAX_CONTEXT_TOKENS
from ..utils import count_tokens
from ..vector_store import VectorStore

# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def embed_query(query: str) -> List[float]:
    """Embed the query using the same model as document chunks."""
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=[query]
    )
    return response.data[0].embedding

# Legacy overlap calculation functions (kept for reference but no longer used
# in the primary retrieval process, which now uses the CoverageManager approach)

def calculate_overlap(existing_texts: List[str], new_text: str, 
                    existing_metadata: List[Dict] = None, new_metadata: Dict = None) -> int:
    """
    LEGACY METHOD: Calculate token overlap between a new text and existing texts.
    
    NOTE: This function is kept for backward compatibility but is no longer used
    in the primary retrieval process. The retrieval now uses the CoverageManager 
    approach which tracks level-0 chunk IDs directly.
    
    Args:
        existing_texts: List of already included chunk texts
        new_text: Text of the chunk being considered
        existing_metadata: Metadata for existing chunks
        new_metadata: Metadata for the new chunk
        
    Returns:
        Estimated token overlap count
    """
    if not existing_texts or not new_text:
        return 0
        
    # If we have metadata with source chunk information, use it for precise overlap detection
    # Check for both source_chunk_indices and source_chunk_ids
    if existing_metadata and new_metadata and ('source_chunk_indices' in new_metadata or 'source_chunk_ids' in new_metadata):
        overlap_tokens = 0
        
        # Try both source_chunk_indices and source_chunk_ids
        new_source_indices = set(new_metadata.get('source_chunk_indices', []))
        new_source_ids = set(new_metadata.get('source_chunk_ids', []))
        
        # If no source identifiers (level 0 chunk), use text matching
        if not new_source_indices and not new_source_ids:
            return calculate_text_overlap(existing_texts, new_text)
            
        # Check each existing chunk's metadata for overlap with this new chunk
        for metadata in existing_metadata:
            # Try both source_chunk_indices and source_chunk_ids
            existing_source_indices = set(metadata.get('source_chunk_indices', []))
            existing_source_ids = set(metadata.get('source_chunk_ids', []))
            
            # If this is a level 0 chunk or has no source identifiers, skip
            if not existing_source_indices and not existing_source_ids:
                continue
                
            # Find overlapping chunks using either indices or ids
            overlapping_indices = set()
            if new_source_indices and existing_source_indices:
                overlapping_indices = new_source_indices.intersection(existing_source_indices)
            elif new_source_ids and existing_source_ids:
                overlapping_indices = new_source_ids.intersection(existing_source_ids)
            
            # If there are overlapping source chunks, calculate token overlap
            if overlapping_indices:
                # Use the actual text for more accurate token counting when possible
                chunk_token_count = metadata.get('token_count', 0)
                source_count = max(1, len(existing_source_indices or existing_source_ids))
                
                # Calculate the proportion of overlapping source chunks
                overlap_ratio = len(overlapping_indices) / source_count
                
                # Apply the ratio to get approximate token overlap
                chunk_overlap_tokens = int(chunk_token_count * overlap_ratio)
                overlap_tokens += chunk_overlap_tokens
        
        return int(overlap_tokens)
    
    # Fallback to text-based overlap estimation
    return calculate_text_overlap(existing_texts, new_text)
    
def calculate_text_overlap(existing_texts: List[str], new_text: str) -> int:
    """
    LEGACY METHOD: Calculate overlap using text similarity when metadata isn't available.
    
    This function is deprecated in favor of the CoverageManager approach that
    tracks level-0 chunk IDs directly.
    """
    # Combine all existing texts
    combined_text = " ".join(existing_texts)
    
    # Use difflib to find approximate matches
    matcher = difflib.SequenceMatcher(None, combined_text, new_text)
    matching_blocks = matcher.get_matching_blocks()
    
    # Estimate overlapping tokens (only count substantial matches)
    overlap_chars = sum(block.size for block in matching_blocks if block.size > 10)
    
    # Extract overlapping text for more accurate token counting
    overlap_text = ""
    for block in matching_blocks:
        if block.size > 10:  # Only consider substantial matches
            overlap_text += new_text[block.b:block.b+block.size]
    
    # Use the actual token counter instead of approximation
    if overlap_text:
        from ..utils import count_tokens
        overlap_tokens = count_tokens(overlap_text)
    else:
        # Fallback to approximation if no substantial overlap
        overlap_tokens = overlap_chars // 4
    
    return overlap_tokens

class CoverageManager:
    """
    Manages the coverage of level-0 chunks during retrieval.
    
    This class encapsulates the logic for tracking which L0 chunks are included,
    calculating token costs for new chunks, and making inclusion decisions.
    """
    
    def __init__(self, l0_metadata, max_tokens, metadata_token_cost=20):
        """
        Initialize the coverage manager.
        
        Args:
            l0_metadata: Dictionary mapping L0 chunk IDs to their metadata
            max_tokens: Maximum token budget for selection
            metadata_token_cost: Estimated token cost per chunk for metadata
        """
        self.l0_metadata = l0_metadata
        self.max_tokens = max_tokens
        self.metadata_token_cost = metadata_token_cost
        
        # Initialize tracking
        self.selected_l0_ids = set()
        self.used_tokens = 0
        
        # For metrics
        self.total_candidates = 0
        self.skipped_redundant = 0
        self.skipped_budget = 0
        self.selected_candidates = []
        self.rejected_candidates = []
    
    def get_candidate_l0_ids(self, metadata):
        """
        Extract the L0 chunk IDs associated with a candidate.
        
        Args:
            metadata: Candidate chunk metadata
            
        Returns:
            List of L0 chunk IDs
        """
        if metadata.get('level', 0) == 0:
            # For L0 chunks, just return their own ID
            return [metadata.get('chunk_index')]
        else:
            # For higher level chunks, return their source chunk IDs
            return metadata.get('source_chunk_ids', [])
            
    def filter_redundant_candidates(self, candidates):
        """
        Filter out candidates that are completely redundant (all L0 chunks already selected).
        
        Args:
            candidates: List of (text, metadata, score) tuples
            
        Returns:
            Filtered list of candidates
        """
        if not self.selected_l0_ids:
            # No L0 chunks selected yet, so all candidates are potentially useful
            return candidates
            
        filtered_candidates = []
        filtered_count = 0
        
        for text, metadata, score in candidates:
            # Get the L0 chunk IDs for this candidate
            l0_ids = self.get_candidate_l0_ids(metadata)
            
            # Check if all L0 chunks are already selected
            if all(l0_id in self.selected_l0_ids for l0_id in l0_ids):
                filtered_count += 1
                continue
                
            # This candidate has at least one new L0 chunk
            filtered_candidates.append((text, metadata, score))
            
        # Update metrics
        self.skipped_redundant += filtered_count
        self.total_candidates += filtered_count
        
        return filtered_candidates
    
    def process_candidate(self, metadata, score):
        """
        Process a candidate chunk to determine if it should be included.
        
        Args:
            metadata: Candidate chunk metadata
            score: Similarity score
            
        Returns:
            (included, new_l0_ids, new_tokens) tuple indicating if candidate adds new content
        """
        self.total_candidates += 1
        
        # Get this candidate's L0 chunk IDs
        candidate_l0_ids = self.get_candidate_l0_ids(metadata)
        
        # Find which L0 chunks are new (not already selected)
        new_l0_ids = [cid for cid in candidate_l0_ids if cid not in self.selected_l0_ids]
        
        # If all L0 chunks are already selected, skip this candidate entirely
        if not new_l0_ids:
            self.skipped_redundant += 1
            return False, [], 0
        
        # Calculate total tokens needed for the new L0 chunks
        new_tokens = sum(self.l0_metadata.get(cid, {}).get('token_count', 0) for cid in new_l0_ids)
        
        # Add metadata token cost
        total_new_cost = new_tokens + self.metadata_token_cost
        
        # Check if adding these new L0 chunks fits within our token budget
        if self.used_tokens + total_new_cost <= self.max_tokens:
            # Accept this candidate and update selected L0 IDs
            self.selected_l0_ids.update(new_l0_ids)
            self.used_tokens += total_new_cost
            
            # Store the candidate for debugging/metrics
            self.selected_candidates.append((metadata, score, new_l0_ids, new_tokens))
            
            # Return success
            return True, new_l0_ids, new_tokens
        else:
            # This candidate doesn't fit within our token budget
            self.skipped_budget += 1
            self.rejected_candidates.append((metadata, score, new_l0_ids, new_tokens))
            return False, new_l0_ids, new_tokens
    
    def get_final_chunks(self, vector_store):
        """
        Get the final set of L0 chunks in document order.
        
        Args:
            vector_store: Vector store with L0 chunk retrieval capability
            
        Returns:
            List of (text, metadata, score) tuples for selected L0 chunks
        """
        # Sort selected L0 chunks by document order
        final_l0_ids = sorted(self.selected_l0_ids, 
                             key=lambda cid: self.l0_metadata.get(cid, {}).get('doc_index', 0))
        
        # Build the final context using in-order L0 chunks
        final_chunks = []
        total_l0_tokens = 0
        
        # Get L0 chunks in document order using the vector store's helper method
        for i, l0_id in enumerate(final_l0_ids):
            # Get the chunk using our optimized helper
            chunk_tuple = vector_store.get_l0_chunk(l0_id)
            
            if chunk_tuple:
                # Unpack tuple values
                text, meta, _ = chunk_tuple
                # Use approximate score (descending order value based on position in selected list)
                score = 1.0 - (i / max(1, len(final_l0_ids)))
                
                final_chunks.append((text, meta, score))
                total_l0_tokens += self.l0_metadata.get(l0_id, {}).get('token_count', 0)
        
        return final_chunks, final_l0_ids, total_l0_tokens
    
    def print_metrics(self):
        """Print metrics about the coverage process."""
        num_chunks = len(self.selected_l0_ids)
        
        print(f"Final selection: {num_chunks} L0 chunks using {self.used_tokens} tokens")
        print(f"Candidate metrics: Considered {self.total_candidates}, "
              f"Skipped {self.skipped_redundant} due to redundancy, "
              f"Skipped {self.skipped_budget} due to budget")
        
        # Show coverage efficiency
        if self.total_candidates > 0:
            non_redundant_coverage = 100 * (num_chunks / self.total_candidates)
            print(f"Non-redundant coverage: {non_redundant_coverage:.1f}% of candidates contributed new content")


def retrieve_relevant_chunks(query: str, 
                            vector_store: VectorStore, 
                            max_tokens: int = MAX_CONTEXT_TOKENS,
                            initial_candidates: int = 1000,  # Set very high to consider many more candidates
                            prioritize_higher_levels: bool = True,
                            coverage_efficiency: bool = True) -> List[Tuple[str, Dict, float]]:
    """
    Retrieve the most relevant chunks for a query that fit within token budget.
    
    This implementation uses a "coverage-based" approach to ensure we only include
    complete sets of level-0 chunks, without redundancy across higher level chunks.
    
    Storage-optimized version: Works with vector store that only stores L0 chunk texts.
    
    Args:
        query: User's query text
        vector_store: Vector store instance
        max_tokens: Maximum tokens to include in context
        initial_candidates: Number of candidates to consider
        prioritize_higher_levels: If True, prioritize higher level chunks 
                                 (which contain more context) when similarity scores are close
        coverage_efficiency: If True, further boost chunks that provide more unique L0 coverage
                            relative to their token cost
        
    Returns:
        List of (chunk_text, metadata, similarity_score) tuples for L0 chunks
    """
    # Embed the query
    query_vector = embed_query(query)
    
    # Get candidate chunks - will include empty text for non-L0 chunks
    candidates = vector_store.search(query_vector, top_k=initial_candidates)
    
    # Track which L0 IDs we've seen (for coverage efficiency calculation)
    l0_metadata = vector_store.l0_metadata
    seen_l0_ids = set()
    
    # Enhanced scoring that considers coverage efficiency
    def calculate_score(item):
        chunk_text, metadata, similarity = item
        
        base_score = similarity
        
        # Add level boost if requested
        if prioritize_higher_levels:
            level = metadata.get('level', 0)
            level_boost = level * 0.01  # Small boost per level (1% per level)
            base_score += level_boost
            
        # Add coverage efficiency boost if requested
        if coverage_efficiency and metadata.get('level', 0) > 0:
            # Get source L0 chunks
            source_l0_ids = metadata.get('source_chunk_ids', [])
            
            # Count how many new L0 chunks this would add
            new_l0_count = sum(1 for l0_id in source_l0_ids if l0_id not in seen_l0_ids)
            
            # Only boost if this adds new coverage
            if new_l0_count > 0:
                # Calculate efficiency as new L0s divided by token cost
                token_count = metadata.get('token_count', 1)
                efficiency = new_l0_count / token_count
                
                # Apply a small boost based on efficiency (0.5% per new L0 per 100 tokens)
                efficiency_boost = efficiency * 50  # Scale for appropriate magnitude
                base_score += min(0.05, efficiency_boost)  # Cap the boost at 5%
        
        return base_score
    
    # Sort candidates by calculated score
    candidates.sort(key=calculate_score, reverse=True)
        
    # Print initial top candidates
    print(f"Top {min(10, len(candidates))} of {len(candidates)} candidates (exact search, guaranteed top-k):")
    for i, (_, metadata, score) in enumerate(candidates[:10]):
        level = metadata.get('level', 0)
        chunk_index = metadata.get('chunk_index', 'unknown')
        token_count = metadata.get('token_count', 0)
        # Show source chunks for higher-level chunks
        source_info = ""
        if level > 0 and 'source_chunk_ids' in metadata:
            num_sources = len(metadata['source_chunk_ids'])
            source_info = f", {num_sources} L0 chunks"
            
        print(f"  {i+1}. {chunk_index} (Level {level}{source_info}, {token_count} tokens, {score:.4f} score)")
    
    # Create a coverage manager to handle L0 chunk selection
    coverage = CoverageManager(
        l0_metadata=vector_store.l0_metadata,
        max_tokens=max_tokens
    )
    
    # Process candidates in batches, filtering redundant ones after each batch
    # This approach maintains the order of candidates by similarity while
    # efficiently skipping those that become redundant
    batch_size = 20  # Process candidates in batches for efficiency
    remaining_candidates = candidates.copy()
    
    while remaining_candidates:
        # Take the next batch of candidates
        batch = remaining_candidates[:batch_size]
        remaining_candidates = remaining_candidates[batch_size:]
        
        # Process each candidate in this batch
        for _, metadata, score in batch:
            included, new_l0_ids, new_tokens = coverage.process_candidate(metadata, score)
            
            # Print debug info for included candidates
            if included:
                level = metadata.get('level', 0)
                chunk_index = metadata.get('chunk_index', 'unknown')
                print(f"  Added {chunk_index} (Level {level}) with {len(new_l0_ids)} new L0 chunks ({new_tokens} tokens)")
                
        # Filter the remaining candidates to remove redundant ones
        if remaining_candidates:
            remaining_candidates = coverage.filter_redundant_candidates(remaining_candidates)
            print(f"  Remaining candidates after filtering: {len(remaining_candidates)}")
            
        # Stop early if we've reached the token budget or have no more candidates
        if coverage.used_tokens >= max_tokens * 0.95 or not remaining_candidates:
            if coverage.used_tokens >= max_tokens * 0.95:
                print("  Token budget nearly exhausted, stopping early")
            break
    
    # Get the final chunks in document order
    final_chunks, final_l0_ids, total_l0_tokens = coverage.get_final_chunks(vector_store)
    
    # Print summary metrics
    coverage.print_metrics()
    
    return final_chunks