#!/usr/bin/env python3

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional

from .utils import ensure_dir

class VectorStore:
    """
    Vector store using FAISS and JSON with optimized storage.
    
    Only stores text content for L0 chunks to reduce storage requirements, 
    while maintaining embeddings for all levels.
    """
    
    def __init__(self, data_dir: str):
        """Initialize vector store with data directory."""
        self.data_dir = data_dir
        ensure_dir(data_dir)
        
        # Create output directory within data_dir for vector storage
        self.output_dir = os.path.join(data_dir, "output")
        ensure_dir(self.output_dir)
        
        self.index_path = os.path.join(self.output_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.output_dir, "metadata.json")
        self.l0_chunks_path = os.path.join(self.output_dir, "l0_chunks.json")  # Only store L0 chunks
        
        # Initialize or load index and metadata
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
                
            # Load L0 chunks if available
            if os.path.exists(self.l0_chunks_path):
                with open(self.l0_chunks_path, 'r') as f:
                    self.l0_chunks = json.load(f)
            else:
                # For backward compatibility with old format
                if os.path.exists(os.path.join(self.output_dir, "chunks.json")):
                    with open(os.path.join(self.output_dir, "chunks.json"), 'r') as f:
                        all_chunks = json.load(f)
                        # Extract just L0 chunks
                        self.l0_chunks = {}
                        for idx, metadata in enumerate(self.metadata):
                            if metadata.get('level', -1) == 0:
                                chunk_id = metadata.get('chunk_index')
                                if chunk_id:
                                    self.l0_chunks[chunk_id] = all_chunks[idx]
                    # Save in new format
                    with open(self.l0_chunks_path, 'w') as f:
                        json.dump(self.l0_chunks, f)
                else:
                    self.l0_chunks = {}
                
            # Build L0 chunk metadata mapping for efficient retrieval
            self.l0_metadata = self._build_l0_metadata_mapping()
        else:
            # Initialize with 1536 dimensions (for OpenAI text-embedding-ada-002) 
            # or 1536 for text-embedding-3-small (both same dimension)
            # Using IndexFlatL2 for exact (exhaustive) search - guarantees finding true top-k nearest neighbors
            self.index = faiss.IndexFlatL2(1536)
            self.metadata = []
            self.l0_chunks = {}  # Dictionary keyed by chunk_id
            self.l0_metadata = {}
            
    def _build_l0_metadata_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a mapping of level-0 chunk IDs to their metadata for efficient retrieval.
        
        Returns:
            Dictionary mapping L0 chunk IDs to metadata including text, token count, and document position
        """
        l0_metadata = {}
        self.l0_chunk_index = {}  # Maps L0 chunk IDs to their indices in metadata list
        
        # Iterate through all chunks
        for i, metadata in enumerate(self.metadata):
            if metadata.get('level', -1) == 0:
                chunk_id = metadata.get('chunk_index')
                if chunk_id:
                    # Store the index to quickly lookup this chunk's metadata later
                    self.l0_chunk_index[chunk_id] = i
                    
                    # Get the text from our L0 chunks dictionary
                    chunk_text = self.l0_chunks.get(chunk_id, "")
                    
                    # Store metadata for token counting and sorting
                    l0_metadata[chunk_id] = {
                        "text": chunk_text,
                        "token_count": metadata.get('token_count', 0),
                        "doc_index": metadata.get('range_in_document', (0, 0))[0]  # Use start position as index
                    }
        
        print(f"Built index of {len(l0_metadata)} level-0 chunks for efficient retrieval")
        return l0_metadata
        
    def get_l0_chunk(self, chunk_id: str) -> Tuple[str, Dict, float]:
        """
        Retrieve a level-0 chunk by its ID.
        
        Args:
            chunk_id: The ID of the level-0 chunk to retrieve
            
        Returns:
            Tuple of (chunk_text, metadata, default_score)
        """
        if chunk_id in self.l0_chunk_index and chunk_id in self.l0_chunks:
            idx = self.l0_chunk_index[chunk_id]
            return (self.l0_chunks[chunk_id], self.metadata[idx], 1.0)
        return None
    
    def add_embeddings(self, embeddings: List[List[float]], 
                     metadata_list: List[Dict[str, Any]],
                     chunks: List[str]) -> None:
        """
        Add embeddings and their metadata to the store.
        
        For storage efficiency, only L0 chunk texts are stored persistently.
        Higher level chunks are only represented by their embeddings and metadata.
        """
        if not embeddings:
            return
            
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Get current index size
        current_size = len(self.metadata)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store metadata
        self.metadata.extend(metadata_list)
        
        # Only store L0 chunk texts to save space
        for i, metadata in enumerate(metadata_list, start=current_size):
            if metadata.get('level', -1) == 0:
                chunk_id = metadata.get('chunk_index')
                if chunk_id:
                    # Store this L0 chunk's text
                    self.l0_chunks[chunk_id] = chunks[i - current_size]
                    
                    # Update our L0 index mapping
                    if hasattr(self, 'l0_chunk_index'):
                        self.l0_chunk_index[chunk_id] = i
                    
                    # Update our L0 metadata
                    if hasattr(self, 'l0_metadata'):
                        self.l0_metadata[chunk_id] = {
                            "text": chunks[i - current_size],
                            "token_count": metadata.get('token_count', 0),
                            "doc_index": metadata.get('range_in_document', (0, 0))[0]
                        }
        
        # Save to disk
        self._save()
        
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        Search for most similar vectors.
        
        Args:
            query_vector: The query vector to search for
            top_k: Number of results to return (will be limited by vector store size)
            
        Returns:
            List of (chunk_text, metadata, similarity_score) tuples.
            For L0 chunks, the actual text is provided.
            For higher-level chunks, an empty string is provided (to save memory).
        """
        query_vector_array = np.array([query_vector]).astype('float32')
        
        # Limit top_k to the size of the index
        actual_top_k = min(top_k, len(self.metadata))
        
        if actual_top_k < top_k:
            print(f"Note: Limited search to {actual_top_k} results (total chunks in vector store)")
        
        # Search the index - IndexFlatL2 performs exact search, guaranteed to find true top-k results
        distances, indices = self.index.search(query_vector_array, actual_top_k)
        
        # Map results to metadata and compute similarity score (1 - normalized distance)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # Valid index
                # Convert distance to similarity score (1 - normalized distance)
                # FAISS uses L2 distance, so we need to normalize
                distance = distances[0][i]
                # Simple normalization, may need adjustment for real-world use
                similarity = 1.0 / (1.0 + distance)
                
                # Get metadata
                metadata = self.metadata[idx]
                
                # Get chunk text - only available for L0 chunks
                chunk_text = ""
                if metadata.get('level', -1) == 0:
                    chunk_id = metadata.get('chunk_index')
                    if chunk_id and chunk_id in self.l0_chunks:
                        chunk_text = self.l0_chunks[chunk_id]
                
                results.append((
                    chunk_text,
                    metadata,
                    similarity
                ))
                
        return results
    
    def _save(self) -> None:
        """Save index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        with open(self.l0_chunks_path, 'w') as f:
            json.dump(self.l0_chunks, f)