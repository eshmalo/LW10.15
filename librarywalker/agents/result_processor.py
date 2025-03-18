#!/usr/bin/env python3
"""
Result Processing AI module for the LibraryWalker enhanced context management system.
This module evaluates retrieval quality and provides feedback for improving results.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from .. import config
from ..utils import get_num_tokens


class ResultProcessorAI:
    """
    OpenAI-based retrieval evaluation system.
    Evaluates retrieval quality, identifies redundancy, and determines stopping criteria.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the ResultProcessorAI.
        
        Args:
            api_key: OpenAI API key, defaults to config.OPENAI_API_KEY
            model: OpenAI model to use, defaults to config.OPENAI_CHAT_MODEL
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_CHAT_MODEL
        
        try:
            self.llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key
            )
        except Exception as e:
            print(f"Error initializing ResultProcessorAI: {e}")
            self.llm = None
            
        # Cache for evaluation results
        self.evaluation_cache = {}
    
    def detect_redundancy(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, List[str]]]:
        """
        Detect redundant content across chunks at a semantic level.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            List of redundancy groups with chunk IDs
        """
        if not chunks or len(chunks) <= 1:
            return []
            
        redundancy_groups = []
        
        # Simple method: Compare text similarity
        from difflib import SequenceMatcher
        
        # Track already grouped chunks
        grouped_chunks = set()
        
        for i in range(len(chunks)):
            if i in grouped_chunks:
                continue
                
            chunk_i = chunks[i]
            chunk_i_id = chunk_i.get("id", f"chunk_{i}")
            chunk_i_text = chunk_i.get("text", "")
            
            redundant_chunks = []
            
            for j in range(i + 1, len(chunks)):
                if j in grouped_chunks:
                    continue
                    
                chunk_j = chunks[j]
                chunk_j_id = chunk_j.get("id", f"chunk_{j}")
                chunk_j_text = chunk_j.get("text", "")
                
                # Calculate similarity
                similarity = SequenceMatcher(None, chunk_i_text, chunk_j_text).ratio()
                
                # If similarity is high, consider redundant
                if similarity > 0.7:  # Threshold can be adjusted
                    redundant_chunks.append(chunk_j_id)
                    grouped_chunks.add(j)
            
            if redundant_chunks:
                redundancy_groups.append({
                    "chunk_ids": [chunk_i_id] + redundant_chunks,
                    "description": f"Similar content about {chunk_i.get('metadata', {}).get('title', 'unknown topic')}"
                })
                grouped_chunks.add(i)
        
        return redundancy_groups
    
    def calculate_chunk_relevance(
        self, 
        query: str, 
        chunk: Dict[str, Any]
    ) -> float:
        """
        Calculate the relevance of a chunk to the query using simple heuristics.
        
        Args:
            query: The user's query
            chunk: The chunk to evaluate
            
        Returns:
            Relevance score between 0 and 1
        """
        if not chunk or "text" not in chunk:
            return 0.0
            
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            return 0.0
            
        # Simple keyword matching for a baseline score
        query_words = set(query.lower().split())
        
        # Remove stopwords
        stopwords = {"the", "a", "an", "in", "on", "at", "of", "to", "for", "with", "about", "is", "are"}
        query_words = query_words - stopwords
        
        # Count matching words
        match_count = 0
        for word in query_words:
            if word in chunk_text.lower():
                match_count += 1
                
        # Calculate basic relevance score
        if not query_words:
            return 0.5  # Default mid-level relevance if no meaningful query words
        
        basic_score = match_count / len(query_words)
        
        # Adjust score based on metadata if available
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "")
        
        if title and any(word in title.lower() for word in query_words):
            basic_score = min(1.0, basic_score + 0.2)  # Boost if title matches query
            
        return basic_score
    
    def recommend_chunk_positions(
        self, 
        chunks: List[Dict[str, Any]], 
        relevance_scores: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Recommend optimal positions for chunks to prevent "lost in the middle" effect.
        
        Args:
            chunks: List of retrieved chunks
            relevance_scores: Dictionary mapping chunk IDs to relevance scores
            
        Returns:
            Dictionary mapping chunk IDs to recommended positions
        """
        if not chunks:
            return {}
            
        positions = {}
        
        # Sort chunks by relevance
        chunk_ids_by_relevance = sorted(
            [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
            key=lambda chunk_id: relevance_scores.get(chunk_id, 0.0),
            reverse=True
        )
        
        # Most relevant chunks go at the beginning
        for i, chunk_id in enumerate(chunk_ids_by_relevance):
            if i < len(chunk_ids_by_relevance) // 3:  # Top third goes at beginning
                positions[chunk_id] = "beginning"
            elif i >= 2 * len(chunk_ids_by_relevance) // 3:  # Bottom third goes at end
                positions[chunk_id] = "end"
            else:  # Middle third stays in middle
                positions[chunk_id] = "middle"
        
        return positions
    
    def evaluate_chunks(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the retrieved chunks for quality and relevance.
        
        Args:
            query: The optimized query
            chunks: The retrieved chunks
            original_query: The original query if different from optimized
            
        Returns:
            Evaluation results
        """
        if not chunks:
            return {
                "quality_assessment": "insufficient",
                "stopping_decision": "continue",
                "token_adjustment": {
                    "current": 0,
                    "recommended": 2000
                },
                "chunk_analysis": [],
                "information_gaps": ["No relevant content retrieved"],
                "redundant_content": [],
                "strategy_adjustments": {
                    "parameter_changes": {"increase_token_limit": True},
                    "recommended_actions": ["try alternative query formulation"],
                    "fallback_strategy": "expand search parameters"
                }
            }
            
        # Calculate basic metrics without LLM
        relevance_scores = {}
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            relevance_scores[chunk_id] = self.calculate_chunk_relevance(
                original_query or query, chunk
            )
            
        redundant_content = self.detect_redundancy(chunks)
        
        # Calculate token counts
        current_tokens = sum(
            get_num_tokens(chunk.get("text", "")) for chunk in chunks
        )
        
        # Check for quoted phrases in the query
        query_to_check = original_query or query
        quoted_phrases = re.findall(r'"([^"]*)"', query_to_check)
        
        # Also check for non-Latin (Hebrew, Arabic, etc.) phrases, treating them as exact phrases to find
        non_latin_phrases = []
        words = query_to_check.split()
        for word in words:
            # Check if word contains non-Latin characters (likely Hebrew or Arabic)
            if any(ord(c) > 127 for c in word):
                non_latin_phrases.append(word)
        
        # Combine both types of phrases to search for
        all_phrases_to_find = quoted_phrases + non_latin_phrases
        exact_phrases_found = {phrase: False for phrase in all_phrases_to_find}

        # Check if any chunks contain these exact phrases
        if all_phrases_to_find:
            for chunk in chunks:
                chunk_text = chunk.get("text", "")
                for phrase in all_phrases_to_find:
                    if phrase.lower() in chunk_text.lower():
                        exact_phrases_found[phrase] = True
        
        # Basic quality assessment
        avg_relevance = sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0
        if avg_relevance > 0.7:
            quality = "sufficient"
            stopping_decision = "stop"
        elif avg_relevance > 0.4:
            quality = "partial"
            stopping_decision = "continue" if current_tokens < config.MAX_CONTEXT_TOKENS // 2 else "stop"
        else:
            quality = "insufficient"
            stopping_decision = "continue"
            
        # Adjust stopping decision based on exact phrase matching and token budget
        missing_phrases = [phrase for phrase, found in exact_phrases_found.items() if not found]
        if missing_phrases:
            stopping_decision = "continue"  # Continue searching if any phrase not found
            quality = "insufficient"  # Mark quality as insufficient to trigger continued search

        # Adjust based on token budget - continue if we've used less than 20% of our budget
        token_budget_percentage = (current_tokens / config.MAX_CONTEXT_TOKENS) * 100
        if token_budget_percentage < 20:
            stopping_decision = "continue"  # Continue when we've used very little of our token budget
            
        # Recommend positions
        positions = self.recommend_chunk_positions(chunks, relevance_scores)
        
        # Create chunk analysis
        chunk_analysis = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            relevance = relevance_scores.get(chunk_id, 0.0)
            
            # Determine redundancy level
            redundancy_level = "none"
            for group in redundant_content:
                if chunk_id in group["chunk_ids"] and len(group["chunk_ids"]) > 1:
                    redundancy_level = "high"
                    break
                    
            # Determine information value
            if relevance > 0.7:
                info_value = "high"
            elif relevance > 0.4:
                info_value = "medium"
            else:
                info_value = "low"
                
            chunk_analysis.append({
                "chunk_id": chunk_id,
                "relevance_score": relevance,
                "redundancy_level": redundancy_level,
                "information_value": info_value,
                "recommended_position": positions.get(chunk_id, "middle")
            })
            
        # Use LLM for more advanced evaluation if available
        if self.llm and query:
            result = self.evaluate_chunks_with_llm(query, chunks, chunk_analysis, original_query)
            
            # Re-check for phrases and low token usage to override LLM decision if needed
            if missing_phrases:
                result["stopping_decision"] = "continue"
                result["quality_assessment"] = "insufficient"
                result["information_gaps"].append(f"Could not find exact phrase(s): {', '.join(missing_phrases)}")
                
            if token_budget_percentage < 20:
                result["stopping_decision"] = "continue"  # Force continue when we've used little of token budget
                if "strategy_adjustments" in result and "recommended_actions" in result["strategy_adjustments"]:
                    result["strategy_adjustments"]["recommended_actions"].append("expand search with more content (low token usage)")
            
            return result
            
        # Basic evaluation without LLM
        return {
            "quality_assessment": quality,
            "stopping_decision": stopping_decision,
            "token_adjustment": {
                "current": current_tokens,
                "recommended": min(current_tokens * 2, config.MAX_CONTEXT_TOKENS) if quality == "insufficient" else current_tokens
            },
            "chunk_analysis": chunk_analysis,
            "information_gaps": (
                ["Possibly incomplete information based on low relevance scores"] if quality != "sufficient" else []
            ) + (
                [f"Could not find exact phrase(s): {', '.join(missing_phrases)}"] if missing_phrases else []
            ),
            "redundant_content": redundant_content,
            "strategy_adjustments": {
                "parameter_changes": {"increase_token_limit": quality != "sufficient"},
                "recommended_actions": (
                    ["try alternative query formulation"] if quality == "insufficient" else []
                ) + (
                    ["expand search with more content (low token usage)"] if token_budget_percentage < 20 else []
                ),
                "fallback_strategy": "expand search parameters" if quality == "insufficient" else ""
            }
        }
    
    def evaluate_chunks_with_llm(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        basic_analysis: List[Dict[str, Any]],
        original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the retrieved chunks using the LLM for advanced analysis.
        
        Args:
            query: The optimized query
            chunks: The retrieved chunks
            basic_analysis: Basic chunk analysis from non-LLM methods
            original_query: The original query if different from optimized
            
        Returns:
            Evaluation results
        """
        # Cache key construction
        cache_key = f"eval_{query}_{len(chunks)}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
            
        query_to_use = original_query or query
        
        try:
            # Prepare chunk information
            chunk_info = []
            for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks to keep prompt size reasonable
                chunk_info.append({
                    "id": chunk.get("id", f"chunk_{i}"),
                    "text": chunk.get("text", "")[:300],  # Truncate text
                    "metadata": chunk.get("metadata", {})
                })
                
            # Create the prompt
            messages = [
                SystemMessage(content="""
                You are a retrieval evaluation system for a Jewish religious text database. Your job is to:
                1. Evaluate if the retrieved chunks adequately address the user's query
                2. Identify information gaps or redundancies in the retrieved content
                3. Determine if more retrieval is needed or if we have sufficient information
                4. Recommend chunk ordering for optimal presentation
                
                You must return a valid JSON object with the following structure:
                {
                  "quality_assessment": "sufficient|partial|insufficient",
                  "stopping_decision": "continue|stop",
                  "token_adjustment": {
                    "current": number,
                    "recommended": number
                  },
                  "information_gaps": [
                    "string"
                  ],
                  "strategy_adjustments": {
                    "parameter_changes": {},
                    "recommended_actions": [],
                    "fallback_strategy": "string"
                  }
                }

                Your response must be valid JSON and NOTHING ELSE.
                """),
                HumanMessage(content=f"""
                User Query: {query_to_use}
                
                Retrieved Chunks (truncated for brevity):
                {json.dumps(chunk_info, indent=2)}
                
                Basic Analysis:
                {json.dumps(basic_analysis, indent=2)}
                
                Evaluate these chunks for their relevance to the user's query.
                Return a valid JSON object with your evaluation.
                """)
            ]
            
            # Call the LLM
            response = self.llm.invoke(messages)
            
            # Parse the response
            try:
                llm_result = json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                json_match = re.search(r'({[\s\S]*})', response.content)
                if json_match:
                    try:
                        llm_result = json.loads(json_match.group(1))
                    except:
                        # Fallback to results from basic analysis
                        return self.create_basic_evaluation_result(chunks, basic_analysis)
                else:
                    # Fallback to results from basic analysis
                    return self.create_basic_evaluation_result(chunks, basic_analysis)
                    
            # Combine LLM results with basic analysis
            result = {
                "quality_assessment": llm_result.get("quality_assessment", "partial"),
                "stopping_decision": llm_result.get("stopping_decision", "continue"),
                "token_adjustment": llm_result.get("token_adjustment", {
                    "current": sum(get_num_tokens(chunk.get("text", "")) for chunk in chunks),
                    "recommended": min(sum(get_num_tokens(chunk.get("text", "")) for chunk in chunks) * 1.5, config.MAX_CONTEXT_TOKENS)
                }),
                "chunk_analysis": basic_analysis,  # Use our basic analysis for chunk-level details
                "information_gaps": llm_result.get("information_gaps", []),
                "redundant_content": self.detect_redundancy(chunks),  # Use our redundancy detection
                "strategy_adjustments": llm_result.get("strategy_adjustments", {})
            }
            
            # Cache the result
            self.evaluation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            # Fallback to basic evaluation
            return self.create_basic_evaluation_result(chunks, basic_analysis)
    
    def create_basic_evaluation_result(
        self, 
        chunks: List[Dict[str, Any]], 
        basic_analysis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a basic evaluation result when LLM evaluation fails.
        
        Args:
            chunks: The retrieved chunks
            basic_analysis: Basic chunk analysis
            
        Returns:
            Basic evaluation results
        """
        # Calculate average relevance
        avg_relevance = sum(analysis.get("relevance_score", 0) for analysis in basic_analysis) / len(basic_analysis) if basic_analysis else 0
        
        # Determine quality and stopping decision
        if avg_relevance > 0.7:
            quality = "sufficient"
            stopping_decision = "stop"
        elif avg_relevance > 0.4:
            quality = "partial"
            stopping_decision = "continue" if len(chunks) < 5 else "stop"
        else:
            quality = "insufficient"
            stopping_decision = "continue"
            
        # Calculate token counts
        current_tokens = sum(get_num_tokens(chunk.get("text", "")) for chunk in chunks)
        
        return {
            "quality_assessment": quality,
            "stopping_decision": stopping_decision,
            "token_adjustment": {
                "current": current_tokens,
                "recommended": min(current_tokens * 1.5, config.MAX_CONTEXT_TOKENS) if quality != "sufficient" else current_tokens
            },
            "chunk_analysis": basic_analysis,
            "information_gaps": ["Possible information gaps based on relevance scores"] if quality != "sufficient" else [],
            "redundant_content": self.detect_redundancy(chunks),
            "strategy_adjustments": {
                "parameter_changes": {"increase_token_limit": quality != "sufficient"},
                "recommended_actions": [],
                "fallback_strategy": ""
            }
        }