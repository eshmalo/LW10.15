#!/usr/bin/env python3
"""
LangChain-based workflow orchestration for the enhanced context management system.
This module coordinates the agent interactions, manages error handling, and ensures fallbacks.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union

from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from .. import config
from ..utils import get_num_tokens
from .memory import LibraryWalkerMemory


class WorkflowOrchestrator:
    """
    Orchestrates the enhanced context management workflow using LangChain components.
    Manages the flow of information between the Prompt Engineering AI, 
    Result Processing AI, and Context Manager AI.
    """
    
    def __init__(
        self,
        memory: Optional[LibraryWalkerMemory] = None,
        use_enhanced_context: Optional[bool] = None,
        use_prompt_engineering: Optional[bool] = None,
        use_result_processing: Optional[bool] = None,
        use_context_manager: Optional[bool] = None,
        incremental_retrieval: Optional[bool] = None
    ):
        """
        Initialize the workflow orchestrator.
        
        Args:
            memory: LibraryWalkerMemory instance for tracking conversation
            use_enhanced_context: Override for the USE_ENHANCED_CONTEXT config setting
            use_prompt_engineering: Override for the USE_PROMPT_ENGINEERING_AI config setting
            use_result_processing: Override for the USE_RESULT_PROCESSING_AI config setting
            use_context_manager: Override for the USE_CONTEXT_MANAGER_AI config setting
            incremental_retrieval: Override for the INCREMENTAL_RETRIEVAL_ENABLED config setting
        """
        # Initialize feature flags
        self.use_enhanced_context = (
            use_enhanced_context if use_enhanced_context is not None 
            else config.USE_ENHANCED_CONTEXT
        )
        
        self.use_prompt_engineering = (
            use_prompt_engineering if use_prompt_engineering is not None 
            else config.USE_PROMPT_ENGINEERING_AI
        )
        
        self.use_result_processing = (
            use_result_processing if use_result_processing is not None 
            else config.USE_RESULT_PROCESSING_AI
        )
        
        self.use_context_manager = (
            use_context_manager if use_context_manager is not None 
            else config.USE_CONTEXT_MANAGER_AI
        )
        
        self.incremental_retrieval = (
            incremental_retrieval if incremental_retrieval is not None 
            else config.INCREMENTAL_RETRIEVAL_ENABLED
        )
        
        # If the master toggle is off, disable all enhanced features
        if not self.use_enhanced_context:
            self.use_prompt_engineering = False
            self.use_result_processing = False
            self.use_context_manager = False
        
        # Initialize memory if not provided
        self.memory = memory if memory is not None else LibraryWalkerMemory()
        
        # Initialize LLM clients
        try:
            self.claude_llm = ChatAnthropic(
                model=config.ANTHROPIC_MODEL,
                anthropic_api_key=config.ANTHROPIC_API_KEY
            )
        except Exception as e:
            print(f"Failed to initialize Claude: {e}")
            self.claude_llm = None
        
        try:
            self.openai_llm = ChatOpenAI(
                model=config.OPENAI_CHAT_MODEL,
                openai_api_key=config.OPENAI_API_KEY
            )
        except Exception as e:
            print(f"Failed to initialize OpenAI: {e}")
            self.openai_llm = None
            
        # Check if we have at least one LLM available
        if self.claude_llm is None and self.openai_llm is None:
            raise ValueError("No LLM available. Please check your API keys and configuration.")
            
        # Set the default LLM based on availability
        self.default_llm = self.claude_llm if self.claude_llm is not None else self.openai_llm
        
        # Track performance metrics
        self.metrics = {
            "prompt_engineering_calls": 0,
            "result_processing_calls": 0,
            "context_manager_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0,
            "total_tokens_used": 0,
            "cumulative_latency": 0
        }
    
    def optimize_query(self, query: str) -> Dict[str, Any]:
        """
        Use the Prompt Engineering AI to optimize the query.
        
        Args:
            query: The user's query
            
        Returns:
            A dictionary with the optimized query and related parameters
        """
        if not self.use_prompt_engineering or not self.openai_llm:
            # Return a default format that matches what the AI would return
            return {
                "optimized_query": query,
                "detected_language": self.memory.detect_language(query),
                "hyde_content": "",
                "token_requirement": {
                    "minimum": 1000,
                    "optimal": 2000,
                    "maximum": config.MAX_CONTEXT_TOKENS
                },
                "query_strategy": {
                    "primary_strategy": "standard",
                    "use_hybrid_search": False,
                    "use_multilingual_expansion": True,
                    "language_expansions": {
                        "hebrew": True,
                        "aramaic": False,
                        "yiddish": False,
                        "other_languages": []
                    }
                }
            }
            
        start_time = time.time()
        try:
            # Get conversation history and important concepts
            conversation_summary = self.memory.get_summary()
            important_concepts = list(self.memory.important_concepts)
            
            # Create the prompt for the Prompt Engineering AI
            messages = [
                SystemMessage(content="""
                You are a query optimization system for a Jewish religious text retrieval system. Your job is to:
                1. Analyze the user's query and conversation context
                2. Detect the language(s) in the query
                3. Generate an optimized version of the query that will retrieve relevant religious texts
                4. Generate multilingual expansions for Hebrew, Aramaic, or Yiddish terms when appropriate
                5. Determine if Hypothetical Document Embedding (HyDE) would help for this query
                6. Estimate token requirements for adequate context
                
                You must strictly return a valid JSON object containing:
                {
                  "optimized_query": "string",
                  "detected_language": "string",
                  "hyde_content": "string",
                  "token_requirement": {
                    "minimum": number,
                    "optimal": number,
                    "maximum": number
                  },
                  "query_strategy": {
                    "primary_strategy": "standard|hyde|decomposition|hybrid",
                    "use_hybrid_search": boolean,
                    "use_multilingual_expansion": boolean,
                    "language_expansions": {
                      "hebrew": boolean,
                      "aramaic": boolean,
                      "yiddish": boolean,
                      "other_languages": []
                    }
                  }
                }

                Your response must be valid JSON and NOTHING ELSE.
                """),
                HumanMessage(content=f"""
                User Query: {query}
                
                Conversation Summary: {conversation_summary}
                
                Important Concepts Mentioned: {", ".join(important_concepts)}
                
                Query Languages Used: {", ".join(self.memory.query_languages)}
                
                Response Languages Used: {", ".join(self.memory.response_languages)}
                
                Optimize this query for Jewish religious text retrieval.
                """)
            ]
            
            # Call the OpenAI LLM
            response = self.openai_llm.invoke(messages)
            self.metrics["prompt_engineering_calls"] += 1
            
            # Extract the JSON response
            try:
                result = json.loads(response.content)
                # Update token usage metrics
                self.metrics["total_tokens_used"] += get_num_tokens(response.content)
                return result
            except json.JSONDecodeError:
                # If the response is not valid JSON, try to extract JSON from the text
                import re
                json_match = re.search(r'({.*})', response.content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        return result
                    except:
                        pass
                
                # Fallback to default
                self.metrics["fallback_used"] += 1
                return {
                    "optimized_query": query,
                    "detected_language": self.memory.detect_language(query),
                    "hyde_content": "",
                    "token_requirement": {
                        "minimum": 1000,
                        "optimal": 2000,
                        "maximum": config.MAX_CONTEXT_TOKENS
                    },
                    "query_strategy": {
                        "primary_strategy": "standard",
                        "use_hybrid_search": False,
                        "use_multilingual_expansion": True,
                        "language_expansions": {
                            "hebrew": True,
                            "aramaic": False,
                            "yiddish": False,
                            "other_languages": []
                        }
                    }
                }
        except Exception as e:
            print(f"Error in optimize_query: {e}")
            self.metrics["failed_calls"] += 1
            # Return default values on error
            return {
                "optimized_query": query,
                "detected_language": "english",
                "hyde_content": "",
                "token_requirement": {
                    "minimum": 1000,
                    "optimal": 2000,
                    "maximum": config.MAX_CONTEXT_TOKENS
                },
                "query_strategy": {
                    "primary_strategy": "standard",
                    "use_hybrid_search": False,
                    "use_multilingual_expansion": False
                }
            }
        finally:
            self.metrics["cumulative_latency"] += (time.time() - start_time)
    
    def evaluate_retrieval(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]],
        original_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use the Result Processing AI to evaluate the retrieval results.
        
        Args:
            query: The optimized query
            chunks: The retrieved chunks
            original_query: The original user query, if different from the optimized query
            
        Returns:
            A dictionary with the evaluation results
        """
        if not self.use_result_processing or not self.openai_llm:
            # Return a default format that matches what the AI would return
            return {
                "quality_assessment": "sufficient",
                "stopping_decision": "stop",
                "token_adjustment": {
                    "current": get_num_tokens("\n".join([c.get("text", "") for c in chunks])),
                    "recommended": get_num_tokens("\n".join([c.get("text", "") for c in chunks]))
                },
                "chunk_analysis": [
                    {
                        "chunk_id": chunk.get("id", f"chunk_{i}"),
                        "relevance_score": 0.8,
                        "redundancy_level": "none",
                        "information_value": "high",
                        "recommended_position": "middle"
                    }
                    for i, chunk in enumerate(chunks)
                ],
                "information_gaps": [],
                "redundant_content": [],
                "strategy_adjustments": {
                    "parameter_changes": {},
                    "recommended_actions": [],
                    "fallback_strategy": ""
                }
            }
            
        start_time = time.time()
        try:
            # Handle empty chunks case
            if not chunks:
                return {
                    "quality_assessment": "insufficient",
                    "stopping_decision": "continue",
                    "token_adjustment": {
                        "current": 0,
                        "recommended": 2000
                    },
                    "chunk_analysis": [],
                    "information_gaps": ["No relevant content found"],
                    "redundant_content": [],
                    "strategy_adjustments": {
                        "parameter_changes": {"expand_search": True},
                        "recommended_actions": ["try alternative query formulation"],
                        "fallback_strategy": "decompose_query"
                    }
                }
            
            # Prepare query information
            query_to_use = original_query if original_query else query
            
            # Prepare chunk information for the prompt
            chunk_info = []
            for i, chunk in enumerate(chunks):
                chunk_info.append({
                    "id": chunk.get("id", f"chunk_{i}"),
                    "text": chunk.get("text", "")[:300],  # Truncate text to keep prompt size reasonable
                    "metadata": chunk.get("metadata", {})
                })
            
            # Create the prompt for the Result Processing AI
            messages = [
                SystemMessage(content="""
                You are a retrieval evaluation system for a Jewish religious text database. Your job is to:
                1. Evaluate if the retrieved chunks adequately address the user's query
                2. Identify information gaps or redundancies in the retrieved content
                3. Determine if more retrieval is needed or if we have sufficient information
                4. Recommend chunk ordering for optimal presentation
                
                You must strictly return a valid JSON object with the following structure:
                {
                  "quality_assessment": "sufficient|partial|insufficient",
                  "stopping_decision": "continue|stop",
                  "token_adjustment": {
                    "current": number,
                    "recommended": number
                  },
                  "chunk_analysis": [
                    {
                      "chunk_id": "string",
                      "relevance_score": number,
                      "redundancy_level": "none|low|high",
                      "information_value": "high|medium|low",
                      "recommended_position": "beginning|middle|end"
                    }
                  ],
                  "information_gaps": [
                    "string"
                  ],
                  "redundant_content": [
                    {
                      "chunk_ids": ["string"],
                      "description": "string"
                    }
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
                
                Retrieved Chunks:
                {json.dumps(chunk_info, indent=2)}
                
                Evaluate these chunks for their relevance to the user's query.
                """)
            ]
            
            # Call the OpenAI LLM
            response = self.openai_llm.invoke(messages)
            self.metrics["result_processing_calls"] += 1
            
            # Extract the JSON response
            try:
                result = json.loads(response.content)
                # Update token usage metrics
                self.metrics["total_tokens_used"] += get_num_tokens(response.content)
                return result
            except json.JSONDecodeError:
                # If the response is not valid JSON, try to extract JSON from the text
                import re
                json_match = re.search(r'({.*})', response.content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        return result
                    except:
                        pass
                
                # Fallback to default
                self.metrics["fallback_used"] += 1
                return {
                    "quality_assessment": "sufficient",
                    "stopping_decision": "stop",
                    "token_adjustment": {
                        "current": get_num_tokens("\n".join([c.get("text", "") for c in chunks])),
                        "recommended": get_num_tokens("\n".join([c.get("text", "") for c in chunks]))
                    },
                    "chunk_analysis": [
                        {
                            "chunk_id": chunk.get("id", f"chunk_{i}"),
                            "relevance_score": 0.8,
                            "redundancy_level": "none",
                            "information_value": "high",
                            "recommended_position": "middle"
                        }
                        for i, chunk in enumerate(chunks)
                    ],
                    "information_gaps": [],
                    "redundant_content": [],
                    "strategy_adjustments": {
                        "parameter_changes": {},
                        "recommended_actions": [],
                        "fallback_strategy": ""
                    }
                }
        except Exception as e:
            print(f"Error in evaluate_retrieval: {e}")
            self.metrics["failed_calls"] += 1
            # Return default values on error
            return {
                "quality_assessment": "sufficient",
                "stopping_decision": "stop",
                "token_adjustment": {
                    "current": 0,
                    "recommended": 0
                },
                "chunk_analysis": [],
                "information_gaps": [],
                "redundant_content": [],
                "strategy_adjustments": {}
            }
        finally:
            self.metrics["cumulative_latency"] += (time.time() - start_time)
    
    def optimize_context(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        evaluation_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use the Context Manager AI to optimize the context for the LLM.
        
        Args:
            query: The user's query
            chunks: The retrieved chunks
            evaluation_result: Results from the Result Processing AI
            
        Returns:
            A dictionary with the optimized context
        """
        if not self.use_context_manager or not self.claude_llm:
            # Return a default format that matches what the AI would return
            # In this case, we'll just include all chunks
            return {
                "selected_chunks": [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
                "excluded_chunks": [],
                "chunk_ordering": [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
                "conversation_relevance": {
                    chunk.get("id", f"chunk_{i}"): 1.0 for i, chunk in enumerate(chunks)
                },
                "context_layout": {
                    "beginning": [],
                    "middle": [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
                    "end": []
                },
                "conversation_summary": {
                    "include_in_prompt": False,
                    "summary_text": ""
                }
            }
            
        start_time = time.time()
        try:
            # Handle empty chunks case
            if not chunks:
                return {
                    "selected_chunks": [],
                    "excluded_chunks": [],
                    "chunk_ordering": [],
                    "conversation_relevance": {},
                    "context_layout": {
                        "beginning": [],
                        "middle": [],
                        "end": []
                    },
                    "conversation_summary": {
                        "include_in_prompt": True,
                        "summary_text": self.memory.get_summary()
                    }
                }
            
            # Get conversation history and summary
            conversation_history = self.memory.load_relevant_history(query)
            conversation_summary = self.memory.get_summary()
            
            # Convert conversation history to a text format
            history_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content[:200]}..."
                for msg in conversation_history
            ])
            
            # Get information about frequently retrieved chunks
            frequent_chunks = self.memory.get_frequently_retrieved_chunks()
            recent_chunks = self.memory.get_recently_retrieved_chunks()
            
            # Prepare chunk information for the prompt
            chunk_info = []
            for i, chunk in enumerate(chunks):
                is_frequent = chunk.get("id") in frequent_chunks
                is_recent = chunk.get("id") in recent_chunks
                
                chunk_info.append({
                    "id": chunk.get("id", f"chunk_{i}"),
                    "text": chunk.get("text", "")[:300],  # Truncate text to keep prompt size reasonable
                    "metadata": chunk.get("metadata", {}),
                    "is_frequently_retrieved": is_frequent,
                    "is_recently_retrieved": is_recent
                })
                
            # Include evaluation result if available
            evaluation_info = ""
            if evaluation_result:
                evaluation_info = f"""
                Evaluation Result:
                - Quality Assessment: {evaluation_result.get('quality_assessment', 'unknown')}
                - Information Gaps: {', '.join(evaluation_result.get('information_gaps', []))}
                - Chunk Analysis: 
                {json.dumps(evaluation_result.get('chunk_analysis', []), indent=2)}
                """
            
            # Create the prompt for the Context Manager AI
            messages = [
                SystemMessage(content="""
                You are a context management system for a Jewish religious text retrieval system. Your job is to:
                1. Analyze retrieved chunks in the context of the conversation history
                2. Select which chunks to include in the final context
                3. Determine the optimal ordering of chunks to avoid the "lost in the middle" effect
                4. Identify the most relevant chunks based on the user's query
                5. Remove redundant or less relevant chunks when necessary
                6. Generate a conversation summary when appropriate
                
                You must strictly return a valid JSON object with the following structure:
                {
                  "selected_chunks": [
                    "chunk_id_string"
                  ],
                  "excluded_chunks": [
                    "chunk_id_string"
                  ],
                  "chunk_ordering": [
                    "chunk_id_string"
                  ],
                  "conversation_relevance": {
                    "chunk_id": "relevance_to_conversation_score"
                  },
                  "context_layout": {
                    "beginning": ["chunk_id_string"],
                    "middle": ["chunk_id_string"],
                    "end": ["chunk_id_string"]
                  },
                  "conversation_summary": {
                    "include_in_prompt": boolean,
                    "summary_text": "string"
                  }
                }

                Your response must be valid JSON and NOTHING ELSE.
                
                Important principles:
                1. Never exclude ancient original-text chunks; supplement them with explanations in user's language
                2. Position key information at beginning or end to mitigate "lost in the middle" effect
                3. Maintain coherence throughout the conversation
                4. Preserve authoritative sources even if they are in a different language than the query
                5. Prioritize primary religious texts over secondary commentaries when appropriate
                """),
                HumanMessage(content=f"""
                User Query: {query}
                
                Conversation History:
                {history_text}
                
                Conversation Summary:
                {conversation_summary}
                
                Retrieved Chunks:
                {json.dumps(chunk_info, indent=2)}
                
                {evaluation_info}
                
                Optimize the context for this query, considering the conversation history and retrieved chunks.
                """)
            ]
            
            # Call the Claude LLM
            response = self.claude_llm.invoke(messages)
            self.metrics["context_manager_calls"] += 1
            
            # Extract the JSON response
            try:
                result = json.loads(response.content)
                # Update token usage metrics
                self.metrics["total_tokens_used"] += get_num_tokens(response.content)
                return result
            except json.JSONDecodeError:
                # If the response is not valid JSON, try to extract JSON from the text
                import re
                json_match = re.search(r'({.*})', response.content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        return result
                    except:
                        pass
                
                # Fallback to default
                self.metrics["fallback_used"] += 1
                return {
                    "selected_chunks": [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
                    "excluded_chunks": [],
                    "chunk_ordering": [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
                    "conversation_relevance": {
                        chunk.get("id", f"chunk_{i}"): 1.0 for i, chunk in enumerate(chunks)
                    },
                    "context_layout": {
                        "beginning": [],
                        "middle": [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
                        "end": []
                    },
                    "conversation_summary": {
                        "include_in_prompt": False,
                        "summary_text": ""
                    }
                }
        except Exception as e:
            print(f"Error in optimize_context: {e}")
            self.metrics["failed_calls"] += 1
            # Return default values on error
            return {
                "selected_chunks": [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
                "excluded_chunks": [],
                "chunk_ordering": [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
                "conversation_relevance": {},
                "context_layout": {
                    "beginning": [],
                    "middle": [chunk.get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)],
                    "end": []
                },
                "conversation_summary": {
                    "include_in_prompt": False,
                    "summary_text": ""
                }
            }
        finally:
            self.metrics["cumulative_latency"] += (time.time() - start_time)
    
    def incremental_retrieval_process(
        self,
        query: str,
        retriever_func,
        max_iterations: int = 4,
        initial_token_limit: int = 1000
    ):
        """
        Implement an incremental retrieval process that expands context when needed.
        
        Args:
            query: The user's query
            retriever_func: Function that performs retrieval, takes token_limit as argument
            max_iterations: Maximum number of retrieval iterations
            initial_token_limit: Starting token limit for retrieval
            
        Returns:
            The final chunks and context manager output
        """
        if not self.incremental_retrieval:
            # If incremental retrieval is disabled, just do a single retrieval
            chunks = retriever_func(token_limit=config.MAX_CONTEXT_TOKENS)
            
            # Convert tuple chunks to dictionaries for processing
            chunk_dicts = [
                {
                    "text": text,
                    "metadata": meta,
                    "score": score,
                    "id": meta.get('chunk_index', f"chunk_{i}")
                }
                for i, (text, meta, score) in enumerate(chunks)
            ]
            
            # Process results if needed
            evaluation_result = None
            if self.use_result_processing:
                evaluation_result = self.evaluate_retrieval(query, chunk_dicts)
                
            context_result = None
            if self.use_context_manager:
                context_result = self.optimize_context(query, chunk_dicts, evaluation_result)
                
            return chunks, evaluation_result, context_result
        
        # Initialize for incremental retrieval
        all_chunks = []
        all_chunk_dicts = []
        current_token_limit = initial_token_limit
        final_evaluation = None
        final_context = None
        
        # Check if we need to search for non-Latin scripts (e.g., Hebrew)
        contains_non_latin = any(ord(c) > 127 for c in query)
        
        for iteration in range(max_iterations):
            # Perform retrieval with current token limit
            new_chunks = retriever_func(token_limit=current_token_limit)
            
            # Convert tuple chunks to dictionaries for processing
            new_chunk_dicts = [
                {
                    "text": text,
                    "metadata": meta,
                    "score": score,
                    "id": meta.get('chunk_index', f"chunk_{i}")
                }
                for i, (text, meta, score) in enumerate(new_chunks)
            ]
            
            # Update all_chunks and all_chunk_dicts
            if iteration == 0:
                all_chunks = new_chunks
                all_chunk_dicts = new_chunk_dicts
            else:
                # Merge new chunks with existing ones, avoiding duplicates
                existing_ids = set(chunk_dict["id"] for chunk_dict in all_chunk_dicts)
                
                # Add tuples to all_chunks
                for chunk in new_chunks:
                    meta = chunk[1]
                    chunk_id = meta.get('chunk_index', f"chunk_{len(all_chunks)}")
                    if chunk_id not in existing_ids:
                        all_chunks.append(chunk)
                
                # Add dictionaries to all_chunk_dicts
                for chunk_dict in new_chunk_dicts:
                    if chunk_dict["id"] not in existing_ids:
                        all_chunk_dicts.append(chunk_dict)
                        existing_ids.add(chunk_dict["id"])
            
            # Evaluate retrieval if we have result processing enabled
            if self.use_result_processing:
                evaluation_result = self.evaluate_retrieval(query, all_chunk_dicts)
                final_evaluation = evaluation_result
                
                # Check if we should stop or continue
                stopping_decision = evaluation_result.get("stopping_decision", "continue")
                
                # Special case for non-Latin scripts or small token usage - continue searching more aggressively
                token_budget_percentage = None
                if "token_adjustment" in evaluation_result:
                    current_tokens = evaluation_result["token_adjustment"].get("current", 0)
                    token_budget_percentage = (current_tokens / config.MAX_CONTEXT_TOKENS) * 100
                    
                    # Force continue retrieval when we've used very little of our token budget and
                    # especially when there's non-Latin text 
                    if (token_budget_percentage < 20 or 
                        (contains_non_latin and token_budget_percentage < 60)):
                        stopping_decision = "continue"
                        print(f"Forcing continued retrieval due to low token usage ({token_budget_percentage:.1f}%)")
                
                # Check if we've found what we're looking for
                if stopping_decision == "stop":
                    # Don't stop too early for queries with non-Latin text unless we've used
                    # a significant portion of our token budget
                    if contains_non_latin and (token_budget_percentage is None or token_budget_percentage < 25):
                        print("Continuing search despite 'stop' decision due to presence of non-Latin text")
                    else:
                        break
                    
                # Update token limit based on recommendation
                if "token_adjustment" in evaluation_result:
                    recommended = evaluation_result["token_adjustment"].get("recommended", 0)
                    if recommended > current_token_limit:
                        current_token_limit = min(recommended, config.MAX_CONTEXT_TOKENS)
                    else:
                        # If no specific recommendation, use geometric expansion for non-Latin queries
                        # to ensure thorough search
                        if contains_non_latin:
                            current_token_limit = min(current_token_limit * 2, config.MAX_CONTEXT_TOKENS)
            else:
                # Without result processing, use geometric expansion
                current_token_limit = min(current_token_limit * 2, config.MAX_CONTEXT_TOKENS)
                
            # If we've reached the maximum context size, stop
            if current_token_limit >= config.MAX_CONTEXT_TOKENS:
                break
        
        # Final context optimization
        if self.use_context_manager:
            final_context = self.optimize_context(query, all_chunk_dicts, final_evaluation)
        
        return all_chunks, final_evaluation, final_context
    
    def run_enhanced_workflow(
        self,
        query: str,
        retriever_func,
        initial_token_limit: int = 1000
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """
        Run the complete enhanced context management workflow.
        
        Args:
            query: The user's query
            retriever_func: Function that performs retrieval, should accept a token_limit parameter
            initial_token_limit: Initial token limit for incremental retrieval
            
        Returns:
            A tuple of (chunks, evaluation_result, context_result)
        """
        if not self.use_enhanced_context:
            # If enhanced context is disabled, just return the basic retrieval
            chunks = retriever_func(token_limit=config.MAX_CONTEXT_TOKENS)
            return chunks, {}, {}
        
        # Track query language
        self.memory.update_query_language(query)
        
        # Step 1: Optimize the query with Prompt Engineering AI
        if self.use_prompt_engineering:
            query_result = self.optimize_query(query)
            optimized_query = query_result.get("optimized_query", query)
            
            # Extract token requirement
            token_requirement = query_result.get("token_requirement", {})
            optimal_tokens = token_requirement.get("optimal", initial_token_limit)
            
            # Update the retriever function to use the optimized query
            original_retriever = retriever_func
            retriever_func = lambda token_limit: original_retriever(
                optimized_query=optimized_query, 
                token_limit=token_limit,
                query_result=query_result
            )
        else:
            # If not using prompt engineering, just use the original query
            optimized_query = query
            optimal_tokens = initial_token_limit
        
        # Step 2: Perform incremental retrieval
        chunks, evaluation_result, context_result = self.incremental_retrieval_process(
            query=optimized_query,
            retriever_func=retriever_func,
            initial_token_limit=optimal_tokens
        )
        
        # Step 3: Update memory with retrieval metadata
        # Transform the tuples into dictionaries
        chunk_dicts = [
            {
                "text": text,
                "metadata": meta,
                "score": score,
                "id": meta.get('chunk_index', f"chunk_{i}")
            }
            for i, (text, meta, score) in enumerate(chunks)
        ]
        
        self.memory.add_retrieval_metadata(
            query=query,
            chunks=chunk_dicts,
            success=bool(chunks),
            strategy="enhanced_context" if self.use_enhanced_context else "standard"
        )
        
        # Extract and store concepts from the query
        self.memory.update_concepts(query)
        
        return chunks, evaluation_result, context_result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the workflow.
        
        Returns:
            A dictionary with performance metrics
        """
        return self.metrics
    
    def reset_metrics(self) -> None:
        """
        Reset the performance metrics.
        """
        self.metrics = {
            "prompt_engineering_calls": 0,
            "result_processing_calls": 0,
            "context_manager_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0,
            "total_tokens_used": 0,
            "cumulative_latency": 0
        }