#!/usr/bin/env python3

import os
import argparse
import warnings
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", message=".*Please see the migration guide at.*", category=Warning)

from ..config import (
    DATA_DIR, 
    MAX_CONTEXT_TOKENS, 
    USE_ENHANCED_CONTEXT,
    USE_LANGCHAIN_MEMORY,
    USE_PROMPT_ENGINEERING_AI,
    USE_RESULT_PROCESSING_AI,
    USE_CONTEXT_MANAGER_AI,
    INCREMENTAL_RETRIEVAL_ENABLED
)
from ..vector_store import VectorStore
from .retriever import retrieve_relevant_chunks, retrieve_with_improved_query
from .prompter import format_context_from_chunks, create_claude_prompt, call_claude_api

# Import enhanced context management components if enabled
if USE_ENHANCED_CONTEXT:
    from ..langchain.memory import LibraryWalkerMemory
    from ..langchain.workflow import WorkflowOrchestrator
    from ..agents.context_manager import ContextManagerAI
    from ..agents.prompt_engineer import PromptEngineerAI
    from ..agents.result_processor import ResultProcessorAI

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
             show_context: bool = False,
             use_hebrew_expansion: bool = True,
             use_hyde: bool = True,
             standard_retrieval: bool = False,
             use_enhanced_context: bool = USE_ENHANCED_CONTEXT) -> None:
    """
    Main chat loop for LangChain.
    
    Args:
        data_dir: Directory with vector database
        max_tokens: Maximum tokens for context
        debug: Enable debug output
        show_context: Show the context sent to Claude
        use_hebrew_expansion: Whether to use Hebrew term expansion
        use_hyde: Whether to use HyDE technique
        standard_retrieval: If True, use standard retrieval instead of enhanced
        use_enhanced_context: Whether to use the enhanced context management system
    """
    # Initialize vector store
    try:
        vector_store = VectorStore(data_dir)
        print(f"Loaded vector store from {data_dir}")
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        print("You may need to ingest some documents first.")
        return
    
    # Initialize the enhanced context management components if enabled
    memory = None
    workflow = None
    
    if use_enhanced_context:
        try:
            print("Initializing enhanced context management system...")
            memory = LibraryWalkerMemory()
            workflow = WorkflowOrchestrator(memory=memory)
            print("Enhanced context management initialized successfully")
        except Exception as e:
            print(f"Error initializing enhanced context management: {str(e)}")
            print("Falling back to standard context management")
            use_enhanced_context = False
    
    print("\nLangChain Chat Interface")
    print("Type 'exit' or 'quit' to end the conversation")
    print("-" * 50)
    
    # Enhanced retrieval function wrapper for the workflow orchestrator
    def enhanced_retrieval_func(optimized_query=None, token_limit=max_tokens, query_result=None):
        """
        Wrapper function for the retrieval process that supports the workflow orchestrator.
        
        Args:
            optimized_query: Optional optimized query text from PromptEngineerAI
            token_limit: Token limit for retrieval
            query_result: Optional query optimization result from PromptEngineerAI
            
        Returns:
            List of retrieved chunks
        """
        query_to_use = optimized_query if optimized_query is not None else user_input
        hyde_content = None
        use_hybrid = False
        exact_phrase = None
        
        # Extract additional parameters from query_result if available
        if query_result:
            hyde_content = query_result.get("hyde_content")
            exact_phrases = query_result.get("exact_phrases", [])
            exact_phrase = exact_phrases[0] if exact_phrases else None
            
            if query_result.get("query_strategy"):
                use_hybrid = query_result.get("query_strategy").get("use_hybrid_search", False)
        
        # Check for exact phrases (quoted or non-Latin script) in the query 
        # if not found in query_result
        if not exact_phrase:
            # Check for quoted text
            import re
            matches = re.findall(r'"([^"]*)"', user_input)
            if matches:
                exact_phrase = matches[0]
                use_hybrid = True
            else:
                # Check for non-Latin (e.g., Hebrew) text
                words = user_input.split()
                for word in words:
                    # If any character is outside of standard ASCII range, it's likely Hebrew/Arabic/etc.
                    if any(ord(c) > 127 for c in word):
                        exact_phrase = word
                        use_hybrid = True
                        break
        
        if standard_retrieval:
            return retrieve_relevant_chunks(
                query_to_use,
                vector_store,
                max_tokens=token_limit
            )
        else:
            return retrieve_with_improved_query(
                query_to_use,
                vector_store,
                max_tokens=token_limit,
                use_hebrew_expansion=use_hebrew_expansion,
                use_hyde=use_hyde,
                hyde_content=hyde_content,
                exact_phrase=exact_phrase,
                use_hybrid_search=use_hybrid
            )
    
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
            # Add the user message to memory if enabled
            if use_enhanced_context and memory:
                memory.add_user_message(user_input)
                memory.update_query_language(user_input)
            
            # Choose between enhanced or standard context management
            if use_enhanced_context and workflow:
                print("Using enhanced context management system...")
                
                # Run the workflow with the enhanced retrieval function
                chunks, evaluation_result, context_result = workflow.run_enhanced_workflow(
                    query=user_input,
                    retriever_func=enhanced_retrieval_func
                )
                
                # Format context based on the context manager's recommendations
                if context_result:
                    print(f"Context optimization: Selected {len(context_result.get('selected_chunks', []))} chunks, excluded {len(context_result.get('excluded_chunks', []))} chunks")
                    
                    # If we have a context layout, reorder chunks based on it
                    if 'context_layout' in context_result:
                        layout = context_result['context_layout']
                        # Create a mapping of chunk_id to chunk
                        chunk_map = {
                            meta.get('chunk_index', f"chunk_{i}"): (text, meta, score) 
                            for i, (text, meta, score) in enumerate(chunks)
                        }
                        
                        # Reorder chunks according to layout
                        ordered_chunks = []
                        for section in ['beginning', 'middle', 'end']:
                            for chunk_id in layout.get(section, []):
                                if chunk_id in chunk_map:
                                    ordered_chunks.append(chunk_map[chunk_id])
                        
                        # Fall back to all chunks if ordering failed
                        if ordered_chunks:
                            chunks = ordered_chunks
                
                # Format the context with the optimized chunks, layout and conversation summary
                context = format_context_from_chunks(
                    chunks,
                    context_layout=context_result.get('context_layout'),
                    conversation_summary=context_result.get('conversation_summary')
                )
                
            else:
                # Use standard retrieval approach
                if standard_retrieval:
                    print("Searching knowledge base with standard retrieval...")
                    chunks = retrieve_relevant_chunks(
                        user_input, 
                        vector_store, 
                        max_tokens=max_tokens
                    )
                else:
                    print("Searching knowledge base with enhanced retrieval...")
                    chunks = retrieve_with_improved_query(
                        user_input, 
                        vector_store, 
                        max_tokens=max_tokens,
                        use_hebrew_expansion=use_hebrew_expansion,
                        use_hyde=use_hyde
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
            
            # Create prompt (use scholarly persona if requested)
            use_scholarly = False
            if use_enhanced_context and context_result and 'use_scholarly_persona' in context_result:
                use_scholarly = context_result.get('use_scholarly_persona', False)
            prompt = create_claude_prompt(user_input, context, use_scholarly_persona=use_scholarly)
            
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
            
            # Add the response to memory if enabled
            if use_enhanced_context and memory:
                memory.add_ai_message(response)
                memory.update_response_language(response)
                memory.update_concepts(response)
            
            # Show response
            print("\nLangChain:", response)
            
            # Show debug info if requested
            if debug:
                print("\nDebug Information:")
                print(f"Retrieved {len(chunks)} chunks")
                for i, (_, metadata, score) in enumerate(chunks):
                    print(f"  Chunk {i+1}: {metadata.get('document_id')} | " 
                         f"Level {metadata.get('level')} | "
                         f"Index {metadata.get('chunk_index')} | "
                         f"Score {score:.3f}")
                
                # Show workflow metrics if enhanced context is enabled
                if use_enhanced_context and workflow:
                    print("\nWorkflow Metrics:")
                    metrics = workflow.get_metrics()
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Command-line interface for LangChain chat."""
    parser = argparse.ArgumentParser(description="LangChain Chat Interface")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory with vector database")
    parser.add_argument("--max-tokens", type=int, default=MAX_CONTEXT_TOKENS, 
                      help="Maximum tokens for context")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--show-context", action="store_true", help="Show the context sent to Claude")
    parser.add_argument("--standard", action="store_true", help="Use standard retrieval instead of enhanced")
    parser.add_argument("--no-hebrew", action="store_true", help="Disable Hebrew term expansion")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE technique")
    
    # Add scholarly persona flag
    parser.add_argument("--scholarly", "--use-scholarly-persona", action="store_true",
                      help="Use scholarly rabbinical persona for responses")
    
    # Enhanced context management options
    parser.add_argument("--use-enhanced-context", action="store_true", 
                      help="Enable enhanced context management system")
    parser.add_argument("--no-enhanced-context", action="store_true", 
                      help="Disable enhanced context management system")
    parser.add_argument("--no-memory", action="store_true", 
                      help="Disable conversation memory")
    parser.add_argument("--no-prompt-engineering", action="store_true", 
                      help="Disable prompt engineering AI")
    parser.add_argument("--no-result-processing", action="store_true", 
                      help="Disable result processing AI")
    parser.add_argument("--no-context-manager", action="store_true", 
                      help="Disable context manager AI")
    parser.add_argument("--no-incremental-retrieval", action="store_true", 
                      help="Disable incremental retrieval")
    
    args = parser.parse_args()
    
    # Determine if enhanced context should be used
    use_enhanced_context = USE_ENHANCED_CONTEXT
    if args.use_enhanced_context:
        use_enhanced_context = True
    if args.no_enhanced_context:
        use_enhanced_context = False
    
    # Override feature flags if specific flags are provided
    if use_enhanced_context:
        if args.no_memory:
            os.environ["USE_LANGCHAIN_MEMORY"] = "false"
        if args.no_prompt_engineering:
            os.environ["USE_PROMPT_ENGINEERING_AI"] = "false"
        if args.no_result_processing:
            os.environ["USE_RESULT_PROCESSING_AI"] = "false"
        if args.no_context_manager:
            os.environ["USE_CONTEXT_MANAGER_AI"] = "false"
        if args.no_incremental_retrieval:
            os.environ["INCREMENTAL_RETRIEVAL_ENABLED"] = "false"
    
    # Override scholarly persona flag if provided
    if args.scholarly:
        os.environ["USE_SCHOLARLY_PERSONA"] = "true"
    
    chat_loop(
        data_dir=args.data_dir,
        max_tokens=args.max_tokens,
        debug=args.debug,
        show_context=args.show_context,
        use_hebrew_expansion=not args.no_hebrew,
        use_hyde=not args.no_hyde,
        standard_retrieval=args.standard,
        use_enhanced_context=use_enhanced_context
    )

if __name__ == "__main__":
    main()