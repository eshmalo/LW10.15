#!/usr/bin/env python3

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from librarywalker.chat.chat import main

if __name__ == "__main__":
    # Get absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Get data directory path with absolute path
    data_dir = os.path.join(project_root, "librarywalker", "data")
    
    # Call main with absolute data_dir
    import argparse
    from librarywalker.config import MAX_CONTEXT_TOKENS
    
    parser = argparse.ArgumentParser(description="Promptchan Chat Interface")
    parser.add_argument("--data-dir", default=data_dir, help="Directory with vector database")
    parser.add_argument("--max-tokens", type=int, default=MAX_CONTEXT_TOKENS, help="Maximum tokens for context")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--show-context", action="store_true", help="Show context")
    
    args = parser.parse_args()
    
    print(f"Using context window of {args.max_tokens} tokens")
    
    from librarywalker.chat.chat import chat_loop
    chat_loop(
        data_dir=args.data_dir,
        max_tokens=args.max_tokens,
        debug=args.debug,
        show_context=args.show_context
    )