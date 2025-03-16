#!/usr/bin/env python3

import os
import json
import tiktoken
from typing import List, Dict, Any, Tuple, Optional

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # For OpenAI embeddings and Claude

def count_tokens(text: str) -> int:
    """Count tokens in a string using tiktoken."""
    if not text:
        return 0
    tokens = tokenizer.encode(text)
    return len(tokens)

def read_file(filepath: str) -> str:
    """Read a file and return its contents as a string."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def save_json(data: Any, filepath: str) -> None:
    """Save data as JSON to filepath."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Any:
    """Load JSON data from filepath."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(dirpath: str) -> None:
    """Ensure directory exists."""
    os.makedirs(dirpath, exist_ok=True)