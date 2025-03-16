#!/usr/bin/env python3

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

# Embedding cost estimates
# Current pricing as of March 2024 - update as needed
EMBEDDING_COSTS = {
    "text-embedding-3-small": 0.00000002,  # $0.02 per 1M tokens
    "text-embedding-3-large": 0.00000013,  # $0.13 per 1M tokens
    "text-embedding-ada-002": 0.00000010,  # $0.10 per 1M tokens
}

# Claude API settings
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")

# Fallback settings
USE_OPENAI_FALLBACK = os.getenv("USE_OPENAI_FALLBACK", "true").lower() == "true"

# Chunking settings
MAX_EMBEDDING_TOKENS = 8000  # Max tokens for embedding
MAX_CONTEXT_TOKENS = 25000    # Max tokens to include in Claude context
BASE_CHUNK_SIZE = "paragraph"  # Options: "sentence", "paragraph", "line"
MAX_CHUNKS_PER_LEVEL = 100000000  # Process all chunks, no matter how many (set very high)

# Vector DB settings
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)