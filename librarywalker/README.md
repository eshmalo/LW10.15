# LibraryWalker 10.15

A multi-level, exponential chunking framework with coverage-based retrieval for efficient document retrieval and chat interactions.

## Overview

This system implements:
- Multi-level, exponential chunking (capped by the embedder's 8k token limit)
- Coverage-based retrieval that tracks level-0 chunks to avoid redundancy
- Efficient batch processing with redundancy filtering
- Context window management with deduplication and optimized token usage
- Storage optimization that only keeps text for L0 chunks
- Detailed logging with chunk breakdown and contiguity analysis
- Chat interface that assembles context for answering queries

## Components

1. **Ingest Script**: Chunks and embeds documents
2. **Chat Script**: Retrieves relevant chunks and manages conversations

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r librarywalker/requirements.txt
   ```
3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```

## Directory Structure

```
librarywalker/
├── data/           # Data storage directory
│   ├── input/      # Original documents are stored here
│   └── output/     # Vector database and embeddings stored here
├── ingest/         # Document ingestion code
├── chat/           # Chat interface code
└── bin/            # Command-line entry points
```

## Usage

### Ingesting Documents

```bash
python librarywalker/bin/ingest.py
```

This will:
1. Find all documents in the input directory (`librarywalker/data/input/`)
2. Process each document into multi-level chunks
3. Embed the chunks
4. Store embeddings in the vector database

Simply place your documents in the input directory before running the script.

### Chat Interface

```bash
python librarywalker/bin/chat.py
```

This will:
1. Load the vector database
2. Accept your questions
3. Retrieve relevant chunks from documents
4. Pass the context to Claude for answering

Options:
- `--data-dir`: Specify the vector database directory
- `--max-tokens`: Set maximum tokens for context (default: 25000)
- `--debug`: Enable debug output
- `--show-context`: Show the context sent to Claude

## Current Progress
- [x] Project structure setup
- [x] Vector store implementation
- [x] Ingest script implementation (chunking + embedding)
- [x] Chat script implementation (retrieval + conversation)
- [x] Coverage-based retrieval with L0 chunk tracking
- [x] Storage optimization to only store L0 chunk text
- [x] Detailed logging with chunk breakdown and analysis
- [x] Expanded context window to 25k tokens
- [x] Batch processing with redundancy filtering
- [x] Context window management with early stopping
- [x] Sample document for testing
- [x] Documentation and usage examples