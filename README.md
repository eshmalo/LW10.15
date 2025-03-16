# Promptchan

A multi-level, exponential chunking framework for efficient document retrieval and chat interactions.

## Quick Start

1. Install dependencies:
   ```
   pip install -r librarywalker/requirements.txt
   ```

2. Set up your environment by creating a `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```

3. Ingest documents:
   
   **First, place your documents in the input directory:**
   ```
   librarywalker/data/input/
   ```
   
   **Then run the ingest script:**
   ```
   python librarywalker/bin/ingest.py
   ```
   
   This will process all documents in the input directory and store embeddings in `librarywalker/data/output/`.

4. Start chatting:
   ```
   python librarywalker/bin/chat.py
   ```
   This will connect to the vector database in `librarywalker/data/output/` and allow you to ask questions about your documents.

For full documentation, see [librarywalker/README.md](librarywalker/README.md)