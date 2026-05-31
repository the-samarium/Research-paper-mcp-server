# Patent IP Research MCP

## Overview

This project provides a FastAPI backend for searching arXiv, downloading PDFs, embedding documents with Ollama, and persisting vector embeddings in Chroma.

## Installation

1. Create and activate a virtual environment:
   - PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

2. Install project dependencies:
   ```powershell
   pip install .
   ```

3. Ensure Ollama is installed and running if you plan to use the RAG pipeline with `langchain-ollama`.

## Run the FastAPI backend

From the repository root:

```powershell
uvicorn backend.app:app --reload
```

The backend will be available at `http://127.0.0.1:8000` by default.

## API Endpoints

- `GET /search?query=<term>`
  - Searches arXiv for papers matching `<term>` and returns metadata.

- `GET /search/rag?query=<term>&top_k=<n>&wipe_db=<true|false>`
  - Searches arXiv, downloads matching PDFs into `assets/`, extracts text, splits into chunks, embeds, and persists into `chromaDB/`.
  - Example:
    ```powershell
    curl "http://127.0.0.1:8000/search/rag?query=machine+learning&top_k=5&wipe_db=false"
    ```

- `GET /wipe?persist_directory=./chromaDB`
  - Deletes the persisted Chroma vector database.

## Future query endpoint

A future endpoint can be added to query the persisted vector store and return answers from the indexed document context. This would allow true RAG-style question answering.

## Storage locations

- `assets/`
  - Downloaded PDF files are saved here by the backend.

- `chromaDB/`
  - Persisted Chroma vector database files are stored here.

## Notes

- The backend expects to be launched from the repository root so paths like `assets/` and `chromaDB/` resolve correctly.
- If `wipe_db=true` is passed to `/search/rag`, the existing vector DB is cleared before re-indexing.
