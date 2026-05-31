# Research MCP Server

A research paper discovery and ingestion tool built on top of arXiv, ChromaDB, and FastMCP.
Originally developed as a FastAPI backend with local LLM inference via Ollama, then migrated to an MCP server so that an LLM host (like Claude) handles all reasoning — keeping the server lean and focused on retrieval.

---

## Project Evolution

### Phase 1 — FastAPI + Local LLM
The project started as a standalone FastAPI backend. It used `langchain-ollama` to run a local LLM, built a RAG pipeline that retrieved relevant chunks from ChromaDB, and returned fully synthesized answers from the server itself.

### Phase 2 — MCP Server (current)
The LLM layer was removed from the server entirely. The backend now exposes retrieval tools via FastMCP. The connected LLM host (Claude or any MCP-compatible client) receives raw document chunks and does the reasoning. This is a cleaner separation of concerns — the server does search and retrieval, the LLM does thinking.

---

## Project Structure

```
patent-ip-research-mcp/
├── server.py                  # FastMCP entry point
├── pyproject.toml
├── backend/
│   ├── __init__.py
│   ├── app.py                 # FastAPI app with router
│   ├── main.py                # APIRouter with all route definitions
│   ├── fncs.py                # All business logic functions
│   └── rag_pipeline/
│       ├── __init__.py
│       ├── rag.py             # Embedding + ChromaDB persistence
│       └── retriver.py        # Vector store retrieval
├── assets/                    # Downloaded PDFs (auto-created)
└── chromaDB/                  # Persisted Chroma vector DB (auto-created)
```

---

## Installation

### Prerequisites
- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.com/) installed and running (required for the embedding model)

### Setup

```powershell
# Clone the repo
git clone https://github.com/your-username/patent-ip-research-mcp.git
cd patent-ip-research-mcp

# Create and activate virtual environment
uv venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install .
```

---

## Running as an MCP Server

### Via FastMCP CLI (inspector / dev mode)
```powershell
uv run fastmcp dev inspector server.py
```

### Via MCP Inspector UI
Set the following in the Inspector:
- **Command:** `uv`
- **Arguments:** `run fastmcp run server.py`

### Registering with Claude Desktop
Add the following to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "patent-ip-research": {
      "command": "uv",
      "args": ["run", "fastmcp", "run", "server.py"],
      "cwd": "C:\\path\\to\\patent-ip-research-mcp"
    }
  }
}
```

---

## Running as a FastAPI Server (legacy mode)

```powershell
uvicorn backend.app:appp --reload
```

API docs available at `http://127.0.0.1:8000/docs`.

---

## API / MCP Tools

| Endpoint | Description |
|---|---|
| `GET /search?query=<term>` | Fetch up to 20 papers from arXiv matching the query |
| `GET /search/rag?query=<term>&top_k=<n>&wipe_db=<bool>` | Fetch papers, download PDFs, embed and persist to ChromaDB |
| `GET /query?query=<term>&top_k=<n>` | Retrieve relevant document chunks from ChromaDB for a query |
| `GET /wipe` | Wipe the ChromaDB vector database |
| `GET /clear-assets` | Delete the downloaded PDFs in `assets/` |

---

## Storage

| Path | Contents |
|---|---|
| `assets/` | Downloaded PDF files from arXiv, auto-created at project root |
| `chromaDB/` | Persisted Chroma vector database, auto-created at project root |

Both directories are created automatically at the project root when first used. Paths are resolved from `fncs.py`'s location so they always land in the right place regardless of working directory.

---

## Dependencies

Key packages from `pyproject.toml`:

- `fastmcp` — MCP server framework
- `fastapi` — underlying HTTP layer
- `langchain-chroma` — ChromaDB vector store integration
- `langchain-ollama` — Ollama embeddings (embedding only, LLM inference removed)
- `langchain-text-splitters` — document chunking
- `pypdf` — PDF text extraction
- `feedparser` — arXiv Atom feed parsing

---

## Notes

- Ollama is still required for **embeddings** (`nomic-embed-text` or similar). Only the LLM inference step was removed during the MCP migration.
- If `wipe_db=true` is passed to `/search/rag`, the existing vector DB is released from memory and cleared before re-indexing. This is important on Windows where ChromaDB file locks can prevent deletion.
- The `assets/` folder must exist and contain PDFs before `/query` will return results. Run `/search/rag` first.