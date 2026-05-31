from fastapi import APIRouter

# backend/app.py
from backend.functions.fncs import (
    fn_clear_assets,
    fn_get_papers,
    fn_retrieve,
    fn_search_and_ingest,
    fn_wipe_chroma,
)

app = APIRouter()


@app.get("/search")
async def get_papers(query: str):
    """Fetch a list of 20 papers from arXiv"""
    return await fn_get_papers(query)


@app.get("/search/rag")
async def search_and_ingest(query: str, top_k: int = 5, wipe_db: bool = False):
    """Fetch papers, download PDFs, and embed into ChromaDB"""
    return await fn_search_and_ingest(query, top_k, wipe_db)


@app.delete("/wipe")
async def wipe_chroma():
    """Wipe the ChromaDB vectorstore"""
    return await fn_wipe_chroma()


@app.get("/query")
async def retrieve(query: str, top_k: int = 4):
    """Retrieve relevant document chunks from ChromaDB for a given query"""
    return await fn_retrieve(query, top_k)


@app.delete("/clear-assets")
async def clear_assets():
    """Delete the /assets directory"""
    return await fn_clear_assets()
