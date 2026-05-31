"""Retriever helpers for serving queries against the persisted Chroma vectorstore."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from rag_pipeline.rag import build_embed_model

# Module-level cache: keyed by (persist_directory, collection_name)
# so multiple collections are supported without re-opening on every call.
_vectorstore_cache: dict[tuple[str, str], Chroma] = {}


def get_vectorstore(
    persist_directory: str = "./chromaDB",
    collection_name: str = "documents",
) -> Chroma:
    """Return a cached Chroma vectorstore, opening it once per process."""
    key = (persist_directory, collection_name)
    if key not in _vectorstore_cache:
        # ✅ Guard: don't silently return an empty store if DB doesn't exist
        if not Path(persist_directory).exists():
            raise FileNotFoundError(
                f"Vectorstore not found at '{persist_directory}'. "
                "Run /search/rag to ingest documents first."
            )
        embed_model = build_embed_model()
        _vectorstore_cache[key] = Chroma(
            collection_name=collection_name,
            embedding_function=embed_model,
            persist_directory=persist_directory,
        )
    return _vectorstore_cache[key]


def invalidate_cache(
    persist_directory: str = "./chromaDB",
    collection_name: str = "documents",
) -> None:
    """Drop the cached vectorstore reference (call this before wiping the DB)."""
    key = (persist_directory, collection_name)
    if key in _vectorstore_cache:
        del _vectorstore_cache[key]


def retrieve_docs_for_query(
    query: str,
    top_k: int = 4,
    persist_directory: str = "./chromaDB",
    collection_name: str = "documents",
) -> List[str]:
    """Return top-k document texts from the Chroma vectorstore for `query`."""
    vectorstore = get_vectorstore(persist_directory, collection_name)

    # ✅ Direct call — Chroma always supports this, no fallback maze needed
    docs = vectorstore.similarity_search(query, k=top_k)

    return [doc.page_content for doc in docs]


if __name__ == "__main__":
    q = "What helped Revathi to claim her plants?"
    for i, t in enumerate(retrieve_docs_for_query(q, top_k=3), start=1):
        print(f"--- doc {i} ---")
        print(t[:1000])
