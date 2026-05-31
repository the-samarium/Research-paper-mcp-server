import asyncio
import gc
import os
import shutil

import feedparser
import requests
from fastapi import HTTPException
from backend.logging import logger
from backend.rag_pipeline.rag import rag_embed, wipe_vectorstore
from backend.rag_pipeline.retriver import invalidate_cache, retrieve_docs_for_query

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIR = os.path.join(BASE_DIR, "chromaDB")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

_rag_state: dict | None = None


def parse_string(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return parse_string(value[0])
    return ""


def _get_papers_sync(query: str):
    try:
        response = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": 20},
            timeout=30,
        )
        response.raise_for_status()
        feed = feedparser.parse(response.text)
        return [
            {
                "title": entry.title,
                "summary": entry.summary,
                "published": entry.published,
                "authors": [author.name for author in entry.authors],
            }
            for entry in feed.entries
        ]
    except requests.exceptions.Timeout:
        logger.error("arXiv request timed out")
        raise HTTPException(status_code=504, detail="arXiv request timed out")
    except requests.exceptions.ConnectionError:
        logger.error("Unable to connect to arXiv")
        raise HTTPException(status_code=503, detail="Unable to connect to arXiv")
    except requests.exceptions.HTTPError as e:
        logger.error(f"arXiv returned an error: {e}")
        raise HTTPException(status_code=502, detail=f"arXiv returned an error: {e}")

async def fn_get_papers(query: str):
    try:
        return await asyncio.to_thread(_get_papers_sync, query)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"fn_get_papers unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

async def fn_search_and_ingest(query: str, top_k: int, wipe_db: bool):
    try:
        return await asyncio.to_thread(_search_and_ingest_sync, query, top_k, wipe_db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"fn_search_and_ingest unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def _search_and_ingest_sync(query: str, top_k: int, wipe_db: bool):
    global _rag_state
    try:
        response = requests.get(
            url="http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": top_k},
            timeout=30,
        )
        response.raise_for_status()

        feed = feedparser.parse(response.text)
        papers, pdf_urls = [], []

        for entry in feed.entries:
            pdf_link = next(
                (
                    parse_string(link.get("href", ""))
                    for link in entry.links
                    if parse_string(link.get("title", "")).lower() == "pdf"
                    or "pdf" in parse_string(link.get("href", "")).lower()
                ),
                "",
            )
            entry_id = parse_string(entry.get("id", ""))
            arxiv_id = entry_id.split("/abs/")[-1] if entry_id else ""
            title = parse_string(entry.get("title", "")).strip().replace("\n", " ")
            summary = parse_string(entry.get("summary", "")).strip().replace("\n", " ")

            pdf_urls.append((pdf_link, arxiv_id))
            papers.append(
                {
                    "id": arxiv_id,
                    "title": title,
                    "summary": summary,
                    "published": entry.published,
                    "authors": [a.name for a in entry.authors],
                    "pdf_url": pdf_link,
                }
            )

        os.makedirs(ASSETS_DIR, exist_ok=True)
        for pdf_link, arxiv_id in pdf_urls:
            if not pdf_link:
                continue
            try:
                pdf_response = requests.get(pdf_link, timeout=30)
                pdf_response.raise_for_status()
                safe_id = arxiv_id.replace("/", "_")
                with open(os.path.join(ASSETS_DIR, f"paper_{safe_id}.pdf"), "wb") as f:
                    f.write(pdf_response.content)
            except Exception as e:
                logger.warning(f"Failed to download PDF for {arxiv_id}: {e}")

        if wipe_db and _rag_state is not None:
            invalidate_cache(PERSIST_DIR)
            del _rag_state["vectorstore"]
            _rag_state = None
            gc.collect()

        rag_op = rag_embed(force_rebuild=wipe_db, persist_directory=PERSIST_DIR)
        if rag_op is None:
            raise HTTPException(
                status_code=500,
                detail="RAG embed failed — no documents found in assets/",
            )

        _rag_state = rag_op
        collection = rag_op["vectorstore"]._collection
        return {
            "papers": papers,
            "indexed_chunks": collection.count(),
            "wipe_db": wipe_db,
        }
    except HTTPException:
        raise
    except requests.exceptions.Timeout:
        logger.error("arXiv request timed out during ingest")
        raise HTTPException(status_code=504, detail="arXiv request timed out")
    except requests.exceptions.ConnectionError:
        logger.error("Unable to connect to arXiv during ingest")
        raise HTTPException(status_code=503, detail="Unable to connect to arXiv")
    except requests.exceptions.HTTPError as e:
        logger.error(f"arXiv HTTP error during ingest: {e}")
        raise HTTPException(status_code=502, detail=f"arXiv returned an error: {e}")
    except Exception as e:
        logger.error(f"_search_and_ingest_sync unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


async def fn_wipe_chroma(persist_directory: str = PERSIST_DIR):
    try:
        return await asyncio.to_thread(_wipe_chroma_sync, persist_directory)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"fn_wipe_chroma unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def _wipe_chroma_sync(persist_directory: str):
    global _rag_state
    try:
        invalidate_cache(PERSIST_DIR)
        if _rag_state is not None:
            del _rag_state["vectorstore"]
            _rag_state = None
            gc.collect()

        wiped = wipe_vectorstore(persist_directory)
        if wiped:
            return {"wiped": True, "message": f"Vector DB wiped at {persist_directory}"}
        return {"wiped": False, "message": f"No vector DB found at {persist_directory}"}
    except Exception as e:
        logger.error(f"_wipe_chroma_sync error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to wipe Chroma: {e}")


async def fn_retrieve(query: str, top_k: int):
    try:
        return await asyncio.to_thread(_retrieve_sync, query, top_k)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"fn_retrieve unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def _retrieve_sync(query: str, top_k: int):
    try:
        docs = retrieve_docs_for_query(
            query=query, top_k=top_k, persist_directory=PERSIST_DIR
        )
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        return {"docs": docs, "count": len(docs)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"_retrieve_sync error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")


async def fn_clear_assets():
    try:
        return await asyncio.to_thread(_clear_assets_sync)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"fn_clear_assets unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def _clear_assets_sync():
    try:
        if not os.path.exists(ASSETS_DIR):
            raise HTTPException(status_code=404, detail="Assets directory not found.")
        shutil.rmtree(ASSETS_DIR)
        return {"cleared": True, "message": "Assets directory deleted."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"_clear_assets_sync error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear assets: {e}")