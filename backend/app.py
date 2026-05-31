import gc
import os

import feedparser
import requests
from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rag_pipeline.rag import build_model, rag_embed, wipe_vectorstore
from rag_pipeline.retriver import invalidate_cache, retrieve_docs_for_query

app = FastAPI()

PERSIST_DIR = "./chromaDB"
model = build_model()  # model defines
parser = StrOutputParser()

# Module-level cache so we can explicitly release the vectorstore before wiping
_rag_state: dict | None = None


@app.get("/search")
async def get_papers(query: str):
    """Get a bulk list of 10 papers"""
    response = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": 10},
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to fetch from arXiv"
        )

    feed = feedparser.parse(response.text)
    papers = []
    for entry in feed.entries:
        papers.append(
            {
                "title": entry.title,
                "summary": entry.summary,
                "published": entry.published,
                "authors": [author.name for author in entry.authors],
            }
        )
    return papers


@app.get("/search/rag")
async def search_and_ingest(query: str, top_k: int = 5, wipe_db: bool = False):
    global _rag_state

    response = requests.get(
        url="http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": top_k},
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to fetch from arXiv"
        )

    feed = feedparser.parse(response.text)

    def parse_string(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            return parse_string(value[0])
        return ""

    papers = []
    pdf_urls = []
    for entry in feed.entries:
        pdf_link = ""
        for link in entry.links:
            href = parse_string(link.get("href", ""))
            link_title = parse_string(link.get("title", ""))
            if link_title.lower() == "pdf" or "pdf" in href.lower():
                pdf_link = href
                break

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
                "authors": [author.name for author in entry.authors],
                "pdf_url": pdf_link,
            }
        )

    os.makedirs("assets", exist_ok=True)
    for pdf_link, arxiv_id in pdf_urls:
        if not pdf_link:
            continue
        try:
            pdf_response = requests.get(pdf_link)
            if pdf_response.status_code == 200:
                safe_id = arxiv_id.replace("/", "_")
                with open(f"assets/paper_{safe_id}.pdf", "wb") as f:
                    f.write(pdf_response.content)
            else:
                print(
                    f"Failed to download PDF for {arxiv_id}: {pdf_response.status_code}"
                )
        except Exception as e:
            print(f"Error downloading {pdf_link}: {e}")

    #  Release the cached vectorstore BEFORE wipe so Windows unlocks the files
    if wipe_db and _rag_state is not None:
        invalidate_cache(PERSIST_DIR)
        del _rag_state["vectorstore"]
        _rag_state = None
        gc.collect()

    rag_op = rag_embed(force_rebuild=wipe_db, persist_directory=PERSIST_DIR)

    if rag_op is None:
        raise HTTPException(
            status_code=500, detail="RAG embed failed — no documents found in assets/"
        )

    # Cache the new state
    _rag_state = rag_op

    #  rag_embed returns "model" and "vectorstore", not "docs"/"chunks"
    # Return the vectorstore document count instead
    collection = rag_op["vectorstore"]._collection
    return {
        "papers": papers,
        "indexed_chunks": collection.count(),
        "wipe_db": wipe_db,
    }


@app.get("/wipe")
async def wipe_chroma(persist_directory: str = "./chromaDB"):
    global _rag_state

    #  Release vectorstore reference before wiping
    invalidate_cache(PERSIST_DIR)
    if _rag_state is not None:
        del _rag_state["vectorstore"]
        _rag_state = None
        gc.collect()

    wiped = wipe_vectorstore(persist_directory)
    if wiped:
        return {"wiped": True, "message": f"Vector DB wiped at {persist_directory}"}
    return {"wiped": False, "message": f"No vector DB found at {persist_directory}"}


@app.get("/query")
async def retrieve(query: str, top_k: int = 4):
    """Answer a query using the persisted Chroma vectorstore and LLM."""
    docs = retrieve_docs_for_query(
        query=query, top_k=top_k, persist_directory=PERSIST_DIR
    )
    if not docs:
        raise HTTPException(
            status_code=404, detail="No relevant documents found for the query."
        )

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful research assistant. Answer the question using only the context provided below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    chain = prompt | model | parser
    res = chain.invoke(
        {
            "context": "\n\n".join(docs),
            "question": query,
        }
    )
    return {
        "answer":res,
        "parsed_docs": len(docs)
    }  # already a dict after JsonOutputParser