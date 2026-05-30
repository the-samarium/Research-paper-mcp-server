import os

import feedparser
import requests
from fastapi import FastAPI, HTTPException
from rag_pipeline.rag import rag_embed, wipe_vectorstore

app = FastAPI()


@app.get("/search")
async def get_papers(query: str):
    """
    Get a bulk list of 10 papers
    """
    response = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": 10},
    )
    #  checking if arXiv actually returned a valid 200 response
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
    """
    Search papers, download PDFs,
    extract text, chunk, embed and store.

    If `wipe_db=true`, the existing persisted vector DB is cleared before re-indexing.
    """
    response = requests.get(
        url="http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": top_k},
    )

    # Check if arXiv actually returned a valid 200 response
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to fetch from arXiv"
        )

    # Parse the Atom/XML feed using feedparser
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
        # arXiv provides the PDF link inside the entry links
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

    # download the pdfs from the returned urls and store in a new /assets directory
    os.makedirs("assets", exist_ok=True)
    for pdf_link, arxiv_id in pdf_urls:
        if not pdf_link:
            continue  # Skip if no PDF link was found for this entry

        try:
            pdf_response = requests.get(pdf_link)
            if pdf_response.status_code == 200:
                safe_id = arxiv_id.replace("/", "_")
                filename = f"assets/paper_{safe_id}.pdf"

                with open(filename, "wb") as f:
                    f.write(pdf_response.content)
            else:
                print(
                    f"Failed to download PDF for ID {arxiv_id}: Status {pdf_response.status_code}"
                )
        except Exception as e:
            print(f"Error downloading {pdf_link}: {e}")

    rag_op = rag_embed(force_rebuild=wipe_db)

    #  Return only JSON-serializable metadata from the RAG operation - solved this error here
    return {
        "papers": papers,
        "ingested_documents": len(rag_op["docs"]),
        "indexed_chunks": len(rag_op["chunks"]),
        "wipe_db": wipe_db,
    }


# an endpoint to wipe the existing vector db
@app.get("/wipe")
async def wipe_chroma(persist_directory: str = "./chromaDB"):
    """Delete the persisted Chroma vector database directory."""
    wiped = wipe_vectorstore(persist_directory)
    if wiped:
        return {
            "wiped": True,
            "message": f"Vector DB wiped at {persist_directory}",
        }

    return {
        "wiped": False,
        "message": f"No vector DB found at {persist_directory}",
    }
