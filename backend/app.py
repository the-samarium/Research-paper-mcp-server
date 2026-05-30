import os

import feedparser
import requests
from fastapi import FastAPI, HTTPException

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
    # 1. Check if arXiv actually returned a valid 200 response
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
async def search_and_ingest(query: str, top_k: int = 5):
    """
    Search papers, download PDFs,
    extract text, chunk, embed and store.
    """
    response = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": top_k},
    )

    # 1. Check if arXiv actually returned a valid 200 response
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to fetch from arXiv"
        )

    # 2. Parse the Atom/XML feed using feedparser
    feed = feedparser.parse(response.text)

    papers = []
    pdf_urls = []
    for entry in feed.entries:
        # arXiv provides the PDF link inside the entry links
        pdf_link = ""
        for link in entry.links:
            if link.get("title") == "pdf" or "pdf" in link.get("href", ""):
                pdf_link = link.get("href")
                break

        pdf_urls.append((pdf_link,entry.id.split("/abs/")[-1]))
        papers.append(
            {
                "id": entry.id.split("/abs/")[-1],  # Extracts the unique arXiv ID
                "title": entry.title.strip().replace("\n", " "),
                "summary": entry.summary.strip().replace("\n", " "),
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
                    # Sanitize the ID so it's a safe filename (removes slashes if any exist)
                    safe_id = arxiv_id.replace("/", "_")
                    filename = f"assets/paper_{safe_id}.pdf"

                    with open(filename, "wb") as f:
                        f.write(pdf_response.content)
                else:
                    print(f"Failed to download PDF for ID {arxiv_id}: Status {pdf_response.status_code}")
            except Exception as e:
                print(f"Error downloading {pdf_link}: {e}")

    # 3. Return the parsed Python list (FastAPI automatically serializes this to JSON)
    return papers
