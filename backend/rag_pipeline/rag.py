"""
RAG pipeline: load PDFs -> split text -> embed -> store in vectorstore -> retrieve
"""

import gc
import os
import shutil
import stat
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(source_dir: str = "assets", glob: str = "*.pdf"):
    """Load PDF documents from a directory."""
    docs = []
    source_path = Path(source_dir)
    if not source_path.exists():
        return docs

    for pdf_path in source_path.glob(glob):
        if pdf_path.is_file():
            loader = PyPDFLoader(str(pdf_path))
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata = doc.metadata or {}
                doc.metadata["source"] = str(pdf_path)
                doc.metadata["source_name"] = pdf_path.name
            docs.extend(loaded_docs)

    return docs


def build_model(model_name: str = "mistral:7b", temperature: float = 0.2):
    """Build an LLM."""
    return ChatOllama(model=model_name, temperature=temperature)


def build_embed_model(embed_model_name: str = "nomic-embed-text:latest"):
    """Build an embedding model."""
    return OllamaEmbeddings(model=embed_model_name)


def build_vectorstore(
    embedding_function,
    collection_name: str = "documents",
    persist_directory: str = "./chromaDB",
):
    """Build and return a Chroma vectorstore."""
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )


def wipe_vectorstore(persist_directory: str = "./chromaDB") -> bool:
    """Delete the persisted vectorstore directory."""
    store_path = Path(persist_directory)
    if not store_path.exists():
        return False

    def _on_rm_error(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            pass
        try:
            func(path)
        except Exception:
            pass

    # Best-effort removal with retries to handle transient Windows locks.
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            gc.collect()
            shutil.rmtree(store_path, onerror=_on_rm_error)
            return True
        except PermissionError:
            if attempt == max_attempts:
                raise
            time.sleep(0.5 * attempt)
    return False


def rag_embed(
    source_dir: str = "assets",
    glob: str = "*.pdf",
    persist_directory: str = "./chromaDB",
    collection_name: str = "documents",
    model_name: str = "mistral:7b",
    embed_model_name: str = "nomic-embed-text:latest",
    chunk_size: int = 550,
    chunk_overlap: int = 25,
    force_rebuild: bool = False,
):
    """Load PDFs, split, embed, and store in vectorstore."""
    # Load and prepare documents
    docs = load_documents(source_dir=source_dir, glob=glob)
    if not docs:
        print("No documents found in assets.")
        return None

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # Build models
    model = build_model(model_name=model_name)
    embed_model = build_embed_model(embed_model_name=embed_model_name)

    # Handle rebuild
    # Handle rebuild
    if force_rebuild:
        wipe_vectorstore(persist_directory)  #  uses retry + chmod logic

    # Create vectorstore and add chunks
    vectorstore = build_vectorstore(
        embedding_function=embed_model,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    vectorstore.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to vectorstore.")

    return {
        "model": model,
        "vectorstore": vectorstore,
    }
