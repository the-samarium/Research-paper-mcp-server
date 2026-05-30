"""
RAG pipeline includes steps:
    Creating a rag application.
    Steps:
    doc load -> text split -> embedding -> storage in vector store -> retriever
"""

import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(source_dir: str = "assets", glob: str = "*.pdf"):
    docs = []
    source_path = Path(source_dir)
    if not source_path.exists():
        return docs

    for pdf_path in source_path.glob(glob):
        if pdf_path.is_file():
            loader = PyPDFLoader(str(pdf_path))
            docs.extend(loader.load())

    return docs


def build_model(model_name: str = "mistral:7b", temperature: float = 0.2):
    return ChatOllama(model=model_name, temperature=temperature)


def build_embed_model(embed_model_name: str = "nomic-embed-text:latest"):
    return OllamaEmbeddings(model=embed_model_name)


def build_vectorstore(
    embedding_function: Any,
    collection_name: str = "news_collection",
    persist_directory: str = "./chromaDB",
):
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )


def wipe_vectorstore(persist_directory: str = "./chromaDB") -> bool:
    store_path = Path(persist_directory)
    if store_path.exists():
        shutil.rmtree(store_path)
        return True
    return False


def rag_embed(
    source_dir: str = "assets",
    glob: str = "*.pdf",
    persist_directory: str = "./chromaDB",
    collection_name: str = "news_collection",
    model_name: str = "mistral:7b",
    embed_model_name: str = "nomic-embed-text:latest",
    chunk_size: int = 550,
    chunk_overlap: int = 25,
    force_rebuild: bool = False,
):
    """Load PDFs, split them, embed chunks, and persist into Chroma."""
    model = build_model(model_name=model_name)

    if force_rebuild and Path(persist_directory).exists():
        shutil.rmtree(persist_directory)

    docs = load_documents(source_dir=source_dir, glob=glob)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embed_model = build_embed_model(embed_model_name=embed_model_name)
    vectorstore = build_vectorstore(
        embedding_function=embed_model,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    if force_rebuild or vectorstore._collection.count() == 0:
        vectorstore.add_documents(chunks)
        print("Documents added to vectorstore.")
    else:
        print("DB already populated.")

    return {
        "model": model,
        "docs": docs,
        "chunks": chunks,
        "vectorstore": vectorstore,
    }
