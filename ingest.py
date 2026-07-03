"""
ingest.py
---------
Step 2 of the pipeline: turn an uploaded file into
  (a) a Chroma vector store (for text/PDF/CSV -> semantic search / RAG)
  (b) a pandas DataFrame (for CSV/Excel -> numeric analysis)

Why both? A marketing report PDF needs semantic search ("what did the report
say about Instagram engagement?"). A campaign performance CSV needs real
math (sums, averages, group-by). One tool can't do both well, so the agent
(in agent.py) will be given both tools and decides which to call.
"""

import os
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"  # pull with: ollama pull nomic-embed-text


def load_documents(file_path: str):
    """Load a file into LangChain Document objects for chunking + embedding."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8").load()
    elif ext == ".csv":
        # Also index CSV rows as text so qualitative/text columns
        # (e.g. campaign notes, ad copy) are searchable.
        return CSVLoader(file_path).load()
    return []


def load_dataframe(file_path: str):
    """Load tabular data into pandas for numeric analysis (used by the
    analyze_marketing_data tool)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    return None


def build_vectorstore(file_path: str, collection_name: str = "marketing_data"):
    """Chunk documents and embed them locally via Ollama, storing vectors
    in a local Chroma DB (no cloud, no API key, fully free)."""
    docs = load_documents(file_path)
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore
