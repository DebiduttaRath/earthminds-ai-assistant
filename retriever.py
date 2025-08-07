# retriever.py

from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import warnings

def load_and_split(file_path):
    """Load and split document into chunks."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

def build_vector_store(chunks, persist_path="./faiss_store"):
    """Build or update FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(persist_path):
        warnings.warn("Using `allow_dangerous_deserialization=True`. Ensure the FAISS store is trusted.")
        vs = FAISS.load_local(
            persist_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(persist_path)
    return vs


# Optional: web scraping
"""
def scrape_url(url):
    import requests
    from bs4 import BeautifulSoup

    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    text = soup.get_text()
    return text[:10000]  # Limit size
"""
def scrape_url(url):
    import requests
    from bs4 import BeautifulSoup
    import re

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"[ERROR] Failed to fetch URL: {e}"

    soup = BeautifulSoup(res.text, "html.parser")

    # Remove script/style/meta/nav tags
    for tag in soup(["script", "style", "meta", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=' ', strip=True)
    # Remove multiple spaces/newlines
    clean_text = re.sub(r'\s+', ' ', text)

    return clean_text[:10000]
