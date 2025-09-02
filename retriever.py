
# retriever.py

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, 
    JSONLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, TokenTextSplitter,
    MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import AutoModel, AutoTokenizer
import os
import warnings
import pandas as pd
import json
import sqlite3
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
import re
import torch
import mimetypes

class IntelligentDocumentLoader:
    """Advanced document loader that handles any file type intelligently."""
    
    def __init__(self):
        self.supported_types = {
            '.pdf': self._load_pdf,
            '.txt': self._load_text,
            '.docx': self._load_docx,
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.html': self._load_html,
            '.htm': self._load_html,
            '.md': self._load_markdown,
            '.pptx': self._load_powerpoint,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.xml': self._load_xml,
            '.sql': self._load_sql,
            '.py': self._load_code,
            '.js': self._load_code,
            '.ts': self._load_code,
            '.java': self._load_code,
            '.cpp': self._load_code,
            '.c': self._load_code,
        }
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load any document type with intelligent processing."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in self.supported_types:
            return self.supported_types[file_ext](file_path)
        else:
            # Try to detect file type by content
            return self._load_generic(file_path)
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_text(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    
    def _load_docx(self, file_path: str) -> List[Document]:
        loader = Docx2txtLoader(file_path)
        return loader.load()
    
    def _load_csv(self, file_path: str) -> List[Document]:
        df = pd.read_csv(file_path)
        content = df.to_string(index=False)
        metadata = {
            "source": file_path,
            "type": "csv",
            "rows": len(df),
            "columns": list(df.columns)
        }
        return [Document(page_content=content, metadata=metadata)]
    
    def _load_json(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        content = json.dumps(data, indent=2)
        metadata = {"source": file_path, "type": "json"}
        return [Document(page_content=content, metadata=metadata)]
    
    def _load_html(self, file_path: str) -> List[Document]:
        loader = UnstructuredHTMLLoader(file_path)
        return loader.load()
    
    def _load_markdown(self, file_path: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()
    
    def _load_powerpoint(self, file_path: str) -> List[Document]:
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()
    
    def _load_excel(self, file_path: str) -> List[Document]:
        loader = UnstructuredExcelLoader(file_path)
        return loader.load()
    
    def _load_xml(self, file_path: str) -> List[Document]:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        def xml_to_text(element, level=0):
            text = "  " * level + f"{element.tag}: {element.text or ''}\n"
            for child in element:
                text += xml_to_text(child, level + 1)
            return text
        
        content = xml_to_text(root)
        metadata = {"source": file_path, "type": "xml"}
        return [Document(page_content=content, metadata=metadata)]
    
    def _load_sql(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {"source": file_path, "type": "sql"}
        return [Document(page_content=content, metadata=metadata)]
    
    def _load_code(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        file_ext = os.path.splitext(file_path)[1]
        metadata = {
            "source": file_path, 
            "type": "code", 
            "language": file_ext[1:] if file_ext else "unknown"
        }
        return [Document(page_content=content, metadata=metadata)]
    
    def _load_generic(self, file_path: str) -> List[Document]:
        """Fallback loader for unknown file types."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            metadata = {"source": file_path, "type": "generic"}
            return [Document(page_content=content, metadata=metadata)]
        except:
            # Try binary read and convert to text
            with open(file_path, 'rb') as f:
                content = str(f.read())
            metadata = {"source": file_path, "type": "binary"}
            return [Document(page_content=content[:10000], metadata=metadata)]

class AdaptiveTextSplitter:
    """Intelligent text splitter that adapts to content type."""
    
    def __init__(self):
        self.splitters = {
            "code": TokenTextSplitter(chunk_size=1500, chunk_overlap=200),
            "markdown": MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
            ),
            "html": HTMLHeaderTextSplitter(
                headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
            ),
            "default": RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        }
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents based on their type."""
        all_chunks = []
        
        for doc in documents:
            doc_type = doc.metadata.get("type", "default")
            
            if doc_type in ["py", "js", "ts", "java", "cpp", "c"]:
                splitter = self.splitters["code"]
            elif doc_type == "markdown":
                splitter = self.splitters["markdown"]
            elif doc_type in ["html", "htm"]:
                splitter = self.splitters["html"]
            else:
                splitter = self.splitters["default"]
            
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        return all_chunks

def load_and_split(file_path: str) -> List[Document]:
    """Load and intelligently split any document type."""
    loader = IntelligentDocumentLoader()
    documents = loader.load_document(file_path)
    
    splitter = AdaptiveTextSplitter()
    chunks = splitter.split_documents(documents)
    
    return chunks

def get_optimal_embeddings(model_preference: str = "auto") -> Any:
    """Get the best embedding model available."""
    if model_preference == "openai" and "OPENAI_API_KEY" in os.environ:
        return OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )
    else:
        # Use best open-source model
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        pre_trained_model=AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model=pre_trained_model,
            model_kwargs={"device": "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"}
        )

def build_vector_store(chunks: List[Document], persist_path: str = "./faiss_store") -> FAISS:
    """Build or update FAISS vector store with optimization."""
    embeddings = get_optimal_embeddings()
    
    if os.path.exists(persist_path):
        warnings.warn("Using `allow_dangerous_deserialization=True`. Ensure the FAISS store is trusted.")
        try:
            vs = FAISS.load_local(
                persist_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            # Add new documents
            if chunks:
                vs.add_documents(chunks)
                vs.save_local(persist_path)
        except Exception as e:
            print(f"Error loading existing store: {e}")
            vs = FAISS.from_documents(chunks, embeddings)
            vs.save_local(persist_path)
    else:
        if chunks:
            vs = FAISS.from_documents(chunks, embeddings)
            vs.save_local(persist_path)
        else:
            # Create empty store
            dummy_doc = Document(page_content="initialization", metadata={})
            vs = FAISS.from_documents([dummy_doc], embeddings)
            vs.save_local(persist_path)
    
    return vs

def scrape_url(url: str) -> str:
    """Enhanced web scraping with better content extraction."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "meta", "noscript", "nav", "footer", "header", "aside", "advertisement"]):
            element.decompose()
        
        # Try to find main content
        main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile("content|main|article"))
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        return text[:50000]  # Increased limit for better content
        
    except Exception as e:
        return f"[ERROR] Failed to scrape URL: {str(e)}"

def process_database_connection(connection_string: str, query: str) -> List[Document]:
    """Process database queries and convert to documents."""
    try:
        if connection_string.startswith("sqlite"):
            conn = sqlite3.connect(connection_string.replace("sqlite://", ""))
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            content = df.to_string(index=False)
            metadata = {
                "source": f"database_query: {query}",
                "type": "database",
                "rows": len(df)
            }
            return [Document(page_content=content, metadata=metadata)]
    except Exception as e:
        return [Document(page_content=f"Database error: {str(e)}", metadata={"source": "database_error"})]
