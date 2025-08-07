# memory_manager.py

import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

def get_short_term_memory():
    """
    Initialize short-term conversational memory.
    """
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def load_faiss_vector_store(persist_path="./faiss_store"):
    """
    Load FAISS vector store from disk if it exists, else return None.
    Automatically selects device: CUDA if available, else CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    if os.path.exists(persist_path):
        return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    else:
        return None

def clear_faiss_store(persist_path="./faiss_store"):
    """
    Deletes the FAISS vector store (for resetting long-term memory).
    """
    import shutil
    if os.path.exists(persist_path):
        shutil.rmtree(persist_path)
