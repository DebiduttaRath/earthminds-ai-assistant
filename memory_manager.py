
# memory_manager.py

import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
import pickle
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import hashlib

class AdvancedMemoryManager:
    """Advanced memory management with auto-learning capabilities."""
    
    def __init__(self, persist_path: str = "./memory_store"):
        self.persist_path = persist_path
        self.conversation_history = []
        self.learning_patterns = {}
        self.user_preferences = {}
        self.context_memory = {}
        
        os.makedirs(persist_path, exist_ok=True)
        self._load_persistent_memory()
    
    def _load_persistent_memory(self):
        """Load persistent memory from disk."""
        try:
            with open(os.path.join(self.persist_path, "learning_patterns.json"), "r") as f:
                self.learning_patterns = json.load(f)
        except FileNotFoundError:
            self.learning_patterns = {}
        
        try:
            with open(os.path.join(self.persist_path, "user_preferences.json"), "r") as f:
                self.user_preferences = json.load(f)
        except FileNotFoundError:
            self.user_preferences = {}
    
    def _save_persistent_memory(self):
        """Save persistent memory to disk."""
        with open(os.path.join(self.persist_path, "learning_patterns.json"), "w") as f:
            json.dump(self.learning_patterns, f, indent=2)
        
        with open(os.path.join(self.persist_path, "user_preferences.json"), "w") as f:
            json.dump(self.user_preferences, f, indent=2)
    
    def learn_from_interaction(self, query: str, response: str, feedback: Optional[str] = None):
        """Learn from user interactions to improve future responses."""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        
        interaction_data = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        }
        
        if query_hash not in self.learning_patterns:
            self.learning_patterns[query_hash] = []
        
        self.learning_patterns[query_hash].append(interaction_data)
        
        # Learn user preferences
        self._extract_preferences(query, response)
        self._save_persistent_memory()
    
    def _extract_preferences(self, query: str, response: str):
        """Extract user preferences from interactions."""
        # Simple preference extraction (can be enhanced with NLP)
        query_lower = query.lower()
        
        if "detailed" in query_lower or "explain" in query_lower:
            self.user_preferences["detail_level"] = "high"
        elif "brief" in query_lower or "summary" in query_lower:
            self.user_preferences["detail_level"] = "low"
        
        if "code" in query_lower or "programming" in query_lower:
            self.user_preferences["preferred_domains"] = self.user_preferences.get("preferred_domains", [])
            if "programming" not in self.user_preferences["preferred_domains"]:
                self.user_preferences["preferred_domains"].append("programming")
    
    def get_relevant_context(self, query: str, max_context: int = 3) -> str:
        """Get relevant context from past interactions."""
        query_lower = query.lower()
        relevant_interactions = []
        
        for query_hash, interactions in self.learning_patterns.items():
            for interaction in interactions[-3:]:  # Last 3 interactions for each pattern
                if any(word in interaction["query"].lower() for word in query_lower.split()):
                    relevant_interactions.append(interaction)
        
        # Sort by relevance and recency
        relevant_interactions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        context_parts = []
        for interaction in relevant_interactions[:max_context]:
            context_parts.append(f"Previous: Q: {interaction['query']} A: {interaction['response'][:200]}...")
        
        return "\n".join(context_parts)
    
    def optimize_memory_usage(self):
        """Optimize memory by removing old or irrelevant data."""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for query_hash in list(self.learning_patterns.keys()):
            interactions = self.learning_patterns[query_hash]
            recent_interactions = [
                interaction for interaction in interactions
                if datetime.fromisoformat(interaction["timestamp"]) > cutoff_date
            ]
            
            if recent_interactions:
                self.learning_patterns[query_hash] = recent_interactions
            else:
                del self.learning_patterns[query_hash]
        
        self._save_persistent_memory()

def get_short_term_memory(llm=None):
    """Initialize enhanced short-term conversational memory."""
    if llm:
        return ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000
        )
    else:
        return ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )

def load_faiss_vector_store(persist_path: str = "./faiss_store"):
    """Load FAISS vector store with enhanced embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try to use best available embeddings
    try:
        if "OPENAI_API_KEY" in os.environ:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": device}
            )
    except:
        # Fallback to basic embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
    
    if os.path.exists(persist_path):
        try:
            return FAISS.load_local(
                persist_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading FAISS store: {e}")
            return None
    else:
        return None

def clear_faiss_store(persist_path: str = "./faiss_store"):
    """Delete FAISS vector store and memory data."""
    import shutil
    if os.path.exists(persist_path):
        shutil.rmtree(persist_path)
    
    # Also clear learning patterns
    memory_path = "./memory_store"
    if os.path.exists(memory_path):
        shutil.rmtree(memory_path)

def auto_optimize_retrieval(vectorstore: FAISS, query: str) -> Dict[str, Any]:
    """Auto-optimize retrieval parameters based on query characteristics."""
    # Analyze query complexity
    query_length = len(query.split())
    
    if query_length <= 5:
        # Short query - increase k, decrease score threshold
        k = 8
        score_threshold = 0.3
    elif query_length <= 15:
        # Medium query - balanced parameters
        k = 5
        score_threshold = 0.5
    else:
        # Long query - fewer results, higher threshold
        k = 3
        score_threshold = 0.7
    
    return {
        "k": k,
        "score_threshold": score_threshold,
        "fetch_k": k * 2
    }
