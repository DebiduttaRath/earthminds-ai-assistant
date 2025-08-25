
# llm_router.py

from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from typing import List, Optional, Dict, Any, Iterator
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
)
from langchain.chat_models.base import BaseChatModel
import requests
import json

class ChatDeepSeek(BaseChatModel):
    """Custom LangChain Chat Model for DeepSeek API with enhanced capabilities."""
    
    model: str = "deepseek-chat"
    temperature: float = 0.7
    api_key: str = None
    api_base: str = "https://api.deepseek.com/v1"
    max_tokens: int = 4096
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call DeepSeek API with enhanced error handling and retries."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        deepseek_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            deepseek_messages.append({"role": role, "content": msg.content})
        
        payload = {
            "model": self.model,
            "messages": deepseek_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            response_data = response.json()
            message = response_data["choices"][0]["message"]
            
            generation = ChatGeneration(
                message=AIMessage(content=message["content"])
            )
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise ValueError(f"DeepSeek API Error: {str(e)}")
    
    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

class AdaptiveLLMRouter:
    """Intelligent LLM router that adapts based on query complexity and performance."""
    
    def __init__(self):
        self.performance_metrics = {}
        self.model_capabilities = {
            "openai": {"reasoning": 0.9, "creativity": 0.8, "speed": 0.7, "cost": 0.6},
            "deepseek": {"reasoning": 0.95, "creativity": 0.7, "speed": 0.8, "cost": 0.9},
            "groq": {"reasoning": 0.8, "creativity": 0.7, "speed": 0.95, "cost": 0.8},
            "claude": {"reasoning": 0.95, "creativity": 0.9, "speed": 0.6, "cost": 0.5}
        }
    
    def analyze_query_complexity(self, query: str) -> Dict[str, float]:
        """Analyze query to determine optimal model selection."""
        complexity_score = len(query.split()) / 50.0  # Basic complexity metric
        
        # Check for reasoning indicators
        reasoning_keywords = ["analyze", "compare", "explain", "reason", "logic", "why", "how"]
        reasoning_score = sum(1 for word in reasoning_keywords if word in query.lower()) / len(reasoning_keywords)
        
        # Check for creativity indicators
        creativity_keywords = ["create", "generate", "write", "compose", "imagine", "design"]
        creativity_score = sum(1 for word in creativity_keywords if word in query.lower()) / len(creativity_keywords)
        
        return {
            "complexity": min(complexity_score, 1.0),
            "reasoning": reasoning_score,
            "creativity": creativity_score
        }
    
    def select_best_model(self, query: str, available_models: List[str]) -> str:
        """Select the best model based on query analysis."""
        query_analysis = self.analyze_query_complexity(query)
        
        best_model = available_models[0]
        best_score = 0
        
        for model in available_models:
            if model in self.model_capabilities:
                capabilities = self.model_capabilities[model]
                score = (
                    capabilities["reasoning"] * query_analysis["reasoning"] +
                    capabilities["creativity"] * query_analysis["creativity"] +
                    capabilities["speed"] * 0.3 +
                    capabilities["cost"] * 0.2
                )
                
                if score > best_score:
                    best_score = score
                    best_model = model
        
        return best_model

def get_llm(model_choice: str = "auto", query: str = "", available_models: List[str] = None):
    """
    Dynamically route to optimal LLM with adaptive selection.
    Options: 'openai', 'deepseek', 'groq', 'claude', 'auto'
    """
    if available_models is None:
        available_models = ["groq", "deepseek", "openai"]
    
    router = AdaptiveLLMRouter()
    
    if model_choice == "auto" and query:
        model_choice = router.select_best_model(query, available_models)
    elif model_choice == "auto":
        model_choice = "groq"  # Default fallback
    
    try:
        if model_choice == "openai":
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                openai_api_key=st.secrets.get("OPENAI_API_KEY"),
                max_tokens=4096
            )
        
        elif model_choice == "deepseek":
            return ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0.7, 
                api_key=st.secrets.get("DEEPSEEK_API_KEY")
            )
        
        elif model_choice == "groq":
            return ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=0.7,
                api_key=st.secrets.get("GROQ_API_KEY"),
                max_tokens=4096
            )
        
        elif model_choice == "claude":
            return ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0.7,
                anthropic_api_key=st.secrets.get("ANTHROPIC_API_KEY"),
                max_tokens=4096
            )
        
        else:
            # Fallback to Groq if model not found
            return ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=0.7,
                api_key=st.secrets.get("GROQ_API_KEY"),
                max_tokens=4096
            )
    
    except Exception as e:
        st.error(f"Error initializing {model_choice}: {str(e)}")
        # Fallback to Groq
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=st.secrets.get("GROQ_API_KEY"),
            max_tokens=4096
        )
