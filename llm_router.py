# llm_router.py

from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq


from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import streamlit as st  # Use Streamlit secrets

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

class ChatDeepSeek(BaseChatModel):
    """Custom LangChain Chat Model for DeepSeek API."""
    
    model: str = "deepseek-chat"
    temperature: float = 0.7
    api_key: str = None
    api_base: str = "https://api.deepseek.com/v1"  # Verify the correct API URL
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call DeepSeek API and return the ChatResult."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Convert LangChain messages to DeepSeek format
        deepseek_messages = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": deepseek_messages,
            "temperature": self.temperature,
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
        )
        
        if response.status_code != 200:
            raise ValueError(f"DeepSeek API Error: {response.text}")
        
        response_data = response.json()
        message = response_data["choices"][0]["message"]
        
        # Convert the response into a LangChain ChatResult
        generation = ChatGeneration(
            message=AIMessage(content=message["content"])
        )
        
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

        
        
def get_llm(model_choice: str = "openai"):
    """
    Dynamically route to selected LLM.
    Options: 'openai', 'deepseek', 'groq'
    """
    if model_choice == "openai":
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )

    elif model_choice == "deepseek":
        return ChatDeepSeek(model="deepseek-chat", temperature=0.7, api_key=st.secrets["DEEPSEEK_API_KEY"])

    elif model_choice == "groq":
        return ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",  #"llama3-70b-8192",
            temperature=0.7,
            api_key=st.secrets["GROQ_API_KEY"]
        )

    else:
        raise ValueError(f"Unsupported model: {model_choice}")
   