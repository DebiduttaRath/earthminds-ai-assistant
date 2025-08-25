
# app.py

import streamlit as st
from llm_router import get_llm, AdaptiveLLMRouter
from retriever import load_and_split, build_vector_store, scrape_url, IntelligentDocumentLoader
from memory_manager import (
    get_short_term_memory, load_faiss_vector_store, clear_faiss_store,
    AdvancedMemoryManager, auto_optimize_retrieval
)
from utils import save_uploaded_file, format_chat_history, embed_pdf_display
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os
import time
from datetime import datetime
import json

# Configure Streamlit
st.set_page_config(
    page_title="🧠 EarthMinds AI Assistant Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🤖 EarthMinds AI Assistant Pro")
st.markdown("*The most intelligent RAG system on the planet*")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = AdvancedMemoryManager()

if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = {
        "total_queries": 0,
        "avg_response_time": 0,
        "successful_responses": 0
    }

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🔧 Advanced Settings")
    
    # Model Selection
    available_models = ["auto", "groq", "deepseek", "openai", "claude"]
    llm_choice = st.selectbox(
        "Choose AI Model", 
        available_models,
        help="Auto will intelligently select the best model for each query"
    )
    
    # Advanced Settings
    with st.expander("🎛️ Advanced Options"):
        auto_learn = st.checkbox("Auto-Learning", value=True, help="Learn from interactions to improve responses")
        adaptive_retrieval = st.checkbox("Adaptive Retrieval", value=True, help="Optimize retrieval based on query type")
        context_enhancement = st.checkbox("Context Enhancement", value=True, help="Use conversation history for better context")
        
        max_chunks = st.slider("Max Retrieval Chunks", 3, 15, 5)
        temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, 0.1)
    
    # Performance Metrics
    st.markdown("### 📊 Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", st.session_state.performance_metrics["total_queries"])
    with col2:
        st.metric("Success Rate", f"{st.session_state.performance_metrics.get('success_rate', 0):.1%}")
    
    # Memory Management
    st.markdown("### 🧠 Memory Management")
    if st.button("🗑️ Clear All Memory"):
        clear_faiss_store()
        st.session_state.memory_manager = AdvancedMemoryManager()
        if "memory" in st.session_state:
            del st.session_state.memory
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        st.success("All memory cleared.")
        st.rerun()
    
    if st.button("⚡ Optimize Memory"):
        st.session_state.memory_manager.optimize_memory_usage()
        st.success("Memory optimized.")

# Initialize core components
if "memory" not in st.session_state:
    st.session_state.memory = get_short_term_memory()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_faiss_vector_store()

# --- Document Upload Section ---
st.markdown("## 📄 Document Processing")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload Documents (Any Format)", 
        accept_multiple_files=True,
        type=None,
        help="Supports PDF, DOCX, TXT, CSV, JSON, HTML, MD, PPTX, XLSX, and more!"
    )

with col2:
    st.markdown("### 📈 Document Stats")
    if st.session_state.vectorstore:
        try:
            doc_count = st.session_state.vectorstore.index.ntotal
            st.metric("Documents in Memory", doc_count)
        except:
            st.metric("Documents in Memory", "Available")

# Process uploaded files
if uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        try:
            # Save file
            file_path = save_uploaded_file(uploaded_file)
            
            # Load and process
            chunks = load_and_split(file_path)
            
            # Update vector store
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = build_vector_store(chunks)
            else:
                st.session_state.vectorstore.add_documents(chunks)
                st.session_state.vectorstore.save_local("./faiss_store")
            
            # Show preview for PDFs
            if file_path.endswith(".pdf"):
                with st.expander(f"📄 Preview: {uploaded_file.name}"):
                    st.markdown(embed_pdf_display(file_path), unsafe_allow_html=True)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    status_text.text("✅ All documents processed successfully!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

# --- Web Content Section ---
st.markdown("## 🌐 Web Content Integration")

col1, col2 = st.columns([3, 1])

with col1:
    url = st.text_input(
        "Enter URL to learn from",
        placeholder="https://example.com/article",
        help="Extract and learn from web content"
    )

with col2:
    if url and st.button("🔍 Learn from Web", use_container_width=True):
        with st.spinner("Extracting content..."):
            scraped_text = scrape_url(url)
            
            if not scraped_text.startswith("[ERROR]"):
                # Create document
                doc = Document(
                    page_content=scraped_text, 
                    metadata={
                        "source": url,
                        "type": "web_content",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Add to vector store
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = build_vector_store([doc])
                else:
                    st.session_state.vectorstore.add_documents([doc])
                    st.session_state.vectorstore.save_local("./faiss_store")
                
                st.success(f"✅ Successfully learned from: {url}")
                
                # Show preview
                with st.expander("📄 Content Preview"):
                    st.text_area("Extracted Content", scraped_text[:2000], height=200)
            else:
                st.error(scraped_text)

# --- Chat Interface ---
st.markdown("## 💬 Intelligent Chat")

# Query input
user_query = st.text_area(
    "Ask me anything...",
    height=100,
    placeholder="Enter your question here. I can analyze documents, answer questions, help with code, and much more!",
    help="I can handle complex queries, multi-step reasoning, and provide contextual responses based on your documents."
)

# Chat controls
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    send_button = st.button("🚀 Send", use_container_width=True)

with col2:
    clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)

if clear_chat:
    st.session_state.memory = get_short_term_memory()
    st.rerun()

# Process query
if (send_button or user_query) and user_query.strip():
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload documents or add web content first to enable intelligent responses.")
    else:
        start_time = time.time()
        
        try:
            with st.spinner("🧠 Thinking..."):
                # Get optimal LLM
                llm = get_llm(llm_choice, query=user_query)
                
                # Auto-optimize retrieval if enabled
                if adaptive_retrieval:
                    retrieval_params = auto_optimize_retrieval(st.session_state.vectorstore, user_query)
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={
                            "k": retrieval_params["k"],
                            "fetch_k": retrieval_params["fetch_k"]
                        }
                    )
                else:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": max_chunks}
                    )
                
                # Enhanced prompt template
                custom_prompt = PromptTemplate(
                    template="""You are an advanced AI assistant with access to comprehensive knowledge. 
                    Use the following context and conversation history to provide accurate, detailed, and helpful responses.
                    
                    Context from documents:
                    {context}
                    
                    Conversation History:
                    {chat_history}
                    
                    Current Question: {question}
                    
                    Instructions:
                    - Provide comprehensive and accurate answers
                    - Cite sources when possible
                    - If information is not in the context, clearly state this
                    - Use examples and explanations to clarify complex concepts
                    - Be conversational but professional
                    
                    Response:""",
                    input_variables=["context", "chat_history", "question"]
                )
                
                # Create QA chain
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=st.session_state.memory,
                    verbose=True,
                    combine_docs_chain_kwargs={"prompt": custom_prompt}
                )
                
                # Get relevant context if enabled
                if context_enhancement:
                    relevant_context = st.session_state.memory_manager.get_relevant_context(user_query)
                    if relevant_context:
                        enhanced_query = f"{user_query}\n\nRelevant context from previous conversations:\n{relevant_context}"
                    else:
                        enhanced_query = user_query
                else:
                    enhanced_query = user_query
                
                # Generate response
                result = qa_chain.run(enhanced_query)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Update performance metrics
                st.session_state.performance_metrics["total_queries"] += 1
                st.session_state.performance_metrics["successful_responses"] += 1
                
                # Auto-learning
                if auto_learn:
                    st.session_state.memory_manager.learn_from_interaction(user_query, result)
                
                # Display response
                st.markdown("### 🤖 Response")
                st.markdown(result)
                
                # Response metadata
                with st.expander("📊 Response Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{response_time:.2f}s")
                    with col2:
                        st.metric("Model Used", llm_choice)
                    with col3:
                        st.metric("Chunks Retrieved", max_chunks)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.performance_metrics["total_queries"] += 1

# --- Chat History ---
if st.session_state.memory.chat_memory.messages:
    with st.expander("📜 Conversation History", expanded=False):
        st.markdown(format_chat_history(st.session_state.memory.chat_memory.messages))

# --- Footer ---
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, LangChain, and advanced AI models | "
    f"Session Queries: {st.session_state.conversation_count}"
)

# Hide Streamlit branding
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.css-1q1n0ol {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
