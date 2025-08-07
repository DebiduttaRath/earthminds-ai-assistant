# app.py

import streamlit as st
from llm_router import get_llm
from retriever import load_and_split, build_vector_store, scrape_url
from memory_manager import get_short_term_memory, load_faiss_vector_store, clear_faiss_store
from utils import save_uploaded_file, format_chat_history, embed_pdf_display
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="🧠 EarthMinds AI Assistant", layout="wide")
st.title("🤖 EarthMinds AI Assistant")

# --- Sidebar ---
st.sidebar.header("🔧 Settings")
llm_choice = st.sidebar.selectbox("Choose LLM", ["groq"])
if st.sidebar.button("🗑️ Clear Long-Term Memory"):
    clear_faiss_store()
    st.sidebar.success("FAISS memory cleared.")

# --- Session State ---
if "memory" not in st.session_state:
    st.session_state.memory = get_short_term_memory()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_faiss_vector_store()

# --- File Upload ---
st.subheader("📄 Upload Document")
uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
if uploaded_file:
    path = save_uploaded_file(uploaded_file)
    st.success(f"Uploaded: {uploaded_file.name}")
    chunks = load_and_split(path)
    st.session_state.vectorstore = build_vector_store(chunks)
    st.info("Vector store updated with new document.")
    if path.endswith(".pdf"):
        st.markdown(embed_pdf_display(path), unsafe_allow_html=True)

# --- Web URL (Optional Web Search) ---
st.subheader("🌐 Scrape URL")
url = st.text_input("Enter URL to fetch text content (optional)")
if url and st.button("🔍 Learn"):
    scraped_text = scrape_url(url)
    st.text_area("Scraped Content (preview)", scraped_text[:1000])
    from langchain.schema import Document
    doc = Document(page_content=scraped_text, metadata={"source": url})
    chunks = [doc]
    st.session_state.vectorstore = build_vector_store(chunks)
    st.success("Learn content embedded into FAISS store.")

# --- Chat Interface ---
st.subheader("💬 Ask a Question")

user_query = st.text_input("Your question")
if user_query and st.session_state.vectorstore:
    llm = get_llm(llm_choice)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=st.session_state.memory,
        verbose=True
    )
    result = qa_chain.run(user_query)
    st.markdown("### 📥 Response")
    st.success(result)

    # Show memory history
    with st.expander("🧠 Memory History"):
        st.markdown(format_chat_history(st.session_state.memory.chat_memory.messages))
elif user_query:
    st.warning("Please upload a document or scrape a URL before asking a question.")
