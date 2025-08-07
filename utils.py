# utils.py

import os
import base64

def ensure_dir(path):
    """Ensure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def save_uploaded_file(uploaded_file, save_dir="data/uploaded_files"):
    """Save uploaded file to local path."""
    ensure_dir(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def format_chat_history(chat_history):
    """Format LangChain chat history for display."""
    return "\n\n".join(
        [f"**{msg.type.capitalize()}**: {msg.content}" for msg in chat_history]
    )

def embed_pdf_display(file_path):
    """Render PDF directly in Streamlit (optional)."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display
