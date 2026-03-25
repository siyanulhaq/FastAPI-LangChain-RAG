# ⚡ High-Performance RAG Knowledge System

A professional, full-stack Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain**, and **ChromaDB**. This system allows users to upload PDF/TXT documents and query them using the Llama 3.1 model via Groq.

## 🚀 Features
- **Modern UI:** Sleek, dark-mode dashboard with Admin and User roles.
- **Dynamic Document Management:** Add, edit, or delete context lines directly from the UI.
- **Multi-File Support:** Upload PDFs and Text files to expand the AI's knowledge base.
- **Locked-In Reliability:** Optimized for Windows with an in-memory vector store to prevent file-locking issues.
- **Fast Inference:** Powered by Groq's Llama 3.1-8b-instant model.

## 🛠️ Tech Stack
- **Frontend:** Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API).
- **Backend:** FastAPI (Python), Uvicorn.
- **AI Core:** LangChain, HuggingFace (Embeddings), Groq (LLM).
- **Vector Store:** ChromaDB.

## 📦 Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install fastapi uvicorn langchain langchain-groq langchain-huggingface pypdf chromadb python-multipart
   ```

2. **Set Environment Variables:**
   Make sure to set your `GROQ_API_KEY` in `rag_backend.py` or as an environment variable.

3. **Run the System:**
   Double-click `run.bat` or run:
   ```bash
   python rag_backend.py
   ```

4. **Access the UI:**
   Open `rag_frontend.html` in your browser.

## 🛡️ Role-Based Access
- **Admin:** Full control over documents, uploads, and AI queries.
- **User:** Read-only access to query the knowledge base.

---
*Created as a high-impact portfolio project.*
