# FastAPI LangChain RAG System

Production-style Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, ChromaDB, and Groq LLMs.

## Highlights
- Document ingestion for PDF and TXT files
- Embedding + retrieval pipeline with ChromaDB
- FastAPI backend with clean API endpoints
- Browser-based UI for querying uploaded knowledge
- Groq-powered low-latency responses

## Tech Stack
- Python, FastAPI, Uvicorn
- LangChain, ChromaDB
- Groq, HuggingFace embeddings
- HTML, CSS, JavaScript

## Project Structure
```text
.
|-- rag_backend.py
|-- rag_frontend.html
|-- run.bat
|-- requirements.txt
`-- document.txt
```

## Quick Start
```bash
pip install -r requirements.txt
python rag_backend.py
```

Then open `rag_frontend.html` in your browser.

## Use Cases
- Internal knowledge assistant
- Domain-specific Q&A over private docs
- Fast portfolio-ready RAG demonstration

## Author
Siyan Ul Haq  
GitHub: https://github.com/siyanulhaq