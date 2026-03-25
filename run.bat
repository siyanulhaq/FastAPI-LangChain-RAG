@echo off
echo Starting RAG Backend Server...
call .\venv\Scripts\activate.bat
python rag_backend.py
pause
