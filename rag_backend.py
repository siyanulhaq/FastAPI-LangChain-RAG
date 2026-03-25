# ============================================
# RAG BACKEND - FastAPI Server
# Clean routes, no conflicts
# ============================================

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, shutil, traceback, warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="RAG System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Config ----
from dotenv import load_dotenv
load_dotenv()

DOCUMENT_PATH = "document.txt"
UPLOADS_DIR   = "./uploaded_docs"
CHROMA_DIR    = "./chroma_storage"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---- Globals ----
rag_chain        = None
retriever_global = None
embedding_model  = None

RAG_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the document."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_rag():
    global rag_chain, retriever_global, embedding_model
    all_chunks = []

    # Load document.txt
    if not os.path.exists(DOCUMENT_PATH):
        with open(DOCUMENT_PATH, "w") as f:
            f.write("No content yet.\n")

    try:
        loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        all_chunks.extend(splitter.split_documents(docs))
    except Exception as e:
        print(f"Error loading document.txt: {e}")

    # Load uploaded files
    for fname in os.listdir(UPLOADS_DIR):
        fpath = os.path.join(UPLOADS_DIR, fname)
        try:
            loader = PyPDFLoader(fpath) if fname.endswith(".pdf") else TextLoader(fpath, encoding="utf-8")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            all_chunks.extend(splitter.split_documents(docs))
        except Exception as e:
            print(f"Could not load {fname}: {e}")

    if not all_chunks:
        print("No chunks found.")
        return 0

    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Keep ChromaDB in memory to avoid Windows file-locking bugs
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model
    )

    retriever_global = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    rag_chain = (
        {"context": retriever_global | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )

    n = len(os.listdir(UPLOADS_DIR))
    print(f"RAG loaded: {len(all_chunks)} chunks from {1 + n} source(s).")
    return len(all_chunks)


@app.on_event("startup")
async def startup_event():
    load_rag()


# ---- Models ----
class QuestionRequest(BaseModel):
    question: str

class AddLineRequest(BaseModel):
    text: str

class UpdateLineRequest(BaseModel):
    line_index: int
    new_text: str


# ---- Q&A ----
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        if rag_chain is None:
            raise HTTPException(status_code=503, detail="RAG not ready.")
        answer = rag_chain.invoke(req.question)
        sources = [doc.page_content[:150] for doc in retriever_global.invoke(req.question)] if retriever_global else []
        return {"answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ---- Document Lines ----
@app.get("/document")
async def get_document():
    try:
        if not os.path.exists(DOCUMENT_PATH):
            return {"lines": []}
        with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]
        return {"lines": lines}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document/add")
async def add_line(req: AddLineRequest):
    try:
        with open(DOCUMENT_PATH, "a", encoding="utf-8") as f:
            f.write(req.text.strip() + "\n")
        load_rag()
        return {"message": "Line added and RAG rebuilt."}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/document/update")
async def update_line(req: UpdateLineRequest):
    try:
        with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if req.line_index < 0 or req.line_index >= len(lines):
            raise HTTPException(status_code=400, detail="Invalid line index.")
        lines[req.line_index] = req.new_text.strip() + "\n"
        with open(DOCUMENT_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines)
        load_rag()
        return {"message": "Line updated and RAG rebuilt."}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/delete/{line_index}")
async def delete_line(line_index: int):
    try:
        with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if line_index < 0 or line_index >= len(lines):
            raise HTTPException(status_code=400, detail="Invalid line index.")
        lines.pop(line_index)
        with open(DOCUMENT_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines)
        load_rag()
        return {"message": "Line deleted and RAG rebuilt."}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ---- File Upload ----
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files supported.")
        save_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        load_rag()
        return {"message": f"{file.filename} uploaded.", "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files():
    try:
        files = []
        for fname in os.listdir(UPLOADS_DIR):
            fpath = os.path.join(UPLOADS_DIR, fname)
            files.append({"name": fname, "size": os.path.getsize(fpath)})
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{filename}")
async def get_file_content(filename: str):
    try:
        fpath = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(fpath):
            raise HTTPException(status_code=404, detail="File not found.")
        if filename.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                return {"content": f.read(), "type": "text"}
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(fpath)
            docs = loader.load()
            content = "\n\n--- Page Break ---\n\n".join(d.page_content for d in docs)
            return {"content": content, "type": "pdf"}
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    try:
        fpath = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(fpath):
            raise HTTPException(status_code=404, detail="File not found.")
        os.remove(fpath)
        load_rag()
        return {"message": f"{filename} deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "rag_ready": rag_chain is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_backend:app", host="0.0.0.0", port=8000, reload=True)
