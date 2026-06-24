import uuid
from contextlib import asynccontextmanager
from io import BytesIO

import ollama
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams
import pandas as pd
from docx import Document

load_dotenv()  # read backend/.env (Supabase keys)

import db  # noqa: E402  (after load_dotenv so env vars are available)

# --- Config ---
COLLECTION = "documents"
VECTOR_SIZE = 768
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "gemma3:4b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

qdrant = QdrantClient(url="http://localhost:6333")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def ensure_collection() -> None:
    """Create the Qdrant collection on startup if it doesn't exist."""
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    db.init_schema()
    yield


app = FastAPI(title="RAG App", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Allow the Vite dev server on both localhost and 127.0.0.1 (any port).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    elif name.endswith(".csv"):
        df = pd.read_csv(BytesIO(data))
        text = df.to_markdown(index=False)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(data))
        text = df.to_markdown(index=False)
    elif name.endswith(".docx"):
        doc = Document(BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, CSV, XLSX, and DOCX files are supported.")
    return text.replace("\x00", "")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    tags: str = Form(""),
    chat_id: str = Form(None),
) -> dict:
    data = await file.read()
    text = extract_text(file.filename, data)
    chunks = [c for c in splitter.split_text(text) if c.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found.")

    if not chat_id:
        chat_id = str(db.create_chat(title=f"Chat about {file.filename}")["id"])

    # Index chunks for vector search in Qdrant.
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embed(chunk),
            payload={"text": chunk, "filename": file.filename, "chat_id": chat_id},
        )
        for chunk in chunks
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)

    # Persist the file + metadata in Postgres.
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    doc = db.insert_document(
        chat_id=chat_id,
        filename=file.filename,
        content=text,
        tags=tag_list,
        chunk_count=len(chunks),
        content_type=file.content_type or "application/octet-stream",
        file_data=data,
    )
    return {
        "id": str(doc["id"]),
        "chat_id": chat_id,
        "filename": file.filename,
        "chunks": len(chunks),
        "tags": tag_list,
    }


from typing import Optional

@app.get("/documents")
def documents(chat_id: Optional[str] = None) -> list[dict]:
    return db.list_documents(chat_id)


@app.get("/search")
def search(query: str, chat_id: Optional[str] = None, limit: int = TOP_K) -> list[dict]:
    query_str = query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Query string is required.")

    query_filter = None
    if chat_id:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))]
        )

    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=embed(query_str),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points

    return [
        {
            "id": h.id,
            "score": h.score,
            "text": h.payload.get("text", ""),
            "filename": h.payload.get("filename", ""),
            "chat_id": h.payload.get("chat_id", ""),
        }
        for h in hits
    ]



# --- Chat ------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    chat_id: str | None = None


@app.get("/chats")
def chats() -> list[dict]:
    return db.list_chats()


@app.get("/chats/{chat_id}/messages")
def chat_messages(chat_id: str) -> list[dict]:
    return db.list_messages(chat_id)


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    # Ensure a chat thread exists; title new chats from the first question.
    chat_id = req.chat_id
    if not chat_id:
        chat_id = str(db.create_chat(title=question[:60])["id"])
    db.add_message(chat_id, "user", question)

    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))]
        ),
        limit=TOP_K,
        with_payload=True,
    ).points
    context = "\n\n".join(h.payload.get("text", "") for h in hits)
    prompt = f"Answer using only this context:\n{context}\n\nQuestion: {question}"

    def generate():
        answer = ""
        for part in ollama.generate(model=GEN_MODEL, prompt=prompt, stream=True):
            token = part.get("response", "")
            answer += token
            yield token
        db.add_message(chat_id, "assistant", answer)

    # Expose the chat id so the client can keep appending to this thread.
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X-Chat-Id": chat_id, "Access-Control-Expose-Headers": "X-Chat-Id"},
    )
