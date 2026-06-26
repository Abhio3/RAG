"""RAG App v2 API — database-first (see docs/DATA_MODEL.md).

Postgres is the system of record; Qdrant holds chunk vectors (point id == chunks.id);
raw files live in Supabase Storage. Retrieval is BGE-M3 hybrid (dense + sparse, fused
in Qdrant) followed by a BGE reranker before the context reaches the LLM.
"""
import hashlib
import os
import re
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import ollama
import pandas as pd
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PyPDF2 import PdfReader
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

load_dotenv()  # read backend/.env (Supabase keys, model overrides) before our modules

import db          # noqa: E402  (after load_dotenv so env vars are available)
import embeddings  # noqa: E402
import storage     # noqa: E402

# --- Config ----------------------------------------------------------------
COLLECTION = "documents"
VECTOR_SIZE = embeddings.EMBED_DIM            # 1024 (BGE-M3 dense)
EMBED_MODEL = embeddings.EMBED_MODEL          # 'BAAI/bge-m3'
RERANK_MODEL = embeddings.RERANK_MODEL        # 'BAAI/bge-reranker-v2-m3'
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3.5:35b")
REASONING_MODEL = os.environ.get("REASONING_MODEL", "deepseek-r1:70b")
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"
PREFETCH_LIMIT = int(os.environ.get("PREFETCH_LIMIT", "30"))  # fused candidates from Qdrant
TOP_K = int(os.environ.get("TOP_K", "6"))                     # passages sent to the LLM

qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


# --- Startup ---------------------------------------------------------------

def ensure_collection() -> None:
    """Create (or recreate) the hybrid Qdrant collection: dense(1024) + sparse.

    A v1 collection (768-d, unnamed vector) is incompatible — the dimension change
    forces a re-index (§9), so we drop and recreate it. Existing v2 collections are
    left untouched.
    """
    vectors = {"dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)}
    sparse = {"sparse": models.SparseVectorParams()}

    if qdrant.collection_exists(COLLECTION):
        info = qdrant.get_collection(COLLECTION)
        cfg = info.config.params
        compatible = (
            isinstance(cfg.vectors, dict)
            and "dense" in cfg.vectors
            and cfg.vectors["dense"].size == VECTOR_SIZE
            and bool(cfg.sparse_vectors)
            and "sparse" in cfg.sparse_vectors
        )
        if compatible:
            return
        qdrant.delete_collection(COLLECTION)

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=vectors,
        sparse_vectors_config=sparse,
    )


def seed_models() -> None:
    """Register the active models so message/chunk FKs resolve (§4, §7)."""
    db.upsert_model(EMBED_MODEL, "embedding", provider="flagembedding", dimensions=VECTOR_SIZE)
    db.upsert_model(RERANK_MODEL, "reranker", provider="flagembedding")
    db.upsert_model(CHAT_MODEL, "chat", provider="ollama")
    db.upsert_model(REASONING_MODEL, "reasoning", provider="ollama")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_schema()
    seed_models()
    ensure_collection()
    if storage.is_configured():
        storage.ensure_bucket()
    yield


app = FastAPI(title="RAG App v2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Allow the Vite dev server on both localhost and 127.0.0.1 (any port).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Parsing ---------------------------------------------------------------

def extract_pages(filename: str, data: bytes) -> list[dict]:
    """Parse a file into [{page_no, text}, ...]. One logical 'page' for non-PDFs."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        pages = [
            {"page_no": i + 1, "text": (page.extract_text() or "").replace("\x00", "")}
            for i, page in enumerate(reader.pages)
        ]
    elif name.endswith(".txt") or name.endswith(".md"):
        text = data.decode("utf-8", errors="ignore")
        pages = [{"page_no": 1, "text": text.replace("\x00", "")}]
    elif name.endswith(".csv"):
        text = pd.read_csv(BytesIO(data)).to_markdown(index=False)
        pages = [{"page_no": 1, "text": text}]
    elif name.endswith(".xlsx"):
        text = pd.read_excel(BytesIO(data)).to_markdown(index=False)
        pages = [{"page_no": 1, "text": text}]
    elif name.endswith(".docx"):
        doc = Document(BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs).replace("\x00", "")
        pages = [{"page_no": 1, "text": text}]
    else:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, MD, CSV, XLSX, and DOCX files are supported.",
        )
    return pages


# --- Retrieval -------------------------------------------------------------

def _chat_filter(chat_id: Optional[str]) -> Optional[models.Filter]:
    if not chat_id:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))]
    )


def hybrid_search(query: str, chat_id: Optional[str], top_k: int = TOP_K) -> list[dict]:
    """Dense+sparse hybrid retrieval (RRF fusion in Qdrant) + optional rerank → top_k."""
    qv = embeddings.embed_query(query)
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=qv["dense"], using="dense", limit=PREFETCH_LIMIT),
            models.Prefetch(
                query=models.SparseVector(indices=qv["sparse"]["indices"], values=qv["sparse"]["values"]),
                using="sparse",
                limit=PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=_chat_filter(chat_id),
        limit=PREFETCH_LIMIT,
        with_payload=True,
    ).points

    results = [
        {
            "chunk_id": str(h.id),
            "score": float(h.score),
            "text": h.payload.get("text", ""),
            "document_id": h.payload.get("document_id"),
            "title": h.payload.get("title", ""),
            "page_no": h.payload.get("page_no"),
        }
        for h in hits
    ]

    if USE_RERANKER and results:
        scores = embeddings.rerank(query, [r["text"] for r in results])
        for r, s in zip(results, scores):
            r["score"] = s
        results.sort(key=lambda r: r["score"], reverse=True)

    return results[:top_k]


# --- Endpoints -------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    tags: str = Form(""),
    chat_id: Optional[str] = Form(None),
) -> dict:
    if not storage.is_configured():
        raise HTTPException(status_code=503, detail="Supabase Storage is not configured.")

    data = await file.read()
    sha256 = hashlib.sha256(data).hexdigest()

    # Dedup on (owner, content hash).
    existing = db.find_document_by_hash(sha256)
    if existing:
        return {
            "id": str(existing["id"]),
            "chat_id": str(existing["chat_id"]) if existing["chat_id"] else None,
            "filename": existing["title"],
            "status": existing["status"],
            "deduped": True,
        }

    pages = extract_pages(file.filename, data)
    if not any((p["text"] or "").strip() for p in pages):
        raise HTTPException(status_code=400, detail="No extractable text found.")

    if not chat_id:
        chat_id = str(db.create_chat(title=f"Chat about {file.filename}")["id"])

    mime = file.content_type or "application/octet-stream"
    doc = db.insert_document(
        title=file.filename,
        chat_id=chat_id,
        source_type="upload",
        original_name=file.filename,
        mime_type=mime,
        file_size=len(data),
        sha256=sha256,
        status="parsing",
    )
    doc_id = str(doc["id"])
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        # 1. raw file → Supabase Storage; record the path.
        storage_path = f"{db.SYSTEM_USER_ID}/{doc_id}/{file.filename}"
        storage.upload(storage_path, data, mime)
        db.update_document(doc_id, storage_path=storage_path, page_count=len(pages))

        # 2. persist per-page parsed text.
        db.set_document_status(doc_id, "chunking")
        db.insert_pages(doc_id, pages)

        # 3. token-chunk each page → chunk rows (their ids become Qdrant point ids).
        chunk_specs: list[dict] = []
        for page in pages:
            for idx, piece in enumerate(embeddings.chunk_text(page["text"])):
                chunk_specs.append(
                    {
                        "chunk_index": len(chunk_specs),
                        "page_no": page["page_no"],
                        "content": piece,
                        "content_tokens": embeddings.count_tokens(piece),
                        "source_kind": "text",
                    }
                )
        if not chunk_specs:
            raise HTTPException(status_code=400, detail="No extractable text found.")
        chunk_rows = db.insert_chunks(
            doc_id, chunk_specs, chat_id=chat_id, embedding_model=EMBED_MODEL
        )

        # 4. embed (dense+sparse) and upsert to Qdrant under each chunk's id.
        db.set_document_status(doc_id, "embedding")
        vectors = embeddings.embed_texts([c["content"] for c in chunk_rows])
        points = [
            models.PointStruct(
                id=str(row["id"]),
                vector={
                    "dense": vec["dense"],
                    "sparse": models.SparseVector(
                        indices=vec["sparse"]["indices"], values=vec["sparse"]["values"]
                    ),
                },
                payload={
                    "text": row["content"],
                    "owner_id": db.SYSTEM_USER_ID,
                    "chat_id": chat_id,
                    "document_id": doc_id,
                    "title": file.filename,
                    "page_no": row["page_no"],
                    "source_type": "upload",
                    "source_kind": "text",
                    "tags": tag_list,
                },
            )
            for row, vec in zip(chunk_rows, vectors)
        ]
        qdrant.upsert(collection_name=COLLECTION, points=points)

        # 5. tags + ready.
        db.attach_tags(doc_id, tag_list)
        db.mark_ready(doc_id)
    except HTTPException:
        db.set_document_status(doc_id, "failed", "no extractable text")
        raise
    except Exception as exc:  # noqa: BLE001 — surface the failure on the document row
        db.set_document_status(doc_id, "failed", str(exc)[:500])
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return {
        "id": doc_id,
        "chat_id": chat_id,
        "filename": file.filename,
        "chunks": len(chunk_rows),
        "tags": tag_list,
        "status": "ready",
    }


@app.get("/documents")
def documents(chat_id: Optional[str] = None) -> list[dict]:
    return db.list_documents(chat_id)


@app.get("/search")
def search(query: str, chat_id: Optional[str] = None, limit: int = TOP_K) -> list[dict]:
    query_str = query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Query string is required.")
    return hybrid_search(query_str, chat_id, top_k=limit)


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


# Heuristics that route a turn to the reasoning model (§7).
_REASONING_RE = re.compile(
    r"\b(prove|derive|calculate|compute|solve|step[- ]by[- ]step|reason|why does|"
    r"how many|equation|integral|algorithm|complexity|optimi[sz]e)\b",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def route_model(question: str) -> str:
    """Pick reasoning vs chat model for this turn; recorded in messages.model_id."""
    return REASONING_MODEL if _REASONING_RE.search(question) else CHAT_MODEL


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    chat_id = req.chat_id
    if not chat_id:
        chat_id = str(db.create_chat(title=question[:60])["id"])
    db.add_message(chat_id, "user", question)
    db.touch_chat(chat_id)

    hits = hybrid_search(question, chat_id)
    context = "\n\n".join(h["text"] for h in hits)
    model_id = route_model(question)
    prompt = (
        "Answer the question using only the context below. If the context is "
        "insufficient, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    def generate():
        answer = ""
        usage = {}
        for part in ollama.generate(model=model_id, prompt=prompt, stream=True):
            token = part.get("response", "")
            answer += token
            yield token
            if part.get("done"):
                usage = {
                    "prompt": part.get("prompt_eval_count"),
                    "completion": part.get("eval_count"),
                    "total": (part.get("prompt_eval_count") or 0) + (part.get("eval_count") or 0),
                }

        # Split out the <think> reasoning trace so it can be shown collapsibly.
        think = _THINK_RE.findall(answer)
        reasoning = "\n".join(t.strip() for t in think) if think else None
        content = _THINK_RE.sub("", answer).strip()

        msg = db.add_message(
            chat_id, "assistant", content or answer,
            reasoning_content=reasoning, model_id=model_id, token_usage=usage or None,
        )
        db.add_citations(
            str(msg["id"]),
            [
                {
                    "chunk_id": h["chunk_id"],
                    "document_id": h["document_id"],
                    "quote": h["text"][:500],
                    "rank": i + 1,
                    "score": h["score"],
                }
                for i, h in enumerate(hits)
            ],
        )

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X-Chat-Id": chat_id, "Access-Control-Expose-Headers": "X-Chat-Id"},
    )
