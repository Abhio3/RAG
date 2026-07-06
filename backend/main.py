"""RAG App v2 API — database-first (see docs/DATA_MODEL.md).

Postgres is the system of record; Qdrant holds chunk vectors (point id == chunks.id);
raw files live in Supabase Storage. Retrieval is BGE-M3 hybrid (dense + sparse, fused
in Qdrant) followed by a BGE reranker before the context reaches the LLM.
"""
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO
from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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
import web         # noqa: E402
from think_stream import ThinkStream  # noqa: E402

# --- Config ----------------------------------------------------------------
COLLECTION = "documents"
VECTOR_SIZE = embeddings.EMBED_DIM            # 1024 (BGE-M3 dense)
EMBED_MODEL = embeddings.EMBED_MODEL          # 'BAAI/bge-m3'
RERANK_MODEL = embeddings.RERANK_MODEL        # 'BAAI/bge-reranker-v2-m3'
CHAT_MODEL = os.environ.get("CHAT_MODEL", "Qwen/Qwen3-14B")
# No separate reasoning model by default — Qwen3 has native thinking mode (the <think>
# trace is split out live by think_stream.ThinkStream). Set REASONING_MODEL in .env only
# if you actually run a second vLLM server.
REASONING_MODEL = os.environ.get("REASONING_MODEL", CHAT_MODEL)
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() == "true"
PREFETCH_LIMIT = int(os.environ.get("PREFETCH_LIMIT", "30"))  # fused candidates from Qdrant
TOP_K = int(os.environ.get("TOP_K", "6"))                     # passages sent to the LLM

# --- LLM serving (vLLM, OpenAI-compatible) ---------------------------------
# vLLM serves one model per process, so chat and reasoning models may live behind
# different base URLs (two vLLM servers). Both default to a single VLLM_BASE_URL;
# override per role only if you run them separately. The model name must match the
# server's --served-model-name. See docs/DATA_MODEL.md §1, §7.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")  # vLLM ignores it unless --api-key is set
CHAT_BASE_URL = os.environ.get("CHAT_BASE_URL", VLLM_BASE_URL)
REASONING_BASE_URL = os.environ.get("REASONING_BASE_URL", VLLM_BASE_URL)
_MODEL_BASE_URL = {CHAT_MODEL: CHAT_BASE_URL, REASONING_MODEL: REASONING_BASE_URL}


@lru_cache(maxsize=8)
def _llm(base_url: str, model_id: str) -> ChatOpenAI:
    """LangChain chat model for a vLLM endpoint (cached per base_url+model)."""
    return ChatOpenAI(
        base_url=base_url, api_key=VLLM_API_KEY, model=model_id,
        streaming=True, stream_usage=True,
    )


def _complete(prompt: str, model_id: str, base_url: str) -> str:
    """One-shot (non-streamed) completion — used for research planning."""
    return _llm(base_url, model_id).invoke(prompt).content or ""


def _event(**kw) -> str:
    """One NDJSON line for the token/reasoning/progress stream."""
    return json.dumps(kw) + "\n"


# Chat + research prompts as LCEL templates; each composes into `prompt | _llm(...)`.
_CHAT_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. If the context is "
    "insufficient, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
)
_RESEARCH_PROMPT = ChatPromptTemplate.from_template(
    "Answer the research question using only the sources below. Cite concrete facts. "
    "If the sources are insufficient, say so.\n\nSources:\n{context}\n\nQuestion: {question}"
)


def _stream_answer(chain, inputs: dict):
    """Yield ('answer'|'reasoning'|'usage', value) from an LCEL chain's token stream.

    Prefers the model's separate reasoning channel (additional_kwargs) if a parser is ever
    enabled; otherwise the <think> splitter routes inline thinking to the reasoning channel.
    """
    splitter = ThinkStream()
    for chunk in chain.stream(inputs):
        if (rc := (chunk.additional_kwargs or {}).get("reasoning_content")):
            yield ("reasoning", rc)
        if (um := getattr(chunk, "usage_metadata", None)):
            yield ("usage", {"prompt": um.get("input_tokens"),
                             "completion": um.get("output_tokens"),
                             "total": um.get("total_tokens")})
        if (text := chunk.content or ""):
            yield from splitter.feed(text)
    yield from splitter.flush()


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
    db.upsert_model(CHAT_MODEL, "chat", provider="vllm")
    db.upsert_model(REASONING_MODEL, "reasoning", provider="vllm")


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


# --- Indexing --------------------------------------------------------------

def index_pages(doc_id: str, chat_id: Optional[str], title: str, pages: list[dict],
                *, source_type: str = "upload", tag_list: Optional[list[str]] = None) -> int:
    """Chunk → embed (dense+sparse) → Qdrant upsert for a doc's pages; flip it ready.

    Shared by /upload and the research crawler (which ingests web pages as documents).
    Returns the number of chunks indexed (0 if the pages had no extractable text).
    """
    tag_list = tag_list or []
    db.set_document_status(doc_id, "chunking")
    db.insert_pages(doc_id, pages)

    chunk_specs: list[dict] = []
    for page in pages:
        for piece in embeddings.chunk_text(page.get("text") or ""):
            chunk_specs.append({
                "chunk_index": len(chunk_specs),
                "page_no": page["page_no"],
                "content": piece,
                "content_tokens": embeddings.count_tokens(piece),
                "source_kind": "text",
            })
    if not chunk_specs:
        return 0

    chunk_rows = db.insert_chunks(doc_id, chunk_specs, chat_id=chat_id, embedding_model=EMBED_MODEL)
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
                "title": title,
                "page_no": row["page_no"],
                "source_type": source_type,
                "source_kind": "text",
                "tags": tag_list,
            },
        )
        for row, vec in zip(chunk_rows, vectors)
    ]
    # ponytail: upload_points batches internally (default 64/req) so large docs don't
    # blow past Qdrant's request-size limit the way one big upsert() call would.
    qdrant.upload_points(collection_name=COLLECTION, points=points, wait=True)
    db.attach_tags(doc_id, tag_list)
    db.mark_ready(doc_id)
    return len(chunk_rows)


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
        # raw file → Supabase Storage; record the path, then index its pages.
        storage_path = f"{db.SYSTEM_USER_ID}/{doc_id}/{file.filename}"
        storage.upload(storage_path, data, mime)
        db.update_document(doc_id, storage_path=storage_path, page_count=len(pages))
        n_chunks = index_pages(doc_id, chat_id, file.filename, pages,
                               source_type="upload", tag_list=tag_list)
        if not n_chunks:
            raise HTTPException(status_code=400, detail="No extractable text found.")
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
        "chunks": n_chunks,
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
    mode: str = "chat"  # 'chat' | 'research' | 'deep_research'


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


def route_model(question: str) -> str:
    """Pick reasoning vs chat model for this turn; recorded in messages.model_id."""
    return REASONING_MODEL if _REASONING_RE.search(question) else CHAT_MODEL


def research_stream(question: str, chat_id: str, mode: str):
    """Search → crawl → ingest → synthesize, streaming NDJSON progress then the answer.

    The reasoning model decomposes the task into sub-queries; each is searched on the
    local SearXNG, the top hits are fetched to markdown (trafilatura) and ingested as
    web documents through the normal pipeline, then the answer is synthesized over the
    freshly-indexed chunks. deep_research just widens the fan-out (§6, §7).
    """
    model_id, base_url = REASONING_MODEL, REASONING_BASE_URL
    run_id = str(db.create_research_run(chat_id, question, mode, model_id=model_id)["id"])
    n_subq = 4 if mode == "deep_research" else 2
    per_q = 4 if mode == "deep_research" else 3
    step = 0

    yield _event(type="status", text="Planning sub-questions…")
    plan = _complete(
        f"Break this research task into {n_subq} focused web-search queries. "
        f"Output one query per line, no numbering, no extra text.\n\nTask: {question}",
        model_id, base_url,
    )
    # ponytail: line-split parse; the prompt pins the format. Tighten only if a model ignores it.
    subqs = [ln.strip(" -*\t") for ln in plan.splitlines() if ln.strip()][:n_subq] or [question]
    db.add_research_step(run_id, step, "reason", input=question, output="\n".join(subqs)); step += 1
    yield _event(type="plan", questions=subqs)

    for sq in subqs:
        yield _event(type="step", kind="search", text=f"Searching: {sq}")
        try:
            results = web.search(sq, k=per_q)
        except Exception as exc:  # noqa: BLE001
            yield _event(type="step", kind="search", text=f"Search failed: {exc}")
            continue
        db.add_research_step(run_id, step, "search", input=sq,
                             output="\n".join(r["url"] for r in results)); step += 1
        for res in results:
            url = res["url"]
            yield _event(type="step", kind="crawl", text=f"Reading {url}")
            try:
                md = web.fetch_markdown(url)
            except Exception as exc:  # noqa: BLE001
                yield _event(type="step", kind="crawl", text=f"Fetch failed: {exc}")
                continue
            if not md.strip():
                continue
            sha = hashlib.sha256(md.encode()).hexdigest()
            existing = db.find_document_by_hash(sha)
            if existing:
                doc_id = str(existing["id"])
            else:
                doc = db.insert_document(
                    title=res["title"], chat_id=chat_id, source_type="web", source_url=url,
                    mime_type="text/markdown", file_size=len(md.encode()), sha256=sha, status="parsing",
                )
                doc_id = str(doc["id"])
                try:
                    index_pages(doc_id, chat_id, res["title"],
                                [{"page_no": 1, "text": md}], source_type="web")
                except Exception as exc:  # noqa: BLE001
                    db.set_document_status(doc_id, "failed", str(exc)[:500])
                    continue
            db.insert_web_source(run_id, url, document_id=doc_id, title=res["title"], content_hash=sha)
            db.add_research_step(run_id, step, "crawl", input=url, output=res["title"]); step += 1

    yield _event(type="status", text="Synthesizing answer…")
    hits = hybrid_search(question, chat_id, top_k=8)
    context = "\n\n".join(h["text"] for h in hits)

    answer, reasoning, usage = "", "", {}
    try:
        chain = _RESEARCH_PROMPT | _llm(base_url, model_id)
        for kind, val in _stream_answer(chain, {"context": context, "question": question}):
            if kind == "usage":
                usage = val
            elif kind == "reasoning":
                reasoning += val
                yield _event(type="reasoning", text=val)
            else:
                answer += val
                yield _event(type="token", text=val)
    except Exception as exc:  # noqa: BLE001 — surface mid-stream failures to the client
        yield _event(type="error", text=f"Synthesis failed: {exc}")
        db.finish_research_run(run_id, "failed")
        return

    msg = db.add_message(chat_id, "assistant", answer.strip() or "(no answer)",
                         reasoning_content=reasoning.strip() or None,
                         model_id=model_id, token_usage=usage or None)
    db.add_citations(str(msg["id"]), [
        {"chunk_id": h["chunk_id"], "document_id": h["document_id"],
         "quote": h["text"][:500], "rank": i + 1, "score": h["score"]}
        for i, h in enumerate(hits)
    ])
    db.finish_research_run(run_id, "done", message_id=str(msg["id"]), token_usage=usage or None)
    yield _event(type="done")


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    chat_id = req.chat_id
    if not chat_id:
        chat_id = str(db.create_chat(title=question[:60], mode=req.mode)["id"])
    db.add_message(chat_id, "user", question)
    db.touch_chat(chat_id)

    headers = {"X-Chat-Id": chat_id, "Access-Control-Expose-Headers": "X-Chat-Id"}
    if req.mode in ("research", "deep_research"):
        return StreamingResponse(
            research_stream(question, chat_id, req.mode),
            media_type="application/x-ndjson", headers=headers,
        )

    hits = hybrid_search(question, chat_id)
    context = "\n\n".join(h["text"] for h in hits)
    model_id = route_model(question)
    base_url = _MODEL_BASE_URL.get(model_id, VLLM_BASE_URL)

    def generate():
        answer, reasoning, usage = "", "", {}
        try:
            chain = _CHAT_PROMPT | _llm(base_url, model_id)
            for kind, val in _stream_answer(chain, {"context": context, "question": question}):
                if kind == "usage":
                    usage = val
                elif kind == "reasoning":
                    reasoning += val
                    yield _event(type="reasoning", text=val)
                else:
                    answer += val
                    yield _event(type="token", text=val)
        except Exception as exc:  # noqa: BLE001 — surface mid-stream failures to the client
            yield _event(type="error", text=f"Generation failed: {exc}")
            db.add_message(chat_id, "assistant", answer.strip() or f"[error] {exc}",
                           reasoning_content=reasoning.strip() or None, model_id=model_id)
            return

        msg = db.add_message(
            chat_id, "assistant", answer.strip() or "(no answer)",
            reasoning_content=reasoning.strip() or None, model_id=model_id, token_usage=usage or None,
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
        yield _event(type="done")

    return StreamingResponse(generate(), media_type="application/x-ndjson", headers=headers)
