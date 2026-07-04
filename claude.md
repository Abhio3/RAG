# RAG App

> v2 is **database-first**. The design is in `docs/DATA_MODEL.md`; the Postgres schema
> (`backend/schema.sql`) is the source of truth and is applied idempotently on startup.

## Stack

* Python FastAPI backend | React + TypeScript (Vite) frontend
* **Postgres = system of record** (Supabase local; chats, messages, documents, pages, chunks, citations, research). Raw `psycopg`, no ORM.
* **Qdrant** local vector DB on `localhost:6333` — collection `documents`, hybrid (dense 1024 + sparse). Point id == `chunks.id` (no mapping table).
* **Supabase Storage** holds raw files + rendered page images (Postgres keeps `storage_path` + `sha256` only).
* Embeddings/rerank: **BGE-M3** (dense+sparse, 1024-d) + **BGE-reranker-v2-m3**, local via `FlagEmbedding` (see `backend/embeddings.py`).
* Generation: **vLLM** (OpenAI-compatible, via the `openai` client). Router picks chat vs reasoning model per turn. One vLLM server per model; both default to `VLLM_BASE_URL`.
* Package managers: `pip` (backend) · `pnpm` (frontend)

## Commands

```
backend:  uvicorn main:app --reload --port 3000   # http://localhost:3000
frontend: pnpm dev                                # http://localhost:3001
infra:    docker compose up -d               # qdrant
supabase: supabase start                     # postgres :54322, storage/api :54321
```

Copy `backend/.env.example` → `backend/.env` and fill in Supabase keys + model names.

## Paths

* Backend entry: `backend/main.py` · data access: `backend/db.py`
* Embeddings/chunking/rerank: `backend/embeddings.py` · Storage: `backend/storage.py`
* Schema: `backend/schema.sql` · Design doc: `docs/DATA_MODEL.md`
* Frontend entry: `frontend/src/App.tsx` · Components: `frontend/src/components/`

## Rules

* No cloud API calls — all models run locally (vLLM for generation, FlagEmbedding for embed/rerank)
* No auth yet, but schema is auth-ready: every owned row has `owner_id` → seeded System user (`00000000-…-0001`)
* No ORM (raw `psycopg`)
* CORS: allow `localhost` and `127.0.0.1` (Vite dev server, any port)
* Qdrant collection name: `documents` · dense size 1024 · cosine · sparse named `sparse`
* Chunking: token-based 512 / 64 overlap (BGE-M3 tokenizer)
* Schema in `backend/schema.sql`, applied on startup, idempotent (IF NOT EXISTS)

## Do not edit

* `qdrant_storage/` (generated)
* `pg_data/` (generated — legacy Postgres data)
* `frontend/dist/` (build output)
