# RAG App

## Stack

* Python FastAPI backend | React + TypeScript (Vite) frontend
* Ollama: `nomic-embed-text` (embed) · `gemma3:4b` (generate)
* Qdrant local vector DB on `localhost:6333` (chunk vectors)
* Postgres on `localhost:5432` (chat history, document metadata + raw files + tags) — raw `psycopg`, no ORM
* Package managers: `pip` (backend) · `pnpm` (frontend)

## Commands

```
backend:  uvicorn main:app --reload          # http://localhost:8000
frontend: pnpm dev                           # http://localhost:5173
infra:    docker compose up -d               # qdrant + postgres
```

## Paths

* Backend entry: `backend/main.py`
* Frontend entry: `frontend/src/App.tsx`
* Components: `frontend/src/components/`

## Rules

* No cloud API calls — Ollama only
* No auth, no ORM (raw `psycopg` against Postgres)
* CORS: allow `localhost` and `127.0.0.1` (Vite dev server, any port)
* Keep LangChain to `langchain-text-splitters` only
* Qdrant collection name: `documents` · vector size: 768 · cosine distance
* Chunk size: 500 chars · overlap: 50
* Postgres: db/user/pass all `rag`; schema in `backend/schema.sql` (applied on startup, idempotent)

## Do not edit

* `qdrant_storage/` (generated)
* `pg_data/` (generated — Postgres data)
* `frontend/dist/` (build output)
