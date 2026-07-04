# RAG App v2

A fully local Retrieval-Augmented Generation app. Upload documents (PDF/TXT/MD/CSV/XLSX/DOCX),
then chat with them. Everything runs on your machine — no cloud APIs.

Design lives in [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md); the Postgres schema
(`backend/schema.sql`) is the source of truth and is applied idempotently on startup.

## Stack

- **Generation**: vLLM (OpenAI-compatible) — chat + reasoning models, router picks per turn
- **Embedding / rerank**: BGE-M3 (dense 1024 + sparse) + BGE-reranker-v2-m3, local via `FlagEmbedding`
- **Vector DB**: Qdrant (local, Docker) — collection `documents`, hybrid; point id == `chunks.id`
- **System of record**: Supabase local (Postgres + Storage + Realtime + Studio)
- **Backend**: FastAPI · **Frontend**: React + TypeScript (Vite) + Tailwind

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (for Qdrant)
- [Supabase CLI](https://supabase.com/docs/guides/cli) (Postgres + Storage)
- [vLLM](https://docs.vllm.ai) serving your chat + reasoning models
- Python 3.10+ · Node 18+ and `pnpm`

## Start everything

**1. Infra — Qdrant + Supabase**

```bash
docker compose up -d        # Qdrant on :6333
supabase start              # Postgres :54322, Storage/API :54321, Studio :54323
```

**2. Models — vLLM** (OpenAI-compatible)

```bash
# One server handles chat + research/reasoning — Qwen3 has native thinking mode
# (backend/main.py splits out the <think> trace itself, no --reasoning-parser needed).
# --gpu-memory-utilization is a FRACTION OF TOTAL GPU MEMORY, not of what's free — check
# `nvidia-smi` / `ss -tlnp` for what else is already running before picking a port/value.
# On this box, chandra-ocr owns :8000/:8001 and reserves a chunk of the GPU, so we use an
# AWQ build (~9GB) on :3004 with a conservative utilization instead of vLLM's ~0.9 default.
vllm serve Qwen/Qwen3-14B-AWQ --port 3004 --gpu-memory-utilization 0.4 --max-model-len 8192
```

Only run a second server (different port, override `REASONING_BASE_URL`) if you want a
dedicated reasoning model distinct from `CHAT_MODEL`.

**Ports used by this app** (pick different ones if something else already holds these):

| Port | Service |
| ---- | ------- |
| 3000 | Backend (FastAPI) |
| 3001 | Frontend (Vite) |
| 3004 | vLLM (Qwen3-14B-AWQ) |
| 6333 | Qdrant |
| 8888 | SearXNG |
| 54321-54323 | Supabase (Storage/API, Postgres, Studio) |

Embeddings and the reranker load locally on first use — no server needed.

**3. Backend** (from `backend/`)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set Supabase keys + VLLM_BASE_URL / model names
uvicorn main:app --reload --port 3000   # http://localhost:3000
```

**4. Frontend** (from `frontend/`)

```bash
cd frontend
pnpm install
pnpm dev                    # http://localhost:3001
```

Open http://localhost:3001, upload a document, and start asking questions.

## API

| Method | Path                   | Description                                                            |
| ------ | ---------------------- | --------------------------------------------------------------------- |
| POST   | `/upload`              | Upload a file (+ optional `tags`, `chat_id`); chunks, embeds, indexes. |
| POST   | `/chat`                | `{ "question", "chat_id"? }` → streamed answer; persists the turn.     |
| GET    | `/search`              | `?query=&chat_id=&limit=` → hybrid + reranked passages.               |
| GET    | `/chats`               | List chat threads (newest first).                                      |
| GET    | `/chats/{id}/messages` | Messages in a thread.                                                  |
| GET    | `/documents`           | List indexed documents with tags + chunk counts.                      |
| GET    | `/health`              | `{ "status": "ok" }`                                                   |

`/chat` returns the thread id in the `X-Chat-Id` response header so the client can keep
appending to the same conversation.

## Notes

- Qdrant collection `documents` is auto-created on startup (dense 1024 cosine + sparse).
- Postgres tables are created from `backend/schema.sql` on every startup (idempotent).
- Chunking: token-based 512 / 64 overlap (BGE-M3 tokenizer).
- `qdrant_storage/`, `pg_data/`, and `supabase/.branches`, `supabase/.temp` are generated — don't edit.
- The bare Postgres + pgAdmin in `docker-compose.yml` are retained under the `legacy`
  profile only (`docker compose --profile legacy up -d`); Supabase is the default.
