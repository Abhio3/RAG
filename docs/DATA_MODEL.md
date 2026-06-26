# RAG App v2 — Database-First Plan

> Target host: **NVIDIA DGX Spark** (GB10, 128 GB unified memory).
> This document defines **how data is stored** for the expanded app: multi-user-ready
> internal AI chatbot with documents, OCR/vision ingestion, web crawl, research &
> deep-research. Code comes after this is agreed.

---

## 1. Component map (what stores what)

| Concern | System | Notes |
|---|---|---|
| **System of record** (chats, messages, documents, research, provenance) | **Supabase Postgres** (local) | Replaces the bare Postgres container. Single source of truth. |
| **Chunk vectors** (dense + sparse) | **Qdrant** | Collection `documents`, 1024-dim, hybrid. Dedicated, as decided. |
| **Raw files & rendered page images** | **Supabase Storage** | Replaces `documents.file_data bytea`. Postgres keeps the path + hash only. |
| **Realtime UI updates** (ingest/research progress) | Supabase Realtime | Optional but free with Supabase; powers live "thinking"/progress UI. |
| **Auth** | *deferred* → **Authentik (OIDC)** later | Schema is auth-ready now (see §8). No auth code yet. |

### Models (DGX Spark)

| Role | Model | Why |
|---|---|---|
| Embedding | **BGE-M3** (1024-dim, dense+sparse, 8192-token ctx) | 2026 RAG default, MIT, strong on large/multilingual docs, gives hybrid search for free. |
| Reranker | **BGE-reranker-v2-m3** | Re-scores top-k from Qdrant before it hits the LLM. |
| OCR (page → markdown) | **DeepSeek-OCR2** (or MinerU/Docling for layout) | Reliable "PDF page → Markdown", tables/formulas. |
| Vision (understand figures/images) | **Qwen3-VL-7B** (or Qwen2.5-VL) | Captions figures/charts/screenshots so images become *retrievable text*. |
| General chat | **Qwen3-14B** | Best all-round: chat, code, multilingual, instruction-following. |
| Heavy reasoning / deep-research | **DeepSeek-R1-Distill-Qwen-14B** | Best pure reasoning at 14B. |
| Router | thin classifier/heuristic | Picks chat vs reasoning model per turn (see §7). |
| Web crawl/search | **SearXNG** (search) + **Crawl4AI** (fetch→markdown), Firecrawl optional | Fully local, no API keys. |

> **Serving:** vLLM is recommended on DGX Spark for concurrent multi-model serving;
> Ollama still works (both expose an OpenAI-compatible API, so app code is identical).
> The "no cloud / local-only" rule is preserved either way.

---

## 2. Storage split: Postgres vs Qdrant

- **Postgres owns the truth.** Every chunk has a row in `chunks`. Its UUID **is** the Qdrant point id — no separate mapping table.
- **Qdrant owns the math.** It stores the dense (1024) + sparse vectors and a small payload (filterable fields only). It is rebuildable from Postgres at any time.
- Crawled web pages are ingested **as documents** (`source_type='web'`) so there is **one** retrieval pipeline for files and web. Crawl provenance lives in `web_sources`/`research_runs`.

---

## 3. Entity overview

```
users (auth-ready, seeded "system" user for now)
  └─ chats ──┬─ messages ──┬─ message_citations ─→ chunks / documents / web_sources
             │             └─ (model_id, reasoning trace, token usage)
             ├─ documents ─┬─ document_pages ── document_assets (figures/tables/images)
             │             └─ chunks ──────────→ Qdrant point (1:1 by id)
             └─ research_runs ─┬─ research_steps
                               └─ web_sources ─→ (ingested as documents)
tags ─ document_tags                models (registry: which LLM/embed/ocr is active)
```

---

## 4. Postgres schema (Supabase) — DDL

Extends/replaces `backend/schema.sql`. Still **idempotent, applied on startup**.

```sql
create extension if not exists "pgcrypto";   -- gen_random_uuid()

-- ── Identity (auth deferred; one seeded system user owns everything for now) ──
create table if not exists users (
    id           uuid primary key default gen_random_uuid(),
    external_id  text unique,                 -- Authentik 'sub' (null until auth lands)
    email        text unique,
    display_name text,
    role         text not null default 'member' check (role in ('member','admin','system')),
    created_at   timestamptz not null default now()
);
insert into users (id, display_name, role)
values ('00000000-0000-0000-0000-000000000001','System','system')
on conflict do nothing;

-- ── Model registry (router reads this) ──
create table if not exists models (
    id          text primary key,             -- e.g. 'qwen3-14b'
    kind        text not null check (kind in ('chat','reasoning','embedding','reranker','ocr','vision')),
    provider    text not null default 'vllm', -- 'vllm' | 'ollama'
    context_len integer,
    dimensions  integer,                       -- for embedding models
    active      boolean not null default true
);

-- ── Chats & messages ──
create table if not exists chats (
    id         uuid primary key default gen_random_uuid(),
    owner_id   uuid not null references users(id) default '00000000-0000-0000-0000-000000000001',
    title      text not null default 'New chat',
    mode       text not null default 'chat' check (mode in ('chat','research','deep_research')),
    archived   boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists messages (
    id                uuid primary key default gen_random_uuid(),
    chat_id           uuid not null references chats(id) on delete cascade,
    parent_id         uuid references messages(id) on delete set null,  -- branching/edits (Claude/GPT-style)
    role              text not null check (role in ('user','assistant','system','tool')),
    content           text not null,
    reasoning_content text,                    -- the <think> trace, shown collapsible
    model_id          text references models(id),
    token_usage       jsonb,                   -- {prompt, completion, total}
    created_at        timestamptz not null default now()
);

-- ── Documents (uploads AND crawled web pages) ──
create table if not exists documents (
    id            uuid primary key default gen_random_uuid(),
    owner_id      uuid not null references users(id) default '00000000-0000-0000-0000-000000000001',
    chat_id       uuid references chats(id) on delete cascade,  -- null = library-wide, not chat-scoped
    title         text not null,
    source_type   text not null default 'upload' check (source_type in ('upload','web','research')),
    source_url    text,                        -- set when source_type='web'
    original_name text,
    mime_type     text,
    file_size     bigint,
    storage_path  text,                        -- Supabase Storage key (replaces bytea)
    sha256        text,                         -- dedup
    status        text not null default 'pending'
                  check (status in ('pending','parsing','ocr','chunking','embedding','ready','failed')),
    page_count    integer,
    language      text,
    error         text,
    created_at    timestamptz not null default now(),
    parsed_at     timestamptz,
    unique (owner_id, sha256)
);

-- ── Per-page parsed content (markdown) + rendered image for vision ──
create table if not exists document_pages (
    id          uuid primary key default gen_random_uuid(),
    document_id uuid not null references documents(id) on delete cascade,
    page_no     integer not null,
    text        text,                          -- OCR/markdown text of the page
    image_path  text,                          -- rendered page PNG in Storage (for VLM)
    has_images  boolean not null default false,
    ocr_model   text references models(id),
    layout      jsonb,                         -- blocks/bboxes from the layout parser
    unique (document_id, page_no)
);

-- ── Extracted figures/tables/images, captioned by the vision model ──
create table if not exists document_assets (
    id          uuid primary key default gen_random_uuid(),
    document_id uuid not null references documents(id) on delete cascade,
    page_no     integer,
    asset_type  text not null check (asset_type in ('image','table','figure','formula')),
    storage_path text,                         -- cropped asset in Storage
    caption     text,                          -- VLM description → embedded for retrieval
    bbox        jsonb,
    created_at  timestamptz not null default now()
);

-- ── Chunks: the retrieval unit. chunk.id == Qdrant point id ──
create table if not exists chunks (
    id              uuid primary key default gen_random_uuid(),
    document_id     uuid not null references documents(id) on delete cascade,
    owner_id        uuid not null references users(id) default '00000000-0000-0000-0000-000000000001',
    chat_id         uuid references chats(id) on delete cascade,
    page_no         integer,
    chunk_index     integer not null,
    content         text not null,
    content_tokens  integer,
    source_kind     text not null default 'text' check (source_kind in ('text','asset_caption')),
    embedding_model text references models(id),
    created_at      timestamptz not null default now()
);

-- ── Tags (kept from v1) ──
create table if not exists tags (
    id   uuid primary key default gen_random_uuid(),
    name text unique not null
);
create table if not exists document_tags (
    document_id uuid references documents(id) on delete cascade,
    tag_id      uuid references tags(id) on delete cascade,
    primary key (document_id, tag_id)
);

-- ── Research / deep-research orchestration ──
create table if not exists research_runs (
    id          uuid primary key default gen_random_uuid(),
    chat_id     uuid references chats(id) on delete cascade,
    message_id  uuid references messages(id) on delete set null,  -- the assistant answer it produced
    query       text not null,
    mode        text not null check (mode in ('research','deep_research')),
    plan        jsonb,                         -- decomposed sub-questions
    status      text not null default 'running'
                check (status in ('running','done','failed','cancelled')),
    model_id    text references models(id),
    token_usage jsonb,
    started_at  timestamptz not null default now(),
    finished_at timestamptz
);

create table if not exists research_steps (
    id         uuid primary key default gen_random_uuid(),
    run_id     uuid not null references research_runs(id) on delete cascade,
    step_no    integer not null,
    type       text not null check (type in ('search','crawl','read','reason','synthesize')),
    input      text,
    output     text,                           -- summary of what this step produced
    created_at timestamptz not null default now()
);

-- ── Web sources (crawl provenance; the page body is ingested as a document) ──
create table if not exists web_sources (
    id            uuid primary key default gen_random_uuid(),
    run_id        uuid references research_runs(id) on delete cascade,
    document_id   uuid references documents(id) on delete set null,  -- the ingested copy
    url           text not null,
    canonical_url text,
    title         text,
    fetcher       text check (fetcher in ('searxng','crawl4ai','firecrawl','http')),
    content_hash  text,
    fetched_at    timestamptz not null default now()
);

-- ── Citations: ties an assistant message to its evidence ──
create table if not exists message_citations (
    id            uuid primary key default gen_random_uuid(),
    message_id    uuid not null references messages(id) on delete cascade,
    chunk_id      uuid references chunks(id) on delete set null,
    document_id   uuid references documents(id) on delete set null,
    web_source_id uuid references web_sources(id) on delete set null,
    quote         text,
    rank          integer,
    score         double precision
);

-- ── Indexes ──
create index if not exists messages_chat_idx     on messages(chat_id, created_at);
create index if not exists documents_owner_idx   on documents(owner_id, created_at desc);
create index if not exists documents_chat_idx    on documents(chat_id, created_at desc);
create index if not exists chunks_document_idx    on chunks(document_id);
create index if not exists chunks_owner_idx       on chunks(owner_id);
create index if not exists pages_document_idx     on document_pages(document_id, page_no);
create index if not exists research_chat_idx      on research_runs(chat_id, started_at desc);
create index if not exists citations_message_idx  on message_citations(message_id);
```

---

## 5. Qdrant collection

```
Collection: documents
  vectors:
    dense:  size 1024, distance Cosine        # BGE-M3 dense
  sparse_vectors:
    sparse: {}                                 # BGE-M3 sparse (lexical) → hybrid search
  point id: = chunks.id (UUID)                 # no mapping table
  payload (filterable only):
    owner_id, chat_id, document_id, page_no,
    source_type ('upload'|'web'), source_kind ('text'|'asset_caption'),
    language, tags[]
```

- **Hybrid query:** dense + sparse, fused (RRF) in Qdrant, then **BGE-reranker-v2-m3** on the top ~30 → top ~6 to the LLM.
- **Filters** map straight to payload (e.g. `chat_id = X` for chat-scoped, or `owner_id = me` once auth lands).
- **Settings to confirm:** chunk size. BGE-M3's 8192-token window lets us go bigger than v1's 500/50 — propose **~512 tokens, 64 overlap** (token-based, not char-based). Flag if you want to keep 500/50.

---

## 6. Ingestion pipeline (with OCR/vision)

```
upload/crawl → Storage(raw) → documents(status=pending)
  → detect type
  → [PDF/scan/image] layout parse (MinerU/Docling) + OCR (DeepSeek-OCR2)  → document_pages.text
       └─ per figure/table/image → crop to Storage → Qwen3-VL caption     → document_assets.caption
  → [text/docx/md] direct extract                                          → document_pages.text
  → chunk pages (token-based) + chunk every asset caption                  → chunks
  → embed each chunk with BGE-M3 (dense+sparse)                            → Qdrant upsert (id = chunk.id)
  → documents.status = ready   (status streamed to UI via Realtime)
```

Key idea: **images become retrievable** because the vision model's caption is stored as an
`asset_caption` chunk and embedded alongside normal text — so "what does Figure 3 show?" works.

---

## 7. Chat / research data flow

- **Plain chat:** user msg → router picks model → retrieve (hybrid+rerank) over the chat's/owner's chunks → answer → store `messages` (+ `reasoning_content`, `model_id`, `token_usage`) + `message_citations`.
- **Router:** classify the turn (math/logic/multi-step → `DeepSeek-R1-Distill`; else `Qwen3-14B`). Record the choice in `messages.model_id`. Both models stay resident in the 128 GB.
- **Research / deep-research:** create `research_run` → R1 model writes a `plan` (sub-questions) → loop of `research_steps` (SearXNG search → Crawl4AI fetch → ingest page as `document` → retrieve → reason) → synthesize final answer as a `message` with `message_citations` pointing at `web_sources`. Deep-research = more sub-questions, more crawl depth, more synthesis passes.

---

## 8. Auth later (Authentik) — why the schema is ready now

- Every owned row already has `owner_id → users.id`; today it points at the seeded **System** user.
- When Authentik (OIDC) lands: FastAPI validates the bearer token, upserts the user by `external_id` (the OIDC `sub`), and sets `owner_id` to the real user. **No schema migration needed.**
- Then enable Postgres **RLS** with policies like `owner_id = current_user_id()` on `chats/documents/chunks/...`, and add the same `owner_id` filter to every Qdrant query. Both layers enforced.

---

## 9. Migration from v1

| v1 | v2 |
|---|---|
| `documents.file_data bytea` | `documents.storage_path` → Supabase Storage (one-time copy out) |
| `documents.content text` | `document_pages.text` (per page) |
| `documents.chunk_count` | derived from `chunks` |
| inline chat-scoped only | `owner_id` added everywhere (defaults to System user) |
| Qdrant `documents` 768-dim | **re-embed** to 1024-dim BGE-M3 (dimension change forces re-index) |
| Postgres container | Supabase local (Postgres + Storage + Realtime + Studio) |

Re-embedding is unavoidable because the vector dimension changes (768 → 1024). Everything
else migrates with data copies, not rewrites.

---

## 10. Open items to confirm before coding

1. **Supabase local** = full stack (Studio/Storage/Realtime/PostgREST) or just Postgres + Storage? (DGX Spark has the headroom for full.)
2. **Chunking**: keep v1's 500/50 chars, or move to token-based 512/64 to exploit BGE-M3?
3. **Serving**: vLLM (recommended for concurrency) or stay on Ollama?
4. **OCR depth**: every PDF through OCR/vision, or only when a page has low extractable text / detected images? (Latter is faster.)
5. **Workspaces/teams** above `users` for the "internal" multi-team case — needed, or flat users for now?
```
```

Sources used for model/infra choices:
- DGX Spark: https://www.nvidia.com/en-us/products/workstations/dgx-spark/ , https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- Embeddings: https://milvus.io/blog/choose-embedding-model-rag-2026.md , https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models
- Reasoning 14B: https://ollama.com/library/deepseek-r1:14b , https://arxiv.org/pdf/2505.09388
- OCR/vision: https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Document-screening , https://unstract.com/blog/best-opensource-ocr-tools/
- Crawl/search: https://github.com/unclecode/crawl4ai , https://www.firecrawl.dev/blog/best-open-source-web-crawler
</content>
</invoke>
