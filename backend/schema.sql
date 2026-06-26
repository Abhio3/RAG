-- RAG App v2 schema (PostgreSQL / Supabase) — see docs/DATA_MODEL.md §4.
-- Idempotent: applied on startup via db.init_schema() (uses IF NOT EXISTS).
-- Postgres is the system of record. Qdrant holds chunk vectors (point id == chunks.id).
-- Raw files live in Supabase Storage; Postgres keeps storage_path + sha256 only.

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
