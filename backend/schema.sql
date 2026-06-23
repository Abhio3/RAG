-- RAG App schema (PostgreSQL)
-- documents: one row per uploaded file (metadata + extracted text + tags + raw file bytes)
-- chats:     a conversation thread
-- messages:  individual user/assistant turns within a chat

create table if not exists documents (
    id           uuid primary key default gen_random_uuid(),
    filename     text not null,
    content      text,
    tags         text[] not null default '{}',
    chunk_count  integer not null default 0,
    content_type text,
    file_data    bytea,
    created_at   timestamptz not null default now()
);

create table if not exists chats (
    id         uuid primary key default gen_random_uuid(),
    title      text not null default 'New chat',
    created_at timestamptz not null default now()
);

create table if not exists messages (
    id         uuid primary key default gen_random_uuid(),
    chat_id    uuid not null references chats(id) on delete cascade,
    role       text not null check (role in ('user', 'assistant')),
    content    text not null,
    created_at timestamptz not null default now()
);

create index if not exists messages_chat_id_idx on messages(chat_id, created_at);
create index if not exists documents_created_at_idx on documents(created_at desc);
