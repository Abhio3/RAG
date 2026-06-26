"""PostgreSQL persistence layer (system of record) for the v2 data model.

See docs/DATA_MODEL.md §4. Raw psycopg, no ORM. Every owned row carries
``owner_id`` (defaults to the seeded System user until auth lands). Chunk rows
are the retrieval unit and their ``id`` is reused verbatim as the Qdrant point id.
"""
import os
from pathlib import Path
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

DSN = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# The seeded user that owns everything until Authentik/OIDC lands (see §8).
SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def init_schema() -> None:
    """Apply schema.sql on startup (idempotent — uses IF NOT EXISTS)."""
    with connect() as conn:
        conn.execute(SCHEMA_PATH.read_text())


# --- Model registry --------------------------------------------------------

def upsert_model(
    model_id: str,
    kind: str,
    provider: str = "ollama",
    context_len: Optional[int] = None,
    dimensions: Optional[int] = None,
    active: bool = True,
) -> None:
    """Register/refresh a model row so message/chunk FKs resolve (see §1, §7)."""
    with connect() as conn:
        conn.execute(
            """
            insert into models (id, kind, provider, context_len, dimensions, active)
            values (%s, %s, %s, %s, %s, %s)
            on conflict (id) do update set
                kind = excluded.kind,
                provider = excluded.provider,
                context_len = excluded.context_len,
                dimensions = excluded.dimensions,
                active = excluded.active
            """,
            (model_id, kind, provider, context_len, dimensions, active),
        )


def list_models() -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "select id, kind, provider, context_len, dimensions, active from models order by kind, id"
        ).fetchall()


# --- Chats & messages ------------------------------------------------------

def create_chat(title: str, mode: str = "chat", owner_id: str = SYSTEM_USER_ID) -> dict:
    with connect() as conn:
        return conn.execute(
            """
            insert into chats (title, mode, owner_id)
            values (%s, %s, %s)
            returning id, title, mode, created_at, updated_at
            """,
            (title, mode, owner_id),
        ).fetchone()


def list_chats(include_archived: bool = False) -> list[dict]:
    with connect() as conn:
        where = "" if include_archived else "where archived = false"
        return conn.execute(
            f"""
            select id, title, mode, archived, created_at, updated_at
            from chats {where}
            order by updated_at desc
            """
        ).fetchall()


def touch_chat(chat_id: str) -> None:
    """Bump updated_at so the most recently used chat sorts to the top."""
    with connect() as conn:
        conn.execute("update chats set updated_at = now() where id = %s", (chat_id,))


def add_message(
    chat_id: str,
    role: str,
    content: str,
    reasoning_content: Optional[str] = None,
    model_id: Optional[str] = None,
    token_usage: Optional[dict] = None,
    parent_id: Optional[str] = None,
) -> dict:
    with connect() as conn:
        return conn.execute(
            """
            insert into messages
                (chat_id, role, content, reasoning_content, model_id, token_usage, parent_id)
            values (%s, %s, %s, %s, %s, %s, %s)
            returning id, role, content, reasoning_content, model_id, token_usage, created_at
            """,
            (
                chat_id,
                role,
                content,
                reasoning_content,
                model_id,
                Jsonb(token_usage) if token_usage is not None else None,
                parent_id,
            ),
        ).fetchone()


def list_messages(chat_id: str) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """
            select id, role, content, reasoning_content, model_id, token_usage, created_at
            from messages
            where chat_id = %s
            order by created_at
            """,
            (chat_id,),
        ).fetchall()


def add_citations(message_id: str, citations: list[dict]) -> None:
    """citations: [{chunk_id, document_id, web_source_id, quote, rank, score}, ...]."""
    if not citations:
        return
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                insert into message_citations
                    (message_id, chunk_id, document_id, web_source_id, quote, rank, score)
                values (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        message_id,
                        c.get("chunk_id"),
                        c.get("document_id"),
                        c.get("web_source_id"),
                        c.get("quote"),
                        c.get("rank"),
                        c.get("score"),
                    )
                    for c in citations
                ],
            )


# --- Documents -------------------------------------------------------------

def find_document_by_hash(sha256: str, owner_id: str = SYSTEM_USER_ID) -> Optional[dict]:
    """Dedup lookup honouring the (owner_id, sha256) unique constraint."""
    with connect() as conn:
        return conn.execute(
            """
            select id, chat_id, title, original_name, status, storage_path, page_count
            from documents
            where owner_id = %s and sha256 = %s
            """,
            (owner_id, sha256),
        ).fetchone()


def insert_document(
    title: str,
    *,
    chat_id: Optional[str] = None,
    owner_id: str = SYSTEM_USER_ID,
    source_type: str = "upload",
    source_url: Optional[str] = None,
    original_name: Optional[str] = None,
    mime_type: Optional[str] = None,
    file_size: Optional[int] = None,
    storage_path: Optional[str] = None,
    sha256: Optional[str] = None,
    status: str = "pending",
    language: Optional[str] = None,
) -> dict:
    with connect() as conn:
        return conn.execute(
            """
            insert into documents
                (owner_id, chat_id, title, source_type, source_url, original_name,
                 mime_type, file_size, storage_path, sha256, status, language)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            returning id, chat_id, title, status, created_at
            """,
            (
                owner_id, chat_id, title, source_type, source_url, original_name,
                mime_type, file_size, storage_path, sha256, status, language,
            ),
        ).fetchone()


def update_document(document_id: str, **fields: Any) -> None:
    """Patch arbitrary document columns (status, storage_path, page_count, error, parsed_at…)."""
    if not fields:
        return
    cols = ", ".join(f"{k} = %s" for k in fields)
    with connect() as conn:
        conn.execute(
            f"update documents set {cols} where id = %s",
            (*fields.values(), document_id),
        )


def set_document_status(document_id: str, status: str, error: Optional[str] = None) -> None:
    with connect() as conn:
        conn.execute(
            "update documents set status = %s, error = %s where id = %s",
            (status, error, document_id),
        )


def mark_ready(document_id: str) -> None:
    """Flip a document to 'ready' and stamp parsed_at (SQL now())."""
    with connect() as conn:
        conn.execute(
            "update documents set status = 'ready', parsed_at = now() where id = %s",
            (document_id,),
        )


def list_documents(chat_id: Optional[str] = None) -> list[dict]:
    with connect() as conn:
        if chat_id:
            rows = conn.execute(
                """
                select d.id, d.chat_id, d.title, d.title as filename, d.original_name,
                       d.source_type, d.status, d.page_count, d.created_at,
                       coalesce(array_agg(t.name) filter (where t.name is not null), '{}') as tags
                from documents d
                left join document_tags dt on dt.document_id = d.id
                left join tags t on t.id = dt.tag_id
                where d.chat_id = %s
                group by d.id
                order by d.created_at desc
                """,
                (chat_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                select d.id, d.chat_id, d.title, d.title as filename, d.original_name,
                       d.source_type, d.status, d.page_count, d.created_at,
                       coalesce(array_agg(t.name) filter (where t.name is not null), '{}') as tags
                from documents d
                left join document_tags dt on dt.document_id = d.id
                left join tags t on t.id = dt.tag_id
                group by d.id
                order by d.created_at desc
                """
            ).fetchall()
    return rows


# --- Pages & chunks --------------------------------------------------------

def insert_pages(document_id: str, pages: list[dict]) -> None:
    """pages: [{page_no, text, image_path?, has_images?, ocr_model?, layout?}, ...]."""
    if not pages:
        return
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                insert into document_pages
                    (document_id, page_no, text, image_path, has_images, ocr_model, layout)
                values (%s, %s, %s, %s, %s, %s, %s)
                on conflict (document_id, page_no) do update set text = excluded.text
                """,
                [
                    (
                        document_id,
                        p["page_no"],
                        p.get("text"),
                        p.get("image_path"),
                        p.get("has_images", False),
                        p.get("ocr_model"),
                        Jsonb(p["layout"]) if p.get("layout") is not None else None,
                    )
                    for p in pages
                ],
            )


def insert_chunks(document_id: str, chunks: list[dict], *,
                  owner_id: str = SYSTEM_USER_ID, chat_id: Optional[str] = None,
                  embedding_model: Optional[str] = None) -> list[dict]:
    """Insert chunk rows and return [{id, chunk_index, page_no, content}, ...].

    The returned ``id`` is reused verbatim as the Qdrant point id (no mapping table).
    """
    if not chunks:
        return []
    with connect() as conn, conn.cursor() as cur:
        out: list[dict] = []
        for c in chunks:
            row = cur.execute(
                """
                insert into chunks
                    (document_id, owner_id, chat_id, page_no, chunk_index,
                     content, content_tokens, source_kind, embedding_model)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                returning id, chunk_index, page_no, content
                """,
                (
                    document_id, owner_id, chat_id, c.get("page_no"), c["chunk_index"],
                    c["content"], c.get("content_tokens"),
                    c.get("source_kind", "text"), embedding_model,
                ),
            ).fetchone()
            out.append(row)
    return out


# --- Tags ------------------------------------------------------------------

def attach_tags(document_id: str, tags: list[str]) -> None:
    if not tags:
        return
    with connect() as conn, conn.cursor() as cur:
        for name in tags:
            tag = cur.execute(
                """
                insert into tags (name) values (%s)
                on conflict (name) do update set name = excluded.name
                returning id
                """,
                (name,),
            ).fetchone()
            cur.execute(
                """
                insert into document_tags (document_id, tag_id) values (%s, %s)
                on conflict do nothing
                """,
                (document_id, tag["id"]),
            )
