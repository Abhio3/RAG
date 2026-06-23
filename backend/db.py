"""PostgreSQL persistence: documents, chat threads, and messages."""
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get(
    "DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"
)
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def init_schema() -> None:
    """Apply schema.sql on startup (idempotent — uses IF NOT EXISTS)."""
    with connect() as conn:
        conn.execute(SCHEMA_PATH.read_text())


# --- Documents -------------------------------------------------------------

def insert_document(
    filename: str,
    content: str,
    tags: list[str],
    chunk_count: int,
    content_type: str,
    file_data: bytes,
) -> dict:
    with connect() as conn:
        row = conn.execute(
            """
            insert into documents
                (filename, content, tags, chunk_count, content_type, file_data)
            values (%s, %s, %s, %s, %s, %s)
            returning id, filename, tags, chunk_count, created_at
            """,
            (filename, content, tags, chunk_count, content_type, file_data),
        ).fetchone()
    return row


def list_documents() -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """
            select id, filename, tags, chunk_count, created_at
            from documents
            order by created_at desc
            """
        ).fetchall()


# --- Chats & messages ------------------------------------------------------

def create_chat(title: str) -> dict:
    with connect() as conn:
        return conn.execute(
            "insert into chats (title) values (%s) returning id, title, created_at",
            (title,),
        ).fetchone()


def list_chats() -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "select id, title, created_at from chats order by created_at desc"
        ).fetchall()


def add_message(chat_id: str, role: str, content: str) -> dict:
    with connect() as conn:
        return conn.execute(
            """
            insert into messages (chat_id, role, content)
            values (%s, %s, %s)
            returning id, role, content, created_at
            """,
            (chat_id, role, content),
        ).fetchone()


def list_messages(chat_id: str) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """
            select id, role, content, created_at
            from messages
            where chat_id = %s
            order by created_at
            """,
            (chat_id,),
        ).fetchall()
