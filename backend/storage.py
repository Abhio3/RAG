"""Supabase Storage: raw files and rendered page images (see docs/DATA_MODEL.md §1).

Postgres keeps only ``storage_path`` + ``sha256``; the bytes live here. The client is
lazy + cached so the API starts without Supabase configured — storage is only required
when something is actually uploaded.

Config (backend/.env):
    SUPABASE_URL=http://localhost:54321
    SUPABASE_SERVICE_ROLE_KEY=...        # service role; bypasses RLS for server-side writes
    SUPABASE_BUCKET=documents            # optional, defaults to 'documents'
"""
import os
from functools import lru_cache

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
BUCKET = os.environ.get("SUPABASE_BUCKET", "documents")


class StorageNotConfigured(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


@lru_cache(maxsize=1)
def _client():
    if not is_configured():
        raise StorageNotConfigured(
            "Supabase Storage is not configured. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY in backend/.env."
        )
    from supabase import create_client  # lazy import

    return create_client(SUPABASE_URL, SUPABASE_KEY)


def ensure_bucket() -> None:
    """Create the private storage bucket if it doesn't exist (idempotent)."""
    client = _client()
    try:
        client.storage.get_bucket(BUCKET)
    except Exception:
        # create_bucket raises if it already exists; swallow that race.
        try:
            client.storage.create_bucket(BUCKET, options={"public": False})
        except Exception:
            pass


def upload(storage_path: str, data: bytes, content_type: str) -> str:
    """Upload bytes to ``BUCKET/storage_path`` (upsert) and return the storage path."""
    client = _client()
    client.storage.from_(BUCKET).upload(
        path=storage_path,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return storage_path


def signed_url(storage_path: str, expires_in: int = 3600) -> str:
    """Time-limited download URL for a stored object."""
    client = _client()
    res = client.storage.from_(BUCKET).create_signed_url(storage_path, expires_in)
    return res.get("signedURL") or res.get("signedUrl", "")
