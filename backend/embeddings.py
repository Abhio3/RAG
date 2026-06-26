"""BGE-M3 hybrid embeddings, BGE reranker, and token-based chunking.

Runs fully local on the DGX Spark (no cloud). BGE-M3 yields a dense (1024-d) and a
sparse (lexical) vector per text in one pass — that sparse vector is what gives us
hybrid search for free (see docs/DATA_MODEL.md §5). The reranker re-scores Qdrant's
fused top-k before it reaches the LLM.

Heavy deps (FlagEmbedding/torch/transformers) are imported lazily so the API process
starts — and the schema/storage paths stay testable — even before the models are
pulled. The first embed/rerank call loads (and on first run downloads) the weights.
"""
import os
from functools import lru_cache
from typing import Optional

# Model ids are overridable via env so swapping to a vLLM-served variant is config-only.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
EMBED_DIM = 1024  # BGE-M3 dense dimension

# Token-based chunking (§5): BGE-M3's 8192-token window lets us go bigger than v1's
# char-based 500/50. 512/64 tokens is the proposed default.
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "64"))

# A sparse vector encoded the way Qdrant expects: parallel indices/values arrays.
SparseVec = dict  # {"indices": list[int], "values": list[float]}


@lru_cache(maxsize=1)
def _embedder():
    from FlagEmbedding import BGEM3FlagModel  # lazy: heavy import

    return BGEM3FlagModel(EMBED_MODEL, use_fp16=True)


@lru_cache(maxsize=1)
def _reranker():
    from FlagEmbedding import FlagReranker  # lazy: heavy import

    return FlagReranker(RERANK_MODEL, use_fp16=True)


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer  # lazy: heavy import

    return AutoTokenizer.from_pretrained(EMBED_MODEL)


def _to_sparse(lexical_weights: dict) -> SparseVec:
    """BGE-M3 returns {token_id(str): weight}; Qdrant wants indices/values arrays."""
    if not lexical_weights:
        return {"indices": [], "values": []}
    items = [(int(k), float(v)) for k, v in lexical_weights.items()]
    return {
        "indices": [i for i, _ in items],
        "values": [v for _, v in items],
    }


def embed_texts(texts: list[str]) -> list[dict]:
    """Embed a batch → [{"dense": [float]*1024, "sparse": {indices, values}}, ...]."""
    if not texts:
        return []
    out = _embedder().encode(texts, return_dense=True, return_sparse=True)
    dense = out["dense_vecs"]
    sparse = out["lexical_weights"]
    return [
        {"dense": [float(x) for x in dense[i]], "sparse": _to_sparse(sparse[i])}
        for i in range(len(texts))
    ]


def embed_query(text: str) -> dict:
    """Embed a single query → {"dense": [...], "sparse": {indices, values}}."""
    return embed_texts([text])[0]


def rerank(query: str, passages: list[str]) -> list[float]:
    """Relevance score per passage (higher = more relevant)."""
    if not passages:
        return []
    scores = _reranker().compute_score([[query, p] for p in passages], normalize=True)
    # FlagReranker returns a float for a single pair, else a list.
    return [float(scores)] if isinstance(scores, (int, float)) else [float(s) for s in scores]


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def chunk_text(text: str) -> list[str]:
    """Split text into ~CHUNK_TOKENS windows with CHUNK_OVERLAP_TOKENS overlap.

    Token-based using the BGE-M3 tokenizer so chunk sizes match what the embedder sees.
    """
    text = (text or "").strip()
    if not text:
        return []
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= CHUNK_TOKENS:
        return [text]

    step = max(1, CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS)
    chunks: list[str] = []
    for start in range(0, len(ids), step):
        window = ids[start : start + CHUNK_TOKENS]
        if not window:
            break
        piece = tok.decode(window, skip_special_tokens=True).strip()
        if piece:
            chunks.append(piece)
        if start + CHUNK_TOKENS >= len(ids):
            break
    return chunks
