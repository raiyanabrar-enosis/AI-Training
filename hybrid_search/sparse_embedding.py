from fastembed import SparseTextEmbedding

from hybrid_search.config import BM25_MODEL

# Loaded once at import (mirrors vector_databases/embedding.py).
_model = SparseTextEmbedding(BM25_MODEL)


def generate_sparse_embeddings(texts):
    """Document-side BM25 vectors. Returns list of (indices, values)."""
    out = []
    for e in _model.embed(texts):
        out.append((e.indices.tolist(), e.values.tolist()))
    return out


def generate_sparse_query(text):
    """Query-side BM25 vector (no doc-length weighting)."""
    e = next(_model.query_embed(text))
    return e.indices.tolist(), e.values.tolist()
