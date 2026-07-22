from sentence_transformers import SentenceTransformer

from vector_databases.config import EMBEDDING_MODEL, BATCH_SIZE

# Loaded once at import. First run downloads the model (~2GB for BGE-M3).
model = SentenceTransformer(EMBEDDING_MODEL)


def generate_embedding(text: str) -> list[float]:
    """Embed a single string."""
    return model.encode(text, normalize_embeddings=True).tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in one batched pass."""
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [v.tolist() for v in vectors]