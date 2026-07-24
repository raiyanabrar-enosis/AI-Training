from qdrant_client import QdrantClient
from qdrant_client import models

from vector_databases.config import QDRANT_HOST, QDRANT_PORT
from vector_databases.embedding import generate_embedding
from hybrid_search.config import HYBRID_COLLECTION, DENSE_NAME, SPARSE_NAME

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_dim():
    """Ask the model how many dimensions it returns. Never guess."""
    return len(generate_embedding("dimension probe"))

def delete_collection():
    if client.collection_exists(HYBRID_COLLECTION):
        return

    client.delete_collection(collection_name=HYBRID_COLLECTION)

def ensure_collection():
    """Create the hybrid collection once. Safe to call repeatedly."""
    if client.collection_exists(HYBRID_COLLECTION):
        return

    client.create_collection(
        collection_name=HYBRID_COLLECTION,
        vectors_config={
            DENSE_NAME: models.VectorParams(size=get_dim(), distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            # IDF applied server-side => proper BM25 scoring.
            SPARSE_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    print(f"Created '{HYBRID_COLLECTION}'")


def upsert_chunks(records):
    """
    records: list of dicts with keys: id, dense (list[float]),
    sparse (indices, values), text, page, source
    """
    points = []
    for r in records:
        idx, val = r["sparse"]
        points.append(
            models.PointStruct(
                id=r["id"],
                vector={
                    DENSE_NAME: r["dense"],
                    SPARSE_NAME: models.SparseVector(indices=idx, values=val),
                },
                payload={"text": r["text"], "page": r["page"], "source": r["source"]},
            )
        )
    client.upsert(collection_name=HYBRID_COLLECTION, points=points)
