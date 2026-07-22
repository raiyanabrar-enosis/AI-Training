from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from vector_databases.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from vector_databases.embedding import generate_embedding

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_embedding_dimension():
    """Ask the model how many dimensions it returns. Never guess."""
    return len(generate_embedding("dimension probe"))


def ensure_collection():
    """Create the collection once. Safe to call repeatedly."""
    if client.collection_exists(COLLECTION_NAME):
        return

    dimension = get_embedding_dimension()
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection '{COLLECTION_NAME}' with dimension {dimension}")


def upsert_chunks(records):
    """
    records: list of dicts with keys: id, vector, text, page, source
    """
    points = [
        PointStruct(
            id=r["id"],
            vector=r["vector"],
            payload={
                "text": r["text"],
                "page": r["page"],
                "source": r["source"],
            },
        )
        for r in records
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
