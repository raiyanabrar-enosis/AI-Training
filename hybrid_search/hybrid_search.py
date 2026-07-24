from qdrant_client import models

from vector_databases.embedding import generate_embedding
from hybrid_search.sparse_embedding import generate_sparse_query
from hybrid_search.qdrant_hybrid import client
from hybrid_search.config import HYBRID_COLLECTION, DENSE_NAME, SPARSE_NAME, LIMIT


def _fmt(points):
    return [
        {
            "score": p.score,
            "text": p.payload["text"],
            "page": p.payload["page"],
            "source": p.payload["source"],
        }
        for p in points
    ]


def search_dense(question, limit=LIMIT):
    query_vector = generate_embedding(question)
    results = client.query_points(
        collection_name=HYBRID_COLLECTION,
        query=query_vector,
        using=DENSE_NAME,
        limit=limit,
        with_payload=True,
    )
    return _fmt(results.points)


def search_sparse(question, limit=LIMIT):
    idx, val = generate_sparse_query(question)
    results = client.query_points(
        collection_name=HYBRID_COLLECTION,
        query=models.SparseVector(indices=idx, values=val),
        using=SPARSE_NAME,
        limit=limit,
        with_payload=True,
    )
    return _fmt(results.points)


def search_hybrid(question, limit=LIMIT, prefetch_limit=20):
    idx, val = generate_sparse_query(question)
    results = client.query_points(
        collection_name=HYBRID_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=generate_embedding(question), using=DENSE_NAME, limit=prefetch_limit
            ),
            models.Prefetch(
                query=models.SparseVector(indices=idx, values=val),
                using=SPARSE_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return _fmt(results.points)


if __name__ == "__main__":
    question = input("Question: ")
    for name, fn in [("DENSE", search_dense), ("SPARSE", search_sparse), ("HYBRID", search_hybrid)]:
        print(f"\n=== {name} ===")
        for hit in fn(question):
            print(f"[{hit['score']:.3f}] {hit['source']} p.{hit['page']}  {hit['text']}")
