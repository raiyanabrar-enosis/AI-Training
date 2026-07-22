from vector_databases.qdrant_db import client
from vector_databases.embedding import generate_embedding
from vector_databases.config import COLLECTION_NAME


def search(question, limit=5):
    query_vector = generate_embedding(question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    hits = []
    for point in results.points:
        hits.append({
            "score": point.score,
            "text": point.payload["text"],
            "page": point.payload["page"],
            "source": point.payload["source"],
        })
    return hits


if __name__ == "__main__":
    question = input("Enter your related question: ")
    for hit in search(question):
        print(f"[{hit['score']:.3f}] {hit['source']} p.{hit['page']}")
        print(hit["text"])
        print("---")