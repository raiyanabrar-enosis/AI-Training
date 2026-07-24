import os

from vector_databases.chunking import read_pdf_pages, chunk_pages
from vector_databases.embedding import generate_embeddings
from vector_databases.config import DOCUMENTS_DIR, BATCH_SIZE

from hybrid_search.sparse_embedding import generate_sparse_embeddings
from hybrid_search.qdrant_hybrid import ensure_collection, upsert_chunks


def index_all():
    ensure_collection()

    pdf_files = sorted(f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf"))
    if not pdf_files:
        print("No PDFs found in documents/. Add one and try again.")
        return

    point_id = 0

    for filename in pdf_files:
        path = os.path.join(DOCUMENTS_DIR, filename)
        print(f"Reading {filename} ...")

        pages = read_pdf_pages(path)
        chunks = chunk_pages(pages)
        print(f"  {len(chunks)} chunks")

        for start in range(0, len(chunks), BATCH_SIZE):
            slice_ = chunks[start : start + BATCH_SIZE]
            texts = [c["text"] for c in slice_]

            dense_vectors = generate_embeddings(texts)
            sparse_vectors = generate_sparse_embeddings(texts)
            if len(dense_vectors) != len(slice_):
                raise RuntimeError(
                    f"Expected {len(slice_)} embeddings, got {len(dense_vectors)}"
                )

            records = []
            for chunk, dense, sparse in zip(slice_, dense_vectors, sparse_vectors):
                records.append(
                    {
                        "id": point_id,
                        "dense": dense,
                        "sparse": sparse,
                        "text": chunk["text"],
                        "page": chunk["page"],
                        "source": filename,
                    }
                )
                point_id += 1

            upsert_chunks(records)
            print(f"  stored {point_id} chunks so far")

    print(f"Done. Indexed {point_id} chunks from {len(pdf_files)} PDF(s).")


if __name__ == "__main__":
    index_all()
