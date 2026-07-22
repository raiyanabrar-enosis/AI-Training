import os

from vector_databases.chunking import read_pdf_pages, chunk_pages
from vector_databases.embedding import generate_embeddings
from vector_databases.qdrant_db import ensure_collection, upsert_chunks

DOCUMENTS_DIR = "documents"
BATCH_SIZE = 50


def index_all():
    ensure_collection()

    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")]
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
            slice_ = chunks[start:start + BATCH_SIZE]
            texts = [c["text"] for c in slice_]

            vectors = generate_embeddings(texts)     # ONE request for 50 chunks
            if len(vectors) != len(slice_):
                raise RuntimeError(
                    f"Expected {len(slice_)} embeddings, got {len(vectors)}"
                )

            records = []
            for chunk, vector in zip(slice_, vectors):
                records.append({
                    "id": point_id,
                    "vector": vector,
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": filename,
                })
                point_id += 1

            upsert_chunks(records)
            print(f"  stored {point_id} chunks so far")

    print(f"Done. Indexed {point_id} chunks from {len(pdf_files)} PDF(s).")


if __name__ == "__main__":
    index_all()