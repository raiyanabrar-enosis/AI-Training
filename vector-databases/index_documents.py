import os

from chunking import read_pdf_pages, chunk_pages
from embedding import generate_embedding
from qdrant_db import ensure_collection, upsert_chunks

DOCUMENTS_DIR = "documents"
BATCH_SIZE = 50


def index_all():
    ensure_collection()

    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found in documents/. Add one and try again.")
        return

    point_id = 0
    batch = []

    for filename in pdf_files:
        path = os.path.join(DOCUMENTS_DIR, filename)
        print(f"Reading {filename} ...")

        pages = read_pdf_pages(path)
        chunks = chunk_pages(pages)
        print(f"  {len(chunks)} chunks")

        for chunk in chunks:
            vector = generate_embedding(chunk["text"])
            batch.append({
                "id": point_id,
                "vector": vector,
                "text": chunk["text"],
                "page": chunk["page"],
                "source": filename,
            })
            point_id += 1

            if len(batch) >= BATCH_SIZE:
                upsert_chunks(batch)
                print(f"  stored {point_id} chunks so far")
                batch = []

    if batch:
        upsert_chunks(batch)

    print(f"Done. Indexed {point_id} chunks from {len(pdf_files)} PDF(s).")


if __name__ == "__main__":
    index_all()