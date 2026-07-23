import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_databases.config import DOCUMENTS_DIR, BATCH_SIZE
from vector_databases.chunking import read_pdf_pages
from vector_databases.embedding import generate_embeddings, generate_embedding

from vector_pgvector.db import connect, insert_into_db, find_similar_vectors


def chunk_pages(pages, chunk_size=500, overlap=100):
    """
    Turn page records into overlapping chunks.

    chunk_size and overlap are measured in CHARACTERS, but we only ever
    cut on spaces, so words stay intact. Each chunk remembers its page.
    """
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        chunk_overlap=overlap,
    )

    for page in pages:
        chunks.extend(splitter.split_text(page["text"]))

    return chunks


def index_doc(filename, connection_cursor):
    path = os.path.join(DOCUMENTS_DIR, filename)

    if not path:
        print("No PDFs found in the given name. Add one and try again.")
        return

    print(f"Reading {filename} ...")
    pages = read_pdf_pages(path)
    chunks = chunk_pages(pages)
    print(f"  {len(chunks)} chunks")

    for start in range(0, len(chunks), BATCH_SIZE):
        slice_ = chunks[start : start + BATCH_SIZE]
        texts = [c for c in slice_]

        vectors = generate_embeddings(texts)
        if len(vectors) != len(slice_):
            raise RuntimeError(f"Expected {len(slice_)} embeddings, got {len(vectors)}")

        records = []
        for chunk, vector in zip(slice_, vectors):
            records.append(
                {
                    "embedding": vector,
                    "content": chunk,
                }
            )

        for record in records:
            insert_into_db(record['content'], record['embedding'], connection_cursor)

def find_similarities(question, cursor):
    query_vector = generate_embedding(question)

    find_similar_vectors(query_vector, cursor)

if __name__ == "__main__":
    connection, connection_cursor = connect()

    index_doc("robotics_paper.pdf", connection_cursor)

    connection.commit()

    find_similarities("Summarize the main points of the paper", connection_cursor)

    connection_cursor.close()
    connection.close()
