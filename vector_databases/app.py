from vector_databases.embedding import generate_embedding
from vector_databases.chunking import read_pdf_pages, read_pdf_pages_unicode, chunk_pages
from vector_databases.qdrant_db import client, ensure_collection

# Embedding example
text = "Python is an object-oriented programming language."
print("Embedding an example text...")
vector = generate_embedding(text)

print(f"Vector dimension: {len(vector)}")
print(vector[:10])


# Read pdf
pages = read_pdf_pages("documents/python.pdf")

print(f"Pages with text: {len(pages)}")
print(f"First page number: {pages[0]['page']}")
print(pages[0]["text"][:500])

# Bengali
pages = read_pdf_pages_unicode("documents/bengali.pdf")

print(f"Pages with text: {len(pages)}")
print(f"First page number: {pages[0]['page']}")
print(pages[0]["text"][:500])

# Chunking
chunks = chunk_pages(pages)

print(f"Total chunks: {len(chunks)}")
print(f"First chunk is from page {chunks[0]['page']}")
print(chunks[0]["text"][:300])
print("---")
print(chunks[1]["text"][:300])

# Check and check collection
ensure_collection()
print(client.get_collections())