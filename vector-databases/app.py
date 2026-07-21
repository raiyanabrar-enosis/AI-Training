from embedding import generate_embedding
from chunking import read_pdf_pages

# Embedding
text = "Python is an object-oriented programming language."

vector = generate_embedding(text)

print(f"Vector dimension: {len(vector)}")
print(vector[:10])


# Read pdf
pages = read_pdf_pages("documents/python.pdf")

print(f"Pages with text: {len(pages)}")
print(f"First page number: {pages[0]['page']}")
print(pages[0]["text"][:500])