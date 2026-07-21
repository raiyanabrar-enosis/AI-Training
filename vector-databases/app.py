from embedding import generate_embedding

text = "Python is an object-oriented programming language."

vector = generate_embedding(text)

print(f"Vector dimension: {len(vector)}")

print(vector[:10])