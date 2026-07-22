import time
from google import genai

from vector_databases.config import GEMINI_API_KEY
from vector_databases.config import EMBEDDING_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding for the given text.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        out.append(generate_embedding(t))
        time.sleep(0.5)  # stay under the per-minute rate limit
    return out
