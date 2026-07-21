from google import genai

from config import GEMINI_API_KEY
from config import EMBEDDING_MODEL

client = genai.Client(
    api_key=GEMINI_API_KEY
)


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding for the given text.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values