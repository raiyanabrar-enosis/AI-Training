from google import genai
from google.genai import errors
from tenacity import retry, wait_exponential, retry_if_exception_type

from config import GEMINI_API_KEY
from config import EMBEDDING_MODEL

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# This will wait 2s, then 4s, then 8s, up to 60s if it hits a ClientError (like a 429)
@retry(
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(errors.ClientError)
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