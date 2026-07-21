from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

EMBEDDING_MODEL = "gemini-embedding-2"

COLLECTION_NAME = "documents"