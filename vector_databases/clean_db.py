from vector_databases.qdrant_db import client
from vector_databases.config import COLLECTION_NAME

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)