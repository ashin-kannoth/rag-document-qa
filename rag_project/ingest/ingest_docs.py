import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Property, DataType

# Load environment variables
load_dotenv(".env")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Connect to Weaviate (cloud or local)
client = weaviate.connect_to_local() if not WEAVIATE_URL else weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)

CLASS_NAME = "Docs"

# Ensure schema
def ensure_schema():
    if not client.collections.exists(CLASS_NAME):
        client.collections.create(
            name=CLASS_NAME,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
            ]
        )
        print("Schema created!")

# Ingest text
def ingest_text(text):
    ensure_schema()
    collection = client.collections.get(CLASS_NAME)
    
    collection.data.insert({
        "text": text
    })

    print("Document ingested successfully!")

if __name__ == "__main__":
    ingest_text("This is a test document for RAG pipeline.")
