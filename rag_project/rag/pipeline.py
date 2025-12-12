import os
from dotenv import load_dotenv
import weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# Load .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------
# WEAVIATE CLIENT (v3) - REST ONLY
# -------------------------

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
)

# Create schema if missing
schema = {
    "classes": [
        {
            "class": "Document",
            "properties": [
                {"name": "content", "dataType": ["text"]}
            ]
        }
    ]
}

existing = client.schema.get()

if not any(cls["class"] == "Document" for cls in existing.get("classes", [])):
    client.schema.create(schema["classes"][0])

# -------------------------
# TEXT SPLITTER
# -------------------------

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# -------------------------
# INGEST
# -------------------------

def ingest_documents(text: str):
    chunks = splitter.split_text(text)

    for chunk in chunks:
        client.data_object.create(
            {"content": chunk},
            "Document"
        )

    return {"chunks_added": len(chunks)}

# -------------------------
# QUERY
# -------------------------

def query_pipeline(question: str):
    result = client.query.get(
        "Document",
        ["content"]
    ).with_bm25(
        query=question,
        properties=["content"]
    ).with_limit(3).do()

    docs = result["data"]["Get"].get("Document", [])

    if not docs:
        return "No relevant documents were found."

    found_text = "\n\n".join(d["content"] for d in docs)

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": f"""
                Context:
                {found_text}

                Question: {question}

                Answer:
                """
            }
        ]
    )

    # FIX HERE
    return response.choices[0].message.content
