from fastapi import FastAPI
from pydantic import BaseModel

from rag_project.rag.pipeline import ingest_documents, query_pipeline

app = FastAPI()


# -------------------------
# REQUEST MODELS
# -------------------------

class IngestRequest(BaseModel):
    text: str


class QueryRequest(BaseModel):
    question: str


# -------------------------
# ROUTES
# -------------------------

@app.post("/ingest")
def ingest(doc: IngestRequest):
    try:
        result = ingest_documents(doc.text)  # should return {"chunks_added": X}
        return result
    except Exception as e:
        return {"error": f"Ingest failed: {str(e)}"}


@app.post("/ask")
def ask(q: QueryRequest):
    try:
        answer = query_pipeline(q.question)

        # Ensure answer is always a clean string
        if hasattr(answer, "content"):
            answer = answer.content

        answer = str(answer).strip()

        return {"answer": answer}

    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}


@app.get("/")
def home():
    return {"message": "RAG API is running!"}
