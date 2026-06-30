from fastapi import FastAPI, Query
from app.rag_service import ConfidenceGatedRAG

app = FastAPI(title="Confidence-Gated RAG")
service = ConfidenceGatedRAG()

service.ingest_documents([
    {"id": "doc-1", "text": "The deductible for comprehensive claims is $500."},
    {"id": "doc-2", "text": "Flood coverage has a 30-day waiting period."},
])
service.fit_gate([
    {"query": "What is the deductible for comprehensive claims?", "answerable": True},
    {"query": "How long is the waiting period for flood coverage?", "answerable": True},
    {"query": "What is the jewelry limit?", "answerable": False},
])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(query: str = Query(...)) -> dict:
    return service.answer_query(query)
