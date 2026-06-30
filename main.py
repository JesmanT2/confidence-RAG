from typing import Any, Dict

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from app.pipeline import build_service, run_eval

app = FastAPI(title="Confidence-Gated RAG")
service = build_service()


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3)
    threshold: float = Field(default=0.552, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ask")
def ask_get(query: str = Query(...), threshold: float = Query(0.552, ge=0.0, le=1.0)) -> Dict[str, Any]:
    return service.answer_query(query, threshold=threshold, use_gate=True)


@app.post("/ask")
def ask_post(request: AskRequest) -> Dict[str, Any]:
    return service.answer_query(request.query, threshold=request.threshold, use_gate=True)


@app.get("/ask/baseline")
def ask_baseline(query: str = Query(...), threshold: float = Query(0.552, ge=0.0, le=1.0)) -> Dict[str, Any]:
    return service.answer_query(query, threshold=threshold, use_gate=False)


@app.get("/eval")
def eval_report(threshold: float = Query(0.552, ge=0.0, le=1.0)) -> Dict[str, Any]:
    return run_eval(threshold=threshold)
