from __future__ import annotations

from typing import Any, Dict

from app.rag_service import ConfidenceGatedRAG, load_json


def build_service() -> ConfidenceGatedRAG:
    service = ConfidenceGatedRAG()
    docs = load_json("data/documents.json")
    train = load_json("data/train_questions.json")

    service.ingest_documents(docs)
    service.fit_gate(train)
    return service


def run_eval(threshold: float = 0.55) -> Dict[str, Any]:
    service = build_service()
    eval_set = load_json("data/eval_questions.json")
    return service.evaluate(eval_set, threshold=threshold)
