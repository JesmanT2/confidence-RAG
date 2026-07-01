from __future__ import annotations

import os
from typing import Any, Dict

from app.rag_service import ConfidenceGatedRAG, load_json


def build_service(
    embedding_provider: str | None = None,
    llm_provider: str | None = None,
) -> ConfidenceGatedRAG:
    selected_embedding_provider = embedding_provider or os.getenv("RAG_EMBEDDING_PROVIDER", "semantic")
    selected_llm_provider = llm_provider or os.getenv("RAG_LLM_PROVIDER", "none")

    service = ConfidenceGatedRAG(
        embedding_provider=selected_embedding_provider,  # type: ignore[arg-type]
        llm_provider=selected_llm_provider,  # type: ignore[arg-type]
    )
    docs = load_json("data/documents.json")
    train = load_json("data/train_questions.json")

    service.ingest_documents(docs)
    service.fit_gate(train)
    return service


def run_eval(threshold: float = 0.55) -> Dict[str, Any]:
    service = build_service()
    eval_set = load_json("data/eval_questions.json")
    return service.evaluate(eval_set, threshold=threshold)
