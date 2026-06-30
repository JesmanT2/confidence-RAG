from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any] | None = None


class ConfidenceGatedRAG:
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.classifier = LogisticRegression(max_iter=1000)
        self._training_data: List[Dict[str, Any]] = []

    def ingest_documents(self, docs: List[Dict[str, Any]]) -> None:
        self.chunks = [Chunk(id=doc["id"], text=doc["text"], metadata=doc.get("metadata")) for doc in docs]
        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.vectorizer.fit([chunk.text for chunk in self.chunks])

    def fit_gate(self, training_examples: List[Dict[str, Any]]) -> None:
        self._training_data = training_examples
        X = []
        y = []
        for example in training_examples:
            features = self._build_features(example["query"])
            X.append(features)
            y.append(1 if example["answerable"] else 0)
        if len(X) < 2:
            raise ValueError("Need at least two training examples")
        self.classifier.fit(np.array(X), np.array(y))

    def answer_query(self, query: str, threshold: float = 0.6) -> Dict[str, Any]:
        if not self.chunks:
            return {
                "answer": "I don't have enough information to answer that confidently.",
                "abstained": True,
                "confidence": 0.0,
            }

        features = self._build_features(query)
        if not hasattr(self.classifier, "classes_"):
            score = 0.5
        else:
            score = float(self.classifier.predict_proba(np.array([features]))[0][1])
        if score < threshold:
            return {
                "answer": "I don't have enough information to answer that confidently.",
                "abstained": True,
                "confidence": score,
            }

        top_chunk = self._retrieve_top_chunk(query)
        return {
            "answer": f"Based on the retrieved context: {top_chunk.text}",
            "abstained": False,
            "confidence": score,
            "context": top_chunk.text,
        }

    def answer_query_baseline(self, query: str) -> Dict[str, Any]:
        top_chunk = self._retrieve_top_chunk(query)
        return {
            "answer": f"Based on the retrieved context: {top_chunk.text}",
            "abstained": False,
            "context": top_chunk.text,
        }

    def evaluate(self, eval_examples: List[Dict[str, Any]], threshold: float = 0.6) -> Dict[str, Any]:
        baseline_errors = 0
        gated_errors = 0
        results = []
        for example in eval_examples:
            baseline = self.answer_query_baseline(example["query"])
            gated = self.answer_query(example["query"], threshold=threshold)

            baseline_error = 0 if example["answerable"] else 1
            gated_error = 1 if gated["abstained"] != example["answerable"] else 0

            baseline_errors += baseline_error
            gated_errors += gated_error
            results.append(
                {
                    "query": example["query"],
                    "expected_answerable": example["answerable"],
                    "baseline_abstained": baseline["abstained"],
                    "gated_abstained": gated["abstained"],
                }
            )

        return {
            "baseline_hallucination_rate": baseline_errors / len(eval_examples),
            "gated_hallucination_rate": gated_errors / len(eval_examples),
            "baseline_errors": baseline_errors,
            "gated_errors": gated_errors,
            "results": results,
        }

    def _retrieve_top_chunk(self, query: str) -> Chunk:
        if not self.chunks:
            raise ValueError("No documents ingested")
        docs = [chunk.text for chunk in self.chunks]
        doc_matrix = self.vectorizer.transform(docs)
        query_matrix = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_matrix, doc_matrix)[0]
        best_index = int(np.argmax(similarities))
        return self.chunks[best_index]

    def _build_features(self, query: str) -> List[float]:
        if not self.chunks:
            return [0.0, 0.0, 0.0]
        docs = [chunk.text for chunk in self.chunks]
        doc_matrix = self.vectorizer.transform(docs)
        query_matrix = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_matrix, doc_matrix)[0]
        top_sim = float(np.max(similarities)) if len(similarities) else 0.0
        sorted_sims = np.sort(similarities)
        if len(sorted_sims) >= 2:
            gap = float(sorted_sims[-1] - sorted_sims[-2])
        else:
            gap = 0.0
        chunk_count = float(len(self.chunks))
        return [top_sim, gap, chunk_count]
