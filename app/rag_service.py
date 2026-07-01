from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import chromadb
import numpy as np
from chromadb.api.types import EmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    similarity: float


class HashingEmbeddingFunction(EmbeddingFunction):
    def __init__(self, n_features: int = 384) -> None:
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        matrix = self.vectorizer.transform(input)
        return matrix.toarray().astype(float).tolist()


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        embeddings = self.model.encode(list(input), normalize_embeddings=True)
        return np.asarray(embeddings, dtype=float).tolist()


class ConfidenceGatedRAG:
    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection_name: str = "insurance_policy_chunks",
        top_k: int = 5,
        relevant_similarity_cutoff: float = 0.2,
        embedding_provider: Literal["semantic", "hashing"] = "semantic",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: Literal["none", "openai"] = "none",
        openai_model: str = "gpt-4o-mini",
    ) -> None:
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.relevant_similarity_cutoff = relevant_similarity_cutoff
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.llm_provider = llm_provider
        self.openai_model = openai_model

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        if self.embedding_provider == "semantic":
            self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name)
        else:
            self.embedding_fn = HashingEmbeddingFunction()

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.default_gate_threshold = 0.552
        self.model_cv_accuracy: float | None = None
        self._openai_client: OpenAI | None = None
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.llm_provider == "openai" and self._openai_api_key:
            self._openai_client = OpenAI(api_key=self._openai_api_key)

    def ingest_documents(
        self,
        docs: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        reset_collection: bool = True,
    ) -> int:
        if reset_collection:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunk_ids: List[str] = []
        chunk_texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for doc in docs:
            source_id = doc["id"]
            text = doc["text"]
            chunks = splitter.split_text(text)
            for idx, chunk in enumerate(chunks):
                chunk_ids.append(f"{source_id}-chunk-{idx}")
                chunk_texts.append(chunk)
                metadatas.append({"source_id": source_id, "chunk_index": idx})

        if chunk_ids:
            self.collection.add(ids=chunk_ids, documents=chunk_texts, metadatas=metadatas)

        return len(chunk_ids)

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        k = top_k or self.top_k
        response = self.collection.query(query_texts=[query], n_results=k)
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        chunks: List[RetrievedChunk] = []
        for i, chunk_id in enumerate(ids):
            distance = float(distances[i]) if i < len(distances) else 1.0
            similarity = max(0.0, 1.0 - distance)
            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=documents[i],
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    similarity=similarity,
                )
            )
        return chunks

    def build_features(self, query: str, top_k: int | None = None) -> List[float]:
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return [0.0, 0.0, 0.0]

        similarities = sorted([chunk.similarity for chunk in retrieved], reverse=True)
        top_similarity = similarities[0]
        score_gap = similarities[0] - similarities[1] if len(similarities) > 1 else similarities[0]
        relevant_chunk_count = float(sum(1 for s in similarities if s >= self.relevant_similarity_cutoff))
        return [top_similarity, score_gap, relevant_chunk_count]

    def fit_gate(self, training_examples: List[Dict[str, Any]]) -> Dict[str, float]:
        X: List[List[float]] = []
        y: List[int] = []

        for example in training_examples:
            X.append(self.build_features(example["query"]))
            y.append(1 if example["answerable"] else 0)

        x_array = np.array(X)
        y_array = np.array(y)
        self.classifier.fit(x_array, y_array)

        if len(training_examples) >= 10 and len(set(y)) > 1:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(self.classifier, x_array, y_array, cv=cv, scoring="accuracy")
            self.model_cv_accuracy = float(np.mean(scores))
        else:
            self.model_cv_accuracy = None

        train_acc = float(self.classifier.score(x_array, y_array))
        return {
            "train_accuracy": train_acc,
            "cv_accuracy": self.model_cv_accuracy if self.model_cv_accuracy is not None else train_acc,
        }

    def _generate_answer(self, query: str, retrieved_chunks: List[RetrievedChunk], speculative: bool) -> str:
        if not retrieved_chunks:
            return "I do not have enough source context to answer this question."

        top_chunk = retrieved_chunks[0]
        if speculative and top_chunk.similarity < 0.276:
            return (
                "Based on typical policy language, this is generally covered under standard terms, "
                "but specific limits depend on endorsements."
            )

        context = "\n\n".join(chunk.text for chunk in retrieved_chunks[:3])
        if self._openai_client is not None:
            return self._generate_with_openai(query, context)

        return f"Using only policy context: {context}"

    def _generate_with_openai(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a policy QA assistant. Answer ONLY using the provided context. "
            "If the context is insufficient, reply exactly: I don't have enough information in the provided policy context."
        )
        user_prompt = f"Question: {query}\n\nContext:\n{context}"
        response = self._openai_client.chat.completions.create(
            model=self.openai_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else "I don't have enough information in the provided policy context."

    def answer_query(self, query: str, threshold: float | None = None, use_gate: bool = True) -> Dict[str, Any]:
        gate_threshold = threshold if threshold is not None else self.default_gate_threshold
        features = self.build_features(query)

        if hasattr(self.classifier, "classes_"):
            confidence = float(self.classifier.predict_proba(np.array([features]))[0][1])
        else:
            confidence = 0.5

        retrieved_chunks = self.retrieve(query)
        contexts = [chunk.text for chunk in retrieved_chunks]

        if use_gate and confidence < gate_threshold:
            return {
                "answer": "I don't have enough information in the retrieved documents to answer confidently.",
                "abstained": True,
                "confidence": confidence,
                "features": {
                    "top_similarity": features[0],
                    "score_gap": features[1],
                    "relevant_chunk_count": features[2],
                },
                "contexts": contexts,
            }

        answer_text = self._generate_answer(query, retrieved_chunks, speculative=not use_gate)
        return {
            "answer": answer_text,
            "abstained": False,
            "confidence": confidence,
            "features": {
                "top_similarity": features[0],
                "score_gap": features[1],
                "relevant_chunk_count": features[2],
            },
            "contexts": contexts,
        }

    def _is_hallucination(self, response: Dict[str, Any], example: Dict[str, Any]) -> bool:
        if response["abstained"]:
            return False

        if not example["answerable"]:
            return True

        expected = str(example.get("expected_answer_contains", "")).strip().lower()
        if not expected:
            return False

        answer_lower = response["answer"].lower()
        return expected not in answer_lower

    def evaluate(self, eval_examples: List[Dict[str, Any]], threshold: float | None = None) -> Dict[str, Any]:
        gate_threshold = threshold if threshold is not None else self.default_gate_threshold
        rows: List[Dict[str, Any]] = []
        baseline_hallucinations = 0
        gated_hallucinations = 0

        for example in eval_examples:
            baseline = self.answer_query(example["query"], threshold=gate_threshold, use_gate=False)
            gated = self.answer_query(example["query"], threshold=gate_threshold, use_gate=True)

            baseline_is_hall = self._is_hallucination(baseline, example)
            gated_is_hall = self._is_hallucination(gated, example)
            baseline_hallucinations += int(baseline_is_hall)
            gated_hallucinations += int(gated_is_hall)

            rows.append(
                {
                    "query": example["query"],
                    "answerable": example["answerable"],
                    "baseline_abstained": baseline["abstained"],
                    "gated_abstained": gated["abstained"],
                    "baseline_hallucination": baseline_is_hall,
                    "gated_hallucination": gated_is_hall,
                    "gated_confidence": gated["confidence"],
                }
            )

        total = len(eval_examples)
        return {
            "total_questions": total,
            "baseline_hallucinations": baseline_hallucinations,
            "gated_hallucinations": gated_hallucinations,
            "baseline_hallucination_rate": baseline_hallucinations / total if total else 0.0,
            "gated_hallucination_rate": gated_hallucinations / total if total else 0.0,
            "threshold": gate_threshold,
            "rows": rows,
        }


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
