from app.rag_service import ConfidenceGatedRAG


def test_gate_abstains_when_context_is_insufficient():
    service = ConfidenceGatedRAG()
    service.ingest_documents([
        {"id": "doc-1", "text": "The deductible for comprehensive claims is $500."},
        {"id": "doc-2", "text": "Flood coverage has a 30-day waiting period."},
    ])
    training_examples = [
        {"query": "What is the deductible for comprehensive claims?", "answerable": True},
        {"query": "How long is the waiting period for flood coverage?", "answerable": True},
        {"query": "What is the jewelry limit?", "answerable": False},
    ]
    service.fit_gate(training_examples)

    result = service.answer_query("What is the jewelry limit?", threshold=0.6)

    assert result["abstained"] is True
    assert "don't have enough information" in result["answer"].lower()
