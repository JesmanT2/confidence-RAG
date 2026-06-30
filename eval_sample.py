from app.rag_service import ConfidenceGatedRAG

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

examples = [
    {"query": "What is the deductible for comprehensive claims?", "answerable": True},
    {"query": "How long is the waiting period for flood coverage?", "answerable": True},
    {"query": "What is the jewelry limit?", "answerable": False},
]
print(service.evaluate(examples))
