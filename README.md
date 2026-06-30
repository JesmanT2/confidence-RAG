# Confidence-Gated RAG

Retrieval-augmented Q&A system over insurance policy documents that predicts retrieval confidence and abstains when evidence is weak, reducing hallucinations in unanswerable or low-support cases.

## Architecture

1. Document ingestion and chunking
   - Source docs in `data/documents.json`
   - Chunking via `RecursiveCharacterTextSplitter` from LangChain
2. Retrieval
   - ChromaDB vector store (persistent local directory: `.chroma`)
   - Custom deterministic hashing-based embedding function for local reproducibility
3. Confidence classifier
   - scikit-learn `LogisticRegression`
   - Features: top similarity, score gap, relevant chunk count
4. Gated generation behavior
   - Low confidence: abstain with "I don't have enough information..."
   - High confidence: answer only from retrieved context
5. Evaluation harness
   - 30-question set in `data/eval_questions.json` (20 answerable, 10 unanswerable)
   - Compare baseline (always answer) vs gated pipeline

## Project structure

- `app/rag_service.py`: retrieval, features, confidence model, gating, eval metrics
- `app/pipeline.py`: service builder and eval runner
- `main.py`: FastAPI app (`/health`, `/ask`, `/ask/baseline`, `/eval`)
- `data/*.json`: documents, gate training data, eval set
- `eval_sample.py`: local eval runner
- `deploy/aws`: ECR + EC2 deployment script and instructions
- `Dockerfile`: container build

## Local run

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python eval_sample.py
uvicorn main:app --reload
```

## API endpoints

- `GET /health` health check
- `GET /ask?query=...&threshold=0.552` gated answer for browser testing
- `POST /ask` gated answer with JSON body: `{"query": "...", "threshold": 0.552}`
- `GET /ask/baseline?query=...` ungated baseline response
- `GET /eval?threshold=0.552` run full benchmark

## Docker

```bash
docker build -t confidence-rag:latest .
docker run --rm -p 8000:8000 confidence-rag:latest
```

## AWS deployment

See `deploy/aws/README.md` for ECR push and EC2 run instructions.

## Notes

- The current generator is deterministic and context-bounded to keep behavior testable and reproducible.
- You can swap in a hosted LLM behind the confidence gate while keeping the same retrieval features and abstention policy.
