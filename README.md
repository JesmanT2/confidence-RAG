# Confidence-Gated RAG

Retrieval-augmented Q&A system over insurance policy documents that predicts retrieval confidence and abstains when evidence is weak, reducing hallucinations in unanswerable or low-support cases.

## Architecture

1. Document ingestion and chunking
   - Source docs in `data/documents.json`
   - Chunking via `RecursiveCharacterTextSplitter` from LangChain
2. Retrieval
   - ChromaDB vector store (persistent local directory: `.chroma`)
   - Semantic embeddings via SentenceTransformers (`all-MiniLM-L6-v2`)
3. Confidence classifier
   - scikit-learn `LogisticRegression`
   - Features: top similarity, score gap, relevant chunk count
4. Gated generation behavior
   - Low confidence: abstain with "I don't have enough information..."
   - High confidence: generate answer only from retrieved context (OpenAI optional)
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

## Optional OpenAI generation

Set environment variables before running the API:

```bash
set OPENAI_API_KEY=your_key_here
set RAG_LLM_PROVIDER=openai
```

By default, `RAG_LLM_PROVIDER=none`, which keeps generation deterministic for offline testing.

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

Current status: deployment-ready with ECR/EC2 automation scripts. If you publish a live endpoint, update this section with the production URL.

## Notes

- Retrieval supports semantic embeddings by default. For offline tests you can set `RAG_EMBEDDING_PROVIDER=hashing`.
- The OpenAI answer generator is wired behind the confidence gate when `RAG_LLM_PROVIDER=openai` and `OPENAI_API_KEY` are set.
