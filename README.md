# Confidence-Gated RAG

This project demonstrates a retrieval-augmented QA pipeline that abstains when the retrieved context is too weak to support a grounded answer.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API:
   ```bash
   uvicorn main:app --reload
   ```
4. Run the test suite:
   ```bash
   pytest -q
   ```

## How it works

- Ingests documents into a lightweight in-memory retrieval layer.
- Computes retrieval features such as top similarity, score gap, and chunk count.
- Trains a logistic regression gate to predict whether a query is answerable from the retrieved context.
- Abstains when confidence is below a threshold.

## Evaluation idea

A small 30-question eval set can be used to compare hallucination rates before and after the gate. The headline result is intended to show a meaningful reduction in unsupported answers.
