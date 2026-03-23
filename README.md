# Moot Court Legal RAG (India-focused)

This project builds a Retrieval-Augmented Generation (RAG) system for legal Q&A with **citation-first answers** and **abstention when evidence is weak**.

## What this gives you

- Ingestion pipeline for `.pdf`, `.txt`, `.md` legal docs.
- Chunking and vector index (FAISS + Sentence Transformers).
- Hybrid retrieval (dense + BM25 via reciprocal rank fusion).
- Grounded answer generation with strict prompt rules.
- Hallucination controls:
  - answer only from retrieved context,
  - mandatory citations,
  - abstain when confidence is low.
- FastAPI service for asking questions.
- Evaluation script for faithfulness/citation checks.

## Important reality check

No LLM can guarantee *zero* hallucinations. This setup minimizes risk by enforcing retrieval grounding and refusal behavior. For production legal usage, keep a human-in-the-loop review.

## 1) Setup

```bash
cd /home/student/Documents/damo
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env.example .env
```

Local free-model mode is enabled by default (`USE_OPENAI=false`).

Optional: if you ever want to use OpenAI, set in `.env`:

```env
USE_OPENAI=true
OPENAI_API_KEY=your_key_here
```

## 2) Put legal corpus files

Place Indian judgments, statutes, and case notes in:

- `data/raw_docs/`

Suggested naming style:

- `SC_2018_ABC_vs_XYZ.pdf`
- `CPC_1908_sections.txt`

## 3) Build index

```bash
python3 -m src.ingest
python3 -m src.retrieval --build
```

## 4) Run API

```bash
uvicorn src.app:app --reload --port 8000
```

## 5) Ask questions

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the essentials of adverse possession under Indian law?"}'
```

## 6) Evaluate

Create `data/eval/qa.json` with items like:

```json
[
  {
    "question": "What did the court hold on limitation in X v Y?",
    "expected_sources": ["SC_2018_ABC_vs_XYZ.pdf"],
    "answerable": true
  },
  {
    "question": "What does this corpus say about a law not in documents?",
    "expected_sources": [],
    "answerable": false
  }
]
```

Run:

```bash
python3 -m src.eval
```

## Production hardening ideas

- Add re-ranker (cross-encoder) before final context selection.
- Add citation verifier that checks each claim against source spans.
- Version datasets and indexes (DVC/LakeFS/S3 manifest).
- Use court/official gazette sources only for trusted corpus quality.
- Add role-based audit logs for all Q&A.

## Legal and compliance notes

- Use only documents you are licensed to store/process.
- Keep PII-sensitive case material encrypted at rest.
- This is research tooling, not legal advice.
