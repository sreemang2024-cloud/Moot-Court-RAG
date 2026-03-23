import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import settings


@dataclass
class RetrievedDoc:
    chunk_id: str
    source: str
    text: str
    score: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def load_chunks(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_index() -> None:
    chunks_path = settings.processed_path / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found. Run `python -m src.ingest` first."
        )

    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(settings.embed_model)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    settings.index_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.index_path / "dense.faiss"))

    with (settings.index_path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Indexed {len(chunks)} chunks into {settings.index_path}")


class HybridRetriever:
    def __init__(self) -> None:
        chunks_path = settings.index_path / "chunks.jsonl"
        index_path = settings.index_path / "dense.faiss"

        if not chunks_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                "Index not found. Run `python -m src.retrieval --build` first."
            )

        self.chunks = load_chunks(chunks_path)
        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(settings.embed_model)

        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, question: str, top_k: int | None = None) -> List[RetrievedDoc]:
        k = top_k or settings.top_k
        n = min(len(self.chunks), max(k * 4, 20))

        q_emb = self.model.encode([question], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        dense_scores, dense_idx = self.index.search(q_emb, n)

        dense_rank = [int(i) for i in dense_idx[0] if i >= 0]
        dense_score_map = {int(i): float(s) for s, i in zip(dense_scores[0], dense_idx[0]) if i >= 0}

        bm25_scores = self.bm25.get_scores(_tokenize(question))
        bm25_rank = np.argsort(-bm25_scores)[:n].tolist()

        rrf_k = 60
        fused: Dict[int, float] = {}

        for rank, idx in enumerate(dense_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        for rank, idx in enumerate(bm25_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

        docs: List[RetrievedDoc] = []
        for idx, rrf_score in ranked:
            row = self.chunks[idx]
            dense_score = dense_score_map.get(idx, 0.0)
            score = max(rrf_score, dense_score)
            docs.append(
                RetrievedDoc(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    text=row["text"],
                    score=float(score),
                )
            )

        return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build retrieval index")
    args = parser.parse_args()

    if args.build:
        build_index()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
