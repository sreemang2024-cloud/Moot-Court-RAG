import json
from pathlib import Path
from typing import Dict, Iterable, List

from pypdf import PdfReader
from tqdm import tqdm

from src.chunking import chunk_text, normalize_text
from src.config import settings

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def iter_documents(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield path


def parse_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return read_text(path)


def build_chunks(paths: Iterable[Path]) -> List[Dict]:
    records: List[Dict] = []

    for doc_path in tqdm(list(paths), desc="Ingesting legal docs"):
        text = normalize_text(parse_document(doc_path))
        if not text:
            continue

        chunks = chunk_text(
            text=text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        for idx, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_id": f"{doc_path.stem}-{idx}",
                    "source": doc_path.name,
                    "source_path": str(doc_path),
                    "text": chunk,
                }
            )

    return records


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    settings.processed_path.mkdir(parents=True, exist_ok=True)
    source_docs = list(iter_documents(settings.raw_docs_path))

    if not source_docs:
        print(f"No documents found in {settings.raw_docs_path}")
        return

    records = build_chunks(source_docs)
    output = settings.processed_path / "chunks.jsonl"
    save_jsonl(output, records)
    print(f"Wrote {len(records)} chunks to {output}")


if __name__ == "__main__":
    main()
