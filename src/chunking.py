import re
from typing import List


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: List[str] = []

    for start in range(0, len(words), step):
        end = start + chunk_size
        window = words[start:end]
        if not window:
            continue
        chunks.append(" ".join(window))
        if end >= len(words):
            break

    return chunks
