import re
from functools import lru_cache
from typing import List

from openai import OpenAI
from transformers import pipeline

from src.config import settings
from src.retrieval import RetrievedDoc


ABSTAIN_ANSWER = (
    "I do not have enough reliable context in the indexed legal documents to answer this question. "
    "Please add the relevant case law/statutory material to the corpus and retry."
)


def _format_context(contexts: List[RetrievedDoc]) -> str:
    blocks = []
    for idx, c in enumerate(contexts, start=1):
        blocks.append(
            f"[C{idx}] source={c.source} chunk_id={c.chunk_id}\n{c.text}"
        )
    return "\n\n".join(blocks)


def _extract_citations(answer: str) -> List[str]:
    return sorted(set(re.findall(r"\[(?:C\d+)\]", answer)))


@lru_cache(maxsize=1)
def _get_local_generator():
    return pipeline(
        task="text2text-generation",
        model=settings.local_llm_model,
        device=-1,
    )


def _extractive_fallback(question: str, contexts: List[RetrievedDoc]) -> tuple[str, List[str]]:
    query_terms = set(re.findall(r"[A-Za-z0-9_]+", question.lower()))
    scored_sentences: List[tuple[float, int, str]] = []

    for idx, ctx in enumerate(contexts, start=1):
        sentences = re.split(r"(?<=[.!?])\s+", ctx.text)
        for sentence in sentences:
            sent = sentence.strip()
            if len(sent) < 40:
                continue
            sent_terms = set(re.findall(r"[A-Za-z0-9_]+", sent.lower()))
            overlap = len(query_terms.intersection(sent_terms))
            if overlap == 0:
                continue
            score = overlap / max(1, len(query_terms))
            scored_sentences.append((score, idx, sent))

    if not scored_sentences:
        return ABSTAIN_ANSWER, []

    top = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:3]
    lines = [f"- {sent} [C{idx}]" for _, idx, sent in top]
    citations = sorted(set(f"[C{idx}]" for _, idx, _ in top))
    answer = "\n".join(lines)
    return answer, citations


def _generate_local_answer(question: str, contexts: List[RetrievedDoc]) -> tuple[str, List[str]]:
    prompt = (
        "You are a legal research assistant for Indian moot-court preparation. "
        "Answer using ONLY the given context. Do not invent any legal facts. "
        "Cite each key claim with tags like [C1], [C2]. "
        "If context is insufficient, say: I do not have enough context.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{_format_context(contexts)}\n\n"
        "Answer:"
    )

    try:
        generator = _get_local_generator()
        outputs = generator(prompt, max_new_tokens=settings.local_max_new_tokens)
        answer = (outputs[0].get("generated_text") or "").strip()
    except Exception:
        answer = ""

    citations = _extract_citations(answer)
    if citations:
        return answer, citations

    return _extractive_fallback(question, contexts)


def generate_grounded_answer(question: str, contexts: List[RetrievedDoc]) -> tuple[str, List[str]]:
    if not contexts:
        return ABSTAIN_ANSWER, []

    if not settings.use_openai:
        return _generate_local_answer(question, contexts)

    if not settings.openai_api_key:
        return _generate_local_answer(question, contexts)

    client = OpenAI(api_key=settings.openai_api_key)

    system_prompt = (
        "You are a legal research assistant for Indian moot-court preparation. "
        "Answer strictly from provided context only. "
        "If the answer is not fully supported, say you do not have enough context. "
        "Do not invent precedents, sections, or holdings. "
        "Every substantive claim must cite one or more context blocks as [C1], [C2], etc."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{_format_context(contexts)}\n\n"
        "Return a concise answer with bullet points where useful, and include citations."
    )

    response = client.responses.create(
        model=settings.llm_model,
        temperature=settings.temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.output_text.strip()
    citations = _extract_citations(answer)
    return answer, citations
