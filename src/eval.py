import json
from pathlib import Path
from typing import Any, Dict, List

from src.config import settings
from src.llm import ABSTAIN_ANSWER, generate_grounded_answer
from src.retrieval import HybridRetriever


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate() -> None:
    eval_path = settings.data_path / "eval" / "qa.json"
    if not eval_path.exists():
        print(f"Missing eval set: {eval_path}")
        return

    retriever = HybridRetriever()
    qa_items = load_eval_set(eval_path)

    total = len(qa_items)
    answered = 0
    abstained = 0
    citation_present = 0

    for item in qa_items:
        question = item["question"]
        docs = retriever.search(question, top_k=settings.top_k)
        best = max((d.score for d in docs), default=0.0)

        if best < settings.min_context_score:
            answer = ABSTAIN_ANSWER
            citations: List[str] = []
        else:
            answer, citations = generate_grounded_answer(question, docs)

        is_abstain = answer.strip() == ABSTAIN_ANSWER
        if is_abstain:
            abstained += 1
        else:
            answered += 1

        if citations:
            citation_present += 1

    print("Evaluation Summary")
    print(f"Total questions: {total}")
    print(f"Answered: {answered}")
    print(f"Abstained: {abstained}")
    print(f"Answers with citations: {citation_present}")
    if total:
        print(f"Citation rate: {citation_present / total:.2%}")
        print(f"Abstain rate: {abstained / total:.2%}")


if __name__ == "__main__":
    evaluate()
