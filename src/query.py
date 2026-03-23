import argparse

from src.config import settings
from src.llm import ABSTAIN_ANSWER, generate_grounded_answer
from src.retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    args = parser.parse_args()

    retriever = HybridRetriever()
    docs = retriever.search(args.question, top_k=settings.top_k)
    best = max((d.score for d in docs), default=0.0)

    if best < settings.min_context_score:
        print(ABSTAIN_ANSWER)
        return

    answer, citations = generate_grounded_answer(args.question, docs)
    if not citations:
        print(ABSTAIN_ANSWER)
        return

    print(answer)


if __name__ == "__main__":
    main()
