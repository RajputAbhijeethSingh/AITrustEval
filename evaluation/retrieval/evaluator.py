import pandas as pd

from metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg,
    hit_rate,
)


def evaluate_retrieval(relevance_lists, k=5):
    """
    Evaluate a collection of ranked retrieval results.

    Parameters
    ----------
    relevance_lists : list[list[int]]
        Each inner list represents the relevance of retrieved
        documents for one query.

        Example:
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0]
        ]

    k : int
        Number of top retrieved documents to evaluate.

    Returns
    -------
    dict
        Aggregated retrieval evaluation metrics.
    """

    if not relevance_lists:
        raise ValueError("relevance_lists cannot be empty.")

    precisions = []
    recalls = []
    reciprocal_ranks = []
    ndcgs = []
    hits = []

    for relevance in relevance_lists:

        total_relevant = sum(relevance)

        precisions.append(
            precision_at_k(relevance, k)
        )

        recalls.append(
            recall_at_k(
                relevance,
                total_relevant,
                k
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(relevance)
        )

        ndcgs.append(
            ndcg(relevance, k)
        )

        hits.append(
            1 if any(relevance[:k]) else 0
        )

    results = {
        f"precision@{k}": sum(precisions) / len(precisions),
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
        f"hit_rate@{k}": sum(hits) / len(hits),
    }

    return results


if __name__ == "__main__":

    print("========== RETRIEVAL EVALUATOR TEST ==========")

    sample_relevance = [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]

    results = evaluate_retrieval(
        sample_relevance,
        k=5
    )

    print("\nRetrieval Evaluation Results:")

    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    print("\n========== TEST COMPLETE ==========")