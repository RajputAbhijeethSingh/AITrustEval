import math


def hit_rate(retrieved_relevant):
    """
    Hit Rate:
    Proportion of queries for which at least one relevant
    document was retrieved.
    """

    if not retrieved_relevant:
        return 0.0

    hits = sum(1 for value in retrieved_relevant if value)
    return hits / len(retrieved_relevant)


def precision_at_k(relevance, k):
    """
    Precision@K:
    Number of relevant documents in the top K
    divided by K.
    """

    if k <= 0:
        return 0.0

    top_k = relevance[:k]

    if not top_k:
        return 0.0

    return sum(top_k) / len(top_k)


def recall_at_k(relevance, total_relevant, k):
    """
    Recall@K:
    Relevant documents retrieved in top K
    divided by total number of relevant documents.
    """

    if total_relevant <= 0 or k <= 0:
        return 0.0

    top_k = relevance[:k]

    return sum(top_k) / total_relevant


def reciprocal_rank(relevance):
    """
    Reciprocal Rank:
    1 / rank of the first relevant document.
    Returns 0 if no relevant document is found.
    """

    for index, relevant in enumerate(relevance, start=1):

        if relevant:
            return 1.0 / index

    return 0.0


def mean_reciprocal_rank(all_relevance):
    """
    Mean Reciprocal Rank across multiple queries.
    """

    if not all_relevance:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(relevance)
        for relevance in all_relevance
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def dcg(relevance, k):
    """
    Discounted Cumulative Gain.
    """

    top_k = relevance[:k]

    score = 0.0

    for rank, relevance_score in enumerate(top_k, start=1):

        score += relevance_score / math.log2(rank + 1)

    return score


def ndcg(relevance, k):
    """
    Normalized Discounted Cumulative Gain.
    """

    actual_dcg = dcg(relevance, k)

    ideal_relevance = sorted(
        relevance,
        reverse=True
    )

    ideal_dcg = dcg(ideal_relevance, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


if __name__ == "__main__":

    print("========== RETRIEVAL METRIC TEST ==========")

    # Example:
    # 1 = relevant document
    # 0 = irrelevant document

    relevance = [1, 0, 1, 0, 0]

    print(f"Relevance list: {relevance}")

    print(
        f"Precision@3: "
        f"{precision_at_k(relevance, 3):.2f}"
    )

    print(
        f"Recall@3: "
        f"{recall_at_k(relevance, 2, 3):.2f}"
    )

    print(
        f"Reciprocal Rank: "
        f"{reciprocal_rank(relevance):.2f}"
    )

    print(
        f"NDCG@5: "
        f"{ndcg(relevance, 5):.2f}"
    )

    print(
        f"Hit Rate: "
        f"{hit_rate([True, True, False, True]):.2f}"
    )

    print("\n========== TEST COMPLETE ==========")
