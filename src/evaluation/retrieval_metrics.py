import math


def dcg_at_k(relevances, k):
    """
    Discounted cumulative gain.
    """

    relevances = relevances[:k]

    score = 0.0

    for rank, relevance in enumerate(
        relevances,
        start=1
    ):
        score += (
            (2 ** relevance - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(relevances, k):
    """
    Normalized discounted cumulative gain.
    """

    actual = dcg_at_k(
        relevances,
        k
    )

    ideal = dcg_at_k(
        sorted(
            relevances,
            reverse=True
        ),
        k
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


def reciprocal_rank(ranks):

    if not ranks:
        return 0.0

    return 1.0 / min(ranks)


def hit_at_k(ranks, k):

    return 1.0 if any(
        rank <= k
        for rank in ranks
    ) else 0.0