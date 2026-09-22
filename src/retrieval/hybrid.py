from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists,
    k=60,
):
    """
    Reciprocal Rank Fusion.

    ranked_lists:
        list of ranked job-ID lists

    RRF score:
        sum(1 / (k + rank))
    """

    scores = defaultdict(float)

    for ranked_list in ranked_lists:

        for rank, job_id in enumerate(
            ranked_list,
            start=1
        ):
            scores[job_id] += (
                1.0 / (k + rank)
            )

    ranked_jobs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        job_id
        for job_id, _ in ranked_jobs
    ]