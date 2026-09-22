import pickle
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.evaluation.retrieval_metrics import (
    hit_at_k,
    reciprocal_rank,
    ndcg_at_k,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text):
    return str(text).lower().split()


def load_gold():

    gold = pd.read_parquet(
        DATA / "gold_pairs.parquet"
    )

    return gold


def load_job_corpus():

    train = pd.read_parquet(
        DATA / "train_pairs.parquet"
    )

    gold = pd.read_parquet(
        DATA / "gold_pairs.parquet"
    )

    all_data = pd.concat(
        [
            train,
            gold
        ],
        ignore_index=True
    )

    jobs = (
        all_data[
            [
                "job_id",
                "job_text"
            ]
        ]
        .drop_duplicates("job_id")
        .reset_index(drop=True)
    )

    return jobs


def build_bm25(jobs):

    corpus = [
        tokenize(text)
        for text in jobs["job_text"]
    ]

    return BM25Okapi(corpus)


def load_dense():

    index = faiss.read_index(
        str(
            MODELS /
            "dense_jobs.faiss"
        )
    )

    with open(
        MODELS /
        "dense_jobs_metadata.pkl",
        "rb"
    ) as f:

        metadata = pickle.load(f)

    return index, metadata


def evaluate_system(
    system_name,
    results,
    gold
):

    metrics = []

    # Group gold judgments by candidate
    grouped = gold.groupby(
        "profile_id"
    )

    for profile_id, group in grouped:

        gold_relevance = {
            row.job_id: row.label
            for _, row in group.iterrows()
        }

        retrieved = results.get(
            profile_id,
            []
        )

        ranks = []

        relevances = []

        for rank, job_id in enumerate(
            retrieved,
            start=1
        ):

            if job_id in gold_relevance:

                ranks.append(rank)

            relevances.append(
                gold_relevance.get(
                    job_id,
                    0
                )
            )

        metrics.append(
            {
                "profile_id": profile_id,

                "hit@5": hit_at_k(
                    ranks,
                    5
                ),

                "hit@10": hit_at_k(
                    ranks,
                    10
                ),

                "rr": reciprocal_rank(
                    ranks
                ),

                "ndcg@5": ndcg_at_k(
                    relevances,
                    5
                ),

                "ndcg@10": ndcg_at_k(
                    relevances,
                    10
                ),
            }
        )

    metrics_df = pd.DataFrame(metrics)

    summary = {
        "system": system_name,
        "candidates_evaluated": len(metrics_df),
        "hit@5": metrics_df["hit@5"].mean(),
        "hit@10": metrics_df["hit@10"].mean(),
        "MRR": metrics_df["rr"].mean(),
        "nDCG@5": metrics_df["ndcg@5"].mean(),
        "nDCG@10": metrics_df["ndcg@10"].mean(),
    }

    return summary, metrics_df


def main():

    print("=" * 70)
    print("TALENTGRAPH RETRIEVAL EVALUATION")
    print("=" * 70)

    gold = load_gold()
    jobs = load_job_corpus()
    job_ids = jobs["job_id"].tolist()

    print("Gold pairs:", len(gold))
    print("Gold candidates:", gold["profile_id"].nunique())
    print("Job corpus:", len(jobs))

    # BUILD CANDIDATE QUERIES

    candidate_texts = (
        gold[
            [
                "profile_id",
                "candidate_text"
            ]
        ]
        .drop_duplicates("profile_id")
        .set_index("profile_id")
        ["candidate_text"]
        .to_dict()
    )

    # BM25
    print("\nEvaluating BM25...")

    bm25 = build_bm25(jobs)

    bm25_results = {}

    for profile_id, candidate_text in candidate_texts.items():

        scores = bm25.get_scores(
            tokenize(candidate_text)
        )

        top_indices = np.argsort(
            scores
        )[::-1][:50]

        bm25_results[profile_id] = [
            job_ids[idx]
            for idx in top_indices
        ]

    bm25_summary, _ = evaluate_system(
        "BM25",
        bm25_results,
        gold
    )

    # DENSE

    print("Evaluating Dense MiniLM...")

    index, metadata = load_dense()

    model = SentenceTransformer(
        MODEL_NAME
    )

    dense_results = {}

    embeddings = model.encode(
        list(candidate_texts.values()),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    profile_list = list(
        candidate_texts.keys()
    )

    scores, indices = index.search(
        embeddings.astype(np.float32),
        50
    )

    for row_idx, profile_id in enumerate(
        profile_list
    ):

        dense_results[profile_id] = [
            metadata["job_ids"][idx]
            for idx in indices[row_idx]
        ]

    dense_summary, _ = evaluate_system(
        "Dense-MiniLM",
        dense_results,
        gold
    )

    results = pd.DataFrame(
        [
            bm25_summary,
            dense_summary,
        ]
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    output = RESULTS / "retrieval_baselines.csv"

    results.to_csv(
        output,
        index=False
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()