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
from src.retrieval.hybrid import (
    reciprocal_rank_fusion,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text):
    return str(text).lower().split()


def load_data():

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

    return train, gold, jobs


def evaluate(
    results,
    gold,
):

    rows = []

    for profile_id, group in gold.groupby(
        "profile_id"
    ):

        relevance = {
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

            relevances.append(
                relevance.get(
                    job_id,
                    0
                )
            )

            if job_id in relevance:
                ranks.append(rank)

        rows.append(
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

                "MRR": reciprocal_rank(
                    ranks
                ),

                "nDCG@5": ndcg_at_k(
                    relevances,
                    5
                ),

                "nDCG@10": ndcg_at_k(
                    relevances,
                    10
                ),
            }
        )

    df = pd.DataFrame(rows)

    return {
        "system": "Hybrid-RRF",
        "candidates_evaluated": len(df),
        "hit@5": df["hit@5"].mean(),
        "hit@10": df["hit@10"].mean(),
        "MRR": df["MRR"].mean(),
        "nDCG@5": df["nDCG@5"].mean(),
        "nDCG@10": df["nDCG@10"].mean(),
    }


def main():

    print("=" * 70)
    print("TALENTGRAPH HYBRID RETRIEVAL")
    print("=" * 70)

    train, gold, jobs = load_data()

    job_ids = jobs["job_id"].tolist()

    # BM25
    
    corpus = [
        tokenize(text)
        for text in jobs["job_text"]
    ]

    bm25 = BM25Okapi(corpus)

    # DENSE

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

    model = SentenceTransformer(
        MODEL_NAME
    )

    # CANDIDATE QUERIES
    
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

    hybrid_results = {}

    # RETRIEVE
    
    for profile_id, candidate_text in candidate_texts.items():

        # BM25 top 100
        bm25_scores = bm25.get_scores(
            tokenize(candidate_text)
        )

        bm25_indices = np.argsort(
            bm25_scores
        )[::-1][:100]

        bm25_ranked = [
            job_ids[idx]
            for idx in bm25_indices
        ]

        # Dense top 100
        embedding = model.encode(
            [candidate_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        _, dense_indices = index.search(
            embedding,
            100
        )

        dense_ranked = [
            metadata["job_ids"][idx]
            for idx in dense_indices[0]
        ]

        # RRF
        hybrid_ranked = reciprocal_rank_fusion(
            [
                bm25_ranked,
                dense_ranked,
            ],
            k=60,
        )

        hybrid_results[profile_id] = (
            hybrid_ranked[:50]
        )

    # EVALUATE
    
    summary = evaluate(
        hybrid_results,
        gold,
    )

    print("\n" + "=" * 70)
    print("HYBRID RESULTS")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    # SAVE
    
    output = RESULTS / "hybrid_retrieval.csv"

    pd.DataFrame(
        [summary]
    ).to_csv(
        output,
        index=False
    )

    print("\nSaved:")
    print(output)

if __name__ == "__main__":
    main()
    