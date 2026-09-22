import pickle
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def tokenize(text):
    return str(text).lower().split()


def main():

    train = pd.read_parquet(
        DATA / "train_pairs.parquet"
    )

    with open(
        MODELS / "bm25_index.pkl",
        "rb"
    ) as f:
        index = pickle.load(f)

    bm25 = index["bm25"]
    job_ids = index["job_ids"]

    # SELECT ONE REAL CANDIDATE
    
    candidate_id = train.iloc[0]["profile_id"]

    candidate_rows = train[
        train["profile_id"] == candidate_id
    ]

    candidate_text = candidate_rows.iloc[0][
        "candidate_text"
    ]

    # RETRIEVE
    
    query_tokens = tokenize(candidate_text)

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:10]

    print("=" * 70)
    print("BM25 RETRIEVAL TEST")
    print("=" * 70)

    print("Candidate:", candidate_id)

    print("\nTop 10 jobs:")

    for rank, idx in enumerate(top_indices, start=1):

        print(
            f"{rank:02d}. "
            f"{job_ids[idx]} "
            f"score={scores[idx]:.4f}"
        )


if __name__ == "__main__":
    main()