import pickle
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def tokenize(text):
    return str(text).lower().split()


def main():

    train = pd.read_parquet(
        DATA / "train_pairs.parquet"
    )

    gold = pd.read_parquet(
        DATA / "gold_pairs.parquet"
    )

   # BUILD UNIQUE JOB CORPUS
   
    all_data = pd.concat(
        [
            train[
                [
                    "job_id",
                    "job_text",
                    "job_title",
                    "job_location",
                    "job_seniority",
                    "job_role_family",
                    "job_domain",
                    "job_experience",
                ]
            ],
            gold[
                [
                    "job_id",
                    "job_text",
                    "job_title",
                    "job_location",
                    "job_seniority",
                    "job_role_family",
                    "job_domain",
                    "job_experience",
                ]
            ],
        ],
        ignore_index=True,
    )

    jobs = (
        all_data
        .drop_duplicates("job_id")
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("BUILDING BM25 INDEX")
    print("=" * 70)

    print("Unique jobs:", len(jobs))

    corpus = [
        tokenize(text)
        for text in jobs["job_text"]
    ]

    bm25 = BM25Okapi(corpus)

    index = {
        "job_ids": jobs["job_id"].tolist(),
        "job_texts": jobs["job_text"].tolist(),
        "bm25": bm25,
    }

    output = MODELS / "bm25_index.pkl"

    with open(output, "wb") as f:
        pickle.dump(index, f)

    print("BM25 index saved to:")
    print(output)


if __name__ == "__main__":
    main()
    