import pickle
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.ranking.features import build_features

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text):
    return str(text).lower().split()


def main():

    df = pd.read_parquet(
        DATA / "train_pairs.parquet"
    )

    print("=" * 70)
    print("BUILDING RANKING FEATURES")
    print("=" * 70)

    print("Training pairs:", len(df))

    # JOB CORPUS
    
    jobs = (
        df[
            [
                "job_id",
                "job_text"
            ]
        ]
        .drop_duplicates("job_id")
        .reset_index(drop=True)
    )

    job_ids = jobs["job_id"].tolist()

   # BM25
    
    print("\nBuilding BM25...")

    corpus = [
        tokenize(x)
        for x in jobs["job_text"]
    ]

    bm25 = BM25Okapi(corpus)

    job_index = {
        job_id: idx
        for idx, job_id in enumerate(job_ids)
    }

    bm25_scores = []

    # DENSE MODEL
    
    print("Loading MiniLM...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    # Encode unique candidate texts.
    candidate_df = (
        df[
            [
                "profile_id",
                "candidate_text"
            ]
        ]
        .drop_duplicates("profile_id")
    )

    candidate_ids = candidate_df[
        "profile_id"
    ].tolist()

    candidate_texts = candidate_df[
        "candidate_text"
    ].tolist()

    print("Encoding candidates...")

    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    candidate_embedding_map = {
        candidate_id: embedding
        for candidate_id, embedding
        in zip(
            candidate_ids,
            candidate_embeddings
        )
    }

    # DENSE JOB EMBEDDINGS
    
    print("Encoding jobs...")

    job_embeddings = model.encode(
        jobs["job_text"].tolist(),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    job_embedding_map = {
        job_id: embedding
        for job_id, embedding
        in zip(
            job_ids,
            job_embeddings
        )
    }

    # COMPUTE FEATURES
    
    feature_rows = []

    for row_idx, row in df.iterrows():

        if row_idx % 1000 == 0:
            print(
                f"Processing {row_idx}/{len(df)}"
            )

        candidate_embedding = (
            candidate_embedding_map[
                row["profile_id"]
            ]
        )

        job_embedding = (
            job_embedding_map[
                row["job_id"]
            ]
        )

        dense_score = float(
            np.dot(
                candidate_embedding,
                job_embedding
            )
        )

        bm25_score = float(
            bm25.get_scores(
                tokenize(
                    row["candidate_text"]
                )
            )[job_index[row["job_id"]]]
        )

        row_dict = row.to_dict()

        row_dict["bm25_score"] = bm25_score
        row_dict["dense_score"] = dense_score

        features = build_features(
            row_dict
        )

        for name, value in features.items():
            row_dict[name] = value

        feature_rows.append(row_dict)

    feature_df = pd.DataFrame(
        feature_rows
    )

    output = DATA / "ranking_train.parquet"

    feature_df.to_parquet(
        output,
        index=False
    )

    print("\n" + "=" * 70)
    print("FEATURE DATASET COMPLETE")
    print("=" * 70)

    print("Rows:", len(feature_df))
    print(
        "Candidates:",
        feature_df["profile_id"].nunique()
    )

    print("\nFeatures:")

    feature_columns = [
        "bm25_score",
        "dense_score",
        "skill_coverage",
        "role_family_match",
        "domain_match",
        "experience_compatibility",
        "seniority_compatibility",
        "location_compatibility",
    ]

    print("\nActual feature columns present in dataframe:")
    print([
        column
        for column in df.columns
        if column in feature_columns
    ])

    print(feature_columns)

    print("\nStatistics:")
    print(
        feature_df[
            feature_columns
        ].describe()
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()