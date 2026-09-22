import pickle
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

MODELS.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    train = pd.read_parquet(
        DATA / "train_pairs.parquet"
    )

    gold = pd.read_parquet(
        DATA / "gold_pairs.parquet"
    )

    # UNIQUE JOB CORPUS
   
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
    print("BUILDING DENSE FAISS INDEX")
    print("=" * 70)

    print("Unique jobs:", len(jobs))
    print("Embedding model:", MODEL_NAME)

    # LOAD MODEL
   
    model = SentenceTransformer(MODEL_NAME)

   # EMBED JOBS
   
    print("\nEncoding jobs...")

    embeddings = model.encode(
        jobs["job_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    print("Embedding shape:", embeddings.shape)

    # FAISS INDEX
    
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS vectors:", index.ntotal)
    print("Embedding dimension:", dimension)

   # SAVE 
    faiss_path = MODELS / "dense_jobs.faiss"
    metadata_path = MODELS / "dense_jobs_metadata.pkl"

    faiss.write_index(
        index,
        str(faiss_path)
    )

    metadata = {
        "job_ids": jobs["job_id"].tolist(),
        "job_texts": jobs["job_text"].tolist(),
        "model_name": MODEL_NAME,
    }

    with open(
        metadata_path,
        "wb"
    ) as f:
        pickle.dump(metadata, f)

    print("\nSaved:")
    print(faiss_path)
    print(metadata_path)


if __name__ == "__main__":
    main()