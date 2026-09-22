import pickle
from pathlib import Path
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():

    train = pd.read_parquet(
        DATA / "train_pairs.parquet"
    )

    index = faiss.read_index(
        str(MODELS / "dense_jobs.faiss")
    )

    with open(
        MODELS / "dense_jobs_metadata.pkl",
        "rb"
    ) as f:
        metadata = pickle.load(f)

    model = SentenceTransformer(MODEL_NAME)

    # SELECT CANDIDATE
    
    candidate_id = train.iloc[0]["profile_id"]

    candidate_rows = train[
        train["profile_id"] == candidate_id
    ]

    candidate_text = candidate_rows.iloc[0][
        "candidate_text"
    ]

    # EMBED QUERY
   
    query_embedding = model.encode(
        [candidate_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # SEARCH
    
    scores, indices = index.search(
        query_embedding,
        10
    )

    print("=" * 70)
    print("DENSE RETRIEVAL TEST")
    print("=" * 70)

    print("Candidate:", candidate_id)

    print("\nTop 10 jobs:")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"{metadata['job_ids'][idx]} "
            f"score={score:.4f}"
        )


if __name__ == "__main__":
    main()