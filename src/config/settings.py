from pathlib import Path

# Project root:
# TalentGraph/
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

RANDOM_SEED = 42

TOP_K = 10

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"