"""Fair sparse-judged gold evaluation for all TalentGraph rankers.

Gold labels are used only after scoring. Unjudged jobs receive zero gain for
this sparse evaluation; this does not assert that they are irrelevant.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.data.contracts import load_candidates, load_jobs, scoring_row
from src.ranking.features import build_features
from src.retrieval.hybrid import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    return str(text).lower().split()


def relevance_grade(score: float) -> int:
    return 0 if score < 60 else 1 if score < 70 else 2 if score < 80 else 3 if score < 90 else 4


def sparse_metrics(ranked_job_ids: list[str], gold_scores: dict[str, float]) -> dict[str, float]:
    labels = np.array([relevance_grade(gold_scores.get(job_id, 0)) for job_id in ranked_job_ids])
    metrics: dict[str, float] = {}
    for k in (5, 10):
        top_labels = labels[:k]
        dcg = float(np.sum(((2**top_labels) - 1) / np.log2(np.arange(2, len(top_labels) + 2))))
        ideal = np.sort([relevance_grade(value) for value in gold_scores.values()])[::-1][:k]
        idcg = float(np.sum(((2 ** np.asarray(ideal)) - 1) / np.log2(np.arange(2, len(ideal) + 2))))
        metrics[f"hit@{k}"] = float(np.any(top_labels > 0))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    positive = np.flatnonzero(labels > 0)
    metrics["mrr"] = float(1 / (positive[0] + 1)) if len(positive) else 0.0
    return metrics


def rank_from_scores(job_ids: list[str], scores: np.ndarray) -> list[str]:
    return [job_ids[index] for index in np.argsort(scores)[::-1]]


def load_evaluation_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_DIR / "train_pairs.parquet")
    gold = pd.read_parquet(DATA_DIR / "gold_pairs.parquet")
    if set(train.profile_id) & set(gold.profile_id):
        raise ValueError("Gold candidates overlap training candidates; refusing to evaluate leaked data.")
    union_ids = set(pd.concat([train.job_id, gold.job_id]).unique())
    jobs = load_jobs().query("job_id in @union_ids").copy().reset_index(drop=True)
    missing_jobs = union_ids - set(jobs.job_id)
    if missing_jobs:
        raise ValueError(f"Raw job metadata is missing {len(missing_jobs)} union jobs.")
    gold_candidate_ids = set(gold.profile_id)
    candidates = load_candidates().query("profile_id in @gold_candidate_ids").copy()
    missing_candidates = set(gold.profile_id) - set(candidates.profile_id)
    if missing_candidates:
        raise ValueError(f"Raw candidate metadata is missing gold profiles: {sorted(missing_candidates)}")
    return train, gold, jobs, candidates


def build_feature_frame(candidate: dict, jobs: pd.DataFrame, bm25_scores: np.ndarray, dense_scores: np.ndarray) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    rows = []
    for index, job in enumerate(jobs.to_dict("records")):
        row = scoring_row(candidate, job, bm25_scores[index], dense_scores[index])
        rows.append({**row, **build_features(row)})
    return pd.DataFrame(rows), time.perf_counter() - start


def summarise(method: str, metrics: list[dict[str, float]], latencies_ms: list[float], jobs: int) -> dict:
    return {
        "method": method,
        "hit@5": float(np.mean([row["hit@5"] for row in metrics])),
        "hit@10": float(np.mean([row["hit@10"] for row in metrics])),
        "mrr": float(np.mean([row["mrr"] for row in metrics])),
        "ndcg@5": float(np.mean([row["ndcg@5"] for row in metrics])),
        "ndcg@10": float(np.mean([row["ndcg@10"] for row in metrics])),
        "num_candidates": len(metrics), "num_jobs": jobs,
        "evaluation_type": "sparse judged gold evaluation",
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "median_latency_ms": float(np.median(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    train, gold, jobs, candidates = load_evaluation_data()
    print(f"Training jobs: {train.job_id.nunique()}\nGold jobs: {gold.job_id.nunique()}\nUnion jobs: {len(jobs)}")
    print(f"Gold candidates: {gold.profile_id.nunique()}\nGold judgments: {len(gold)}")
    job_ids = jobs.job_id.tolist()
    bm25 = BM25Okapi([tokenize(text) for text in jobs.job_text])
    encoder = SentenceTransformer(MODEL_NAME)
    job_embeddings = encoder.encode(jobs.job_text.tolist(), batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    bundles = {
        "XGBoost-LTR": joblib.load(MODELS_DIR / "talentgraph_ltr.joblib"),
        "XGBoost-LTR-no-location": joblib.load(MODELS_DIR / "talentgraph_ltr_no_location.joblib"),
    }
    methods = {name: {"metrics": [], "latencies": []} for name in ["BM25", "Dense-MiniLM", "RRF", *bundles]}
    rankings = []
    gold_by_candidate = gold.groupby("profile_id")
    for number, candidate in enumerate(candidates.to_dict("records"), start=1):
        candidate_id = candidate["profile_id"]
        gold_scores = gold_by_candidate.get_group(candidate_id).set_index("job_id")["label"].to_dict()
        start = time.perf_counter(); bm25_scores = bm25.get_scores(tokenize(candidate["candidate_text"])); bm25_ms = (time.perf_counter() - start) * 1000
        bm25_ranked = rank_from_scores(job_ids, bm25_scores)
        start = time.perf_counter()
        embedding = encoder.encode([candidate["candidate_text"]], convert_to_numpy=True, normalize_embeddings=True)[0]
        dense_scores = job_embeddings @ embedding; dense_ms = (time.perf_counter() - start) * 1000
        dense_ranked = rank_from_scores(job_ids, dense_scores)
        start = time.perf_counter(); rrf_ranked = reciprocal_rank_fusion([bm25_ranked, dense_ranked]); rrf_ms = (time.perf_counter() - start) * 1000
        ranked = {"BM25": bm25_ranked, "Dense-MiniLM": dense_ranked, "RRF": rrf_ranked}
        scores = {"BM25": dict(zip(job_ids, bm25_scores)), "Dense-MiniLM": dict(zip(job_ids, dense_scores)), "RRF": {}}
        latencies = {"BM25": bm25_ms, "Dense-MiniLM": dense_ms, "RRF": rrf_ms}
        feature_frame, feature_elapsed = build_feature_frame(candidate, jobs, bm25_scores, dense_scores)
        for name, bundle in bundles.items():
            start = time.perf_counter()
            ltr_scores = bundle["model"].predict(feature_frame[bundle["features"]])
            elapsed = feature_elapsed + (time.perf_counter() - start)
            ranked[name] = rank_from_scores(job_ids, ltr_scores)
            scores[name] = dict(zip(job_ids, ltr_scores)); latencies[name] = elapsed * 1000
        for method, ranking in ranked.items():
            methods[method]["metrics"].append(sparse_metrics(ranking, gold_scores))
            methods[method]["latencies"].append(latencies[method])
            rankings.extend({"candidate_id": candidate_id, "method": method, "rank": rank, "job_id": job_id,
                             "score": scores[method].get(job_id), "gold_label_if_available": gold_scores.get(job_id)}
                            for rank, job_id in enumerate(ranking[:50], start=1))
        print(f"Scored {number}/{len(candidates)} gold candidates", end="\r")
    comparison = pd.DataFrame([summarise(name, data["metrics"], data["latencies"], len(jobs)) for name, data in methods.items()])
    comparison.to_csv(RESULTS_DIR / "final_comparison.csv", index=False)
    pd.DataFrame(rankings).to_parquet(RESULTS_DIR / "gold_rankings.parquet", index=False)
    payload = {"evaluation_type": "sparse judged gold evaluation", "gold_candidates": int(gold.profile_id.nunique()),
               "gold_judgments": int(len(gold)), "job_corpus": int(len(jobs)), "results": comparison.to_dict("records"),
               "compute": {"embedding_dimension": int(job_embeddings.shape[1]), "embedding_memory_mb": round(job_embeddings.nbytes / 1024**2, 2), "ltr_feature_count": 8}}
    (RESULTS_DIR / "gold_ltr_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()
