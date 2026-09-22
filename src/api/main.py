"""FastAPI serving for hybrid retrieval followed by XGBoost LTR."""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.data.contracts import build_candidate_text, load_jobs, normalize_candidate, scoring_row
from src.ranking.features import build_features, get_candidate_skills, skill_matches
from src.retrieval.hybrid import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 200

FEATURES = ["bm25_score", "dense_score", "skill_coverage", "role_family_match", "domain_match",
            "experience_compatibility", "seniority_compatibility", "location_compatibility"]

app = FastAPI(title="TalentGraph API", description="Hybrid IR and learning-to-rank job recommendations.", version="1.1.0")


class CandidateRequest(BaseModel):
    profile_text: str = Field(..., min_length=5, description="Free-text candidate profile or career summary.")
    skills: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    seniority: str = ""
    experience_years: float = Field(default=0.0, ge=0, le=80)
    education: str = ""
    location: str = ""


def _load_resources() -> tuple[pd.DataFrame, BM25Okapi, faiss.Index, SentenceTransformer, dict]:
    with (MODELS_DIR / "dense_jobs_metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    index = faiss.read_index(str(MODELS_DIR / "dense_jobs.faiss"))
    job_ids = metadata["job_ids"]
    jobs_by_id = load_jobs().set_index("job_id", drop=False)
    missing = set(job_ids) - set(jobs_by_id.index)
    if missing:
        raise RuntimeError(f"API cannot serve: raw metadata is missing {len(missing)} indexed jobs.")
    jobs = jobs_by_id.loc[job_ids].reset_index(drop=True)
    if index.ntotal != len(jobs):
        raise RuntimeError("API cannot serve: FAISS index and dense metadata have different job counts.")
    bm25 = BM25Okapi([str(text).lower().split() for text in jobs.job_text])
    encoder = SentenceTransformer(MODEL_NAME)
    bundle = joblib.load(MODELS_DIR / "talentgraph_ltr.joblib")
    if bundle.get("features") != FEATURES:
        raise RuntimeError("API cannot serve: saved LTR feature schema differs from the API schema.")
    return jobs, bm25, index, encoder, bundle


JOBS, BM25, FAISS_INDEX, ENCODER, LTR_BUNDLE = _load_resources()


def request_candidate(request: CandidateRequest) -> dict:
    raw = {
        "profile_id": "api-request", "skills_primary": request.skills, "skills_secondary": [],
        "roles": request.roles, "domains": request.domains, "experience_years": request.experience_years,
        "seniority": request.seniority, "preferences": {"locations": [request.location] if request.location else []},
        "career_intent": request.profile_text, "dealbreakers": [],
    }
    text = "Profile: " + request.profile_text + "\nEducation: " + request.education + "\n" + build_candidate_text(raw)
    return normalize_candidate(raw, candidate_text=text)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "jobs": len(JOBS), "dense_index_vectors": int(FAISS_INDEX.ntotal), "model": "XGBoost-LTR"}


@app.post("/recommend")
def recommend(request: CandidateRequest, top_k: int = Query(default=10, ge=1, le=50)) -> dict:
    candidate = request_candidate(request)
    bm25_scores = BM25.get_scores(candidate["candidate_text"].lower().split())
    bm25_indices = np.argsort(bm25_scores)[::-1][:RETRIEVAL_K]
    bm25_ranked = [JOBS.job_id.iloc[index] for index in bm25_indices]
    embedding = ENCODER.encode([candidate["candidate_text"]], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    dense_scores, dense_indices = FAISS_INDEX.search(embedding, min(RETRIEVAL_K, len(JOBS)))
    dense_ranked = [JOBS.job_id.iloc[index] for index in dense_indices[0] if index >= 0]
    candidate_ids = reciprocal_rank_fusion([bm25_ranked, dense_ranked])
    candidate_jobs = JOBS.set_index("job_id", drop=False).loc[candidate_ids].reset_index(drop=True)
    dense_score_by_id = dict(zip(dense_ranked, dense_scores[0]))
    rows = []
    for _, job in candidate_jobs.iterrows():
        row = scoring_row(candidate, job.to_dict(), bm25_scores[JOBS.index[JOBS.job_id == job.job_id][0]], dense_score_by_id.get(job.job_id, 0.0))
        rows.append({**row, **build_features(row)})
    frame = pd.DataFrame(rows)
    frame["score"] = LTR_BUNDLE["model"].predict(frame[FEATURES])
    skills = get_candidate_skills(candidate)
    recommendations = []
    for _, row in frame.sort_values("score", ascending=False).head(top_k).iterrows():
        matched = [skill for skill in skills if skill_matches(skill, row.job_text)]
        factors = []
        if matched:
            factors.append(f"matched {len(matched)} of {len(skills)} listed skills")
        if row.role_family_match:
            factors.append("role-family alignment")
        if row.experience_compatibility >= 0.75:
            factors.append("experience compatible")
        if row.location_compatibility:
            factors.append("location compatible")
        recommendations.append({
            "job_id": row.job_id, "title": row.job_title, "company": row.company, "location": row.job_location,
            "score": round(float(row.score), 4), "matched_skills": matched, "skill_coverage": round(float(row.skill_coverage), 3),
            "semantic_score": round(float(row.dense_score), 3), "lexical_score": round(float(row.bm25_score), 3),
            "role_family_match": float(row.role_family_match), "domain_match": float(row.domain_match),
            "experience_compatibility": round(float(row.experience_compatibility), 3),
            "seniority_compatibility": round(float(row.seniority_compatibility), 3),
            "location_compatibility": float(row.location_compatibility),
            "explanation": "; ".join(factors) if factors else "Ranked from lexical and semantic profile similarity.",
        })
    return {"count": len(recommendations), "retrieved_candidates": len(frame), "recommendations": recommendations}
