"""Canonical data contracts shared by training, evaluation, and serving.

The processed pair datasets are useful training artifacts, but scorer inputs must
always be built from these functions.  This keeps raw JSON, API requests, and
feature engineering on the same explicit schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "role-radar"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if str(item).strip()]


def _pipe(value: Any) -> str:
    return "|".join(_as_list(value))


def build_candidate_text(profile: dict[str, Any]) -> str:
    preferences = profile.get("preferences") or {}
    return "\n".join(
        [
            "Roles: " + ", ".join(_as_list(profile.get("roles"))),
            "Primary skills: " + ", ".join(_as_list(profile.get("skills_primary"))),
            "Secondary skills: " + ", ".join(_as_list(profile.get("skills_secondary"))),
            f"Experience years: {profile.get('experience_years', 0)}",
            f"Seniority: {profile.get('seniority', '')}",
            "Domains: " + ", ".join(_as_list(profile.get("domains"))),
            "Preferred locations: " + ", ".join(_as_list(preferences.get("locations"))),
            f"Career intent: {profile.get('career_intent', '')}",
            "Dealbreakers: " + ", ".join(_as_list(profile.get("dealbreakers"))),
        ]
    )


def normalize_candidate(profile: dict[str, Any], *, candidate_text: str | None = None) -> dict[str, Any]:
    """Return the exact candidate fields required by ranking.features."""
    if "profile_id" not in profile:
        raise ValueError("Candidate contract requires 'profile_id' from synthetic_profiles.json or the API.")
    return {
        "profile_id": profile["profile_id"],
        "candidate_text": candidate_text or build_candidate_text(profile),
        "candidate_primary_skills": _pipe(profile.get("skills_primary")),
        "candidate_secondary_skills": _pipe(profile.get("skills_secondary")),
        "candidate_roles": _pipe(profile.get("roles")),
        "candidate_experience": float(profile.get("experience_years") or 0),
        "candidate_seniority": str(profile.get("seniority") or ""),
        "candidate_domains": _pipe(profile.get("domains")),
        "candidate_preferences": profile.get("preferences") or {},
        "candidate_career_intent": str(profile.get("career_intent") or ""),
        "candidate_dealbreakers": _pipe(profile.get("dealbreakers")),
    }


def build_job_text(job: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Title: {job.get('title', '')}",
            f"Company: {job.get('company', '')}",
            f"Location: {job.get('location', '')}",
            f"Description: {job.get('description', '')}",
            f"Seniority: {job.get('seniority_level', '')}",
            f"Employment type: {job.get('employment_type', '')}",
            f"Job function: {job.get('job_function', '')}",
            f"Industry: {job.get('industry', '')}",
            f"Role family: {job.get('role_family_hint', '')}",
            f"Domain: {job.get('domain_hint', '')}",
            f"Experience required: {job.get('experience_years_hint', '')}",
        ]
    )


def normalize_job(job: dict[str, Any], *, job_text: str | None = None) -> dict[str, Any]:
    """Return the exact job fields required by ranking.features and serving."""
    if "id" not in job and "job_id" not in job:
        raise ValueError("Job contract requires 'id' from scraped_jobs.json.")
    job_id = job.get("id", job.get("job_id"))
    return {
        "job_id": job_id,
        "job_text": job_text or build_job_text(job),
        "job_title": str(job.get("title") or ""),
        "company": str(job.get("company") or ""),
        "job_location": str(job.get("location") or ""),
        "job_seniority": str(job.get("seniority_level") or ""),
        "job_role_family": str(job.get("role_family_hint") or ""),
        "job_domain": str(job.get("domain_hint") or ""),
        "job_experience": float(job.get("experience_years_hint") or 0),
    }


def _load_json(name: str) -> list[dict[str, Any]]:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required raw data file is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_candidates() -> pd.DataFrame:
    return pd.DataFrame([normalize_candidate(row) for row in _load_json("synthetic_profiles.json")])


def load_jobs() -> pd.DataFrame:
    return pd.DataFrame([normalize_job(row) for row in _load_json("scraped_jobs.json")])


def scoring_row(candidate: dict[str, Any], job: dict[str, Any], bm25_score: float, dense_score: float) -> dict[str, Any]:
    """Combine canonical records into the complete feature-engineering row."""
    return {**candidate, **job, "bm25_score": float(bm25_score), "dense_score": float(dense_score)}
