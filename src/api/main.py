from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "role-radar"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

TRAIN_FILE = PROCESSED_DIR / "train_pairs.parquet"
GOLD_FILE = PROCESSED_DIR / "gold_pairs.parquet"

RAW_JOBS_FILE = RAW_DIR / "scraped_jobs.json"

FAISS_FILE = MODELS_DIR / "dense_jobs.faiss"
FAISS_METADATA_FILE = MODELS_DIR / "dense_jobs_metadata.pkl"

BM25_FILE = MODELS_DIR / "bm25_index.pkl"
LTR_FILE = MODELS_DIR / "talentgraph_ltr.joblib"

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_K = 100
BM25_RRF_K = 200
DENSE_RRF_K = 200

FEATURES = [
    "bm25_score",
    "dense_score",
    "skill_coverage",
    "role_family_match",
    "domain_match",
    "experience_compatibility",
    "seniority_compatibility",
    "location_compatibility",
]


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="TalentGraph API",
    description="Hybrid candidate-job retrieval and Learning-to-Rank API",
    version="1.0.0",
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class CandidateRequest(BaseModel):
    name: str = "Candidate"

    roles: List[str] = Field(default_factory=list)

    skills_primary: List[str] = Field(default_factory=list)

    skills_secondary: List[str] = Field(default_factory=list)

    experience_years: float = 0.0

    seniority: str = ""

    domains: List[str] = Field(default_factory=list)

    preferences: Any = ""

    career_intent: str = ""

    dealbreakers: Any = ""

    location: str = ""


# ============================================================
# GLOBAL RESOURCES
# ============================================================

FAISS_INDEX = None
FAISS_METADATA = None

JOBS_DF = None

BM25 = None
DENSE_MODEL = None
LTR_MODEL = None

JOB_IDS = None
JOB_TEXTS = None

RESOURCE_STATUS = {
    "loaded": False,
    "jobs": 0,
    "faiss": 0,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip().lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, float) and np.isnan(value):
            return default

        return float(value)
    except Exception:
        return default


def parse_list(value: Any) -> List[str]:
    """
    Converts lists, tuples, JSON strings, pipe-separated strings,
    comma-separated strings, etc. into a clean list.
    """

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    # JSON list
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]
        except Exception:
            pass

    # pipe separated
    if "|" in text:
        return [
            x.strip()
            for x in text.split("|")
            if x.strip()
        ]

    # comma separated
    if "," in text:
        return [
            x.strip()
            for x in text.split(",")
            if x.strip()
        ]

    return [text]


def tokens(text: str) -> List[str]:
    return re.findall(r"\w+", normalize(text))


# ============================================================
# DATA LOADING
# ============================================================

def load_raw_jobs() -> pd.DataFrame:

    if not RAW_JOBS_FILE.exists():
        raise FileNotFoundError(
            f"Raw jobs file not found: {RAW_JOBS_FILE}"
        )

    with open(RAW_JOBS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = pd.DataFrame(data)

    if "id" not in raw.columns:
        raise RuntimeError(
            "scraped_jobs.json does not contain 'id'."
        )

    raw["job_id"] = raw["id"].astype(str)

    # These are the original dataset fields.
    rename_map = {
        "role_family_hint": "job_role_family",
        "domain_hint": "job_domain",
        "experience_years_hint": "job_experience",
        "seniority_level": "job_seniority",
        "location": "job_location",
    }

    raw = raw.rename(columns=rename_map)

    for column in [
        "title",
        "company",
        "job_location",
        "job_role_family",
        "job_domain",
        "job_seniority",
        "job_experience",
        "url",
        "description",
    ]:
        if column not in raw.columns:
            raw[column] = ""

    raw["job_id"] = raw["job_id"].astype(str)

    raw["job_experience"] = pd.to_numeric(
        raw["job_experience"],
        errors="coerce",
    ).fillna(0.0)

    return raw


def build_canonical_job_corpus() -> pd.DataFrame:

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {TRAIN_FILE}"
        )

    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {GOLD_FILE}"
        )

    print("Loading canonical train/gold job corpus...")

    train = pd.read_parquet(TRAIN_FILE)
    gold = pd.read_parquet(GOLD_FILE)

    required = {"job_id", "job_text"}

    if not required.issubset(train.columns):
        raise RuntimeError(
            f"train_pairs.parquet must contain {required}"
        )

    if not required.issubset(gold.columns):
        raise RuntimeError(
            f"gold_pairs.parquet must contain {required}"
        )

    # IMPORTANT:
    #
    # This is the same corpus construction used during retrieval
    # evaluation.
    #
    # The raw dataset contains 2082 jobs, while the trained dense
    # index contains 2054 jobs.
    #
    # Therefore we MUST use the train+gold corpus here.
    #
    jobs = pd.concat(
        [
            train[["job_id", "job_text"]],
            gold[["job_id", "job_text"]],
        ],
        ignore_index=True,
    )

    jobs["job_id"] = jobs["job_id"].astype(str)

    jobs = (
        jobs[
            ["job_id", "job_text"]
        ]
        .drop_duplicates("job_id")
        .reset_index(drop=True)
    )

    print(f"Canonical processed corpus: {len(jobs)} jobs")

    raw = load_raw_jobs()

    # Attach human-readable job metadata.
    raw_lookup = raw.drop_duplicates(
        "job_id"
    ).set_index("job_id")

    jobs = jobs.join(
        raw_lookup[
            [
                "title",
                "company",
                "job_location",
                "job_role_family",
                "job_domain",
                "job_seniority",
                "job_experience",
                "url",
            ]
        ],
        on="job_id",
    )

    # Fill missing metadata safely.
    metadata_columns = [
        "title",
        "company",
        "job_location",
        "job_role_family",
        "job_domain",
        "job_seniority",
        "job_experience",
        "url",
    ]

    for column in metadata_columns:
        if column not in jobs.columns:
            jobs[column] = ""

    jobs[metadata_columns] = jobs[
        metadata_columns
    ].fillna("")

    jobs["job_experience"] = pd.to_numeric(
        jobs["job_experience"],
        errors="coerce",
    ).fillna(0.0)

    return jobs


# ============================================================
# FAISS METADATA
# ============================================================

def extract_indexed_job_ids(metadata: Any) -> List[str]:

    """
    Handles common metadata formats used when saving FAISS
    metadata.
    """

    if metadata is None:
        raise RuntimeError(
            "FAISS metadata is empty."
        )

    # Dictionary
    if isinstance(metadata, dict):

        possible_keys = [
            "job_ids",
            "ids",
            "job_id",
            "metadata",
        ]

        for key in possible_keys:

            if key not in metadata:
                continue

            value = metadata[key]

            if isinstance(value, dict):
                # index -> job id
                try:
                    ordered = [
                        value[k]
                        for k in sorted(
                            value,
                            key=lambda x: int(x)
                        )
                    ]
                    return [str(x) for x in ordered]
                except Exception:
                    pass

            if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
                return [
                    str(x)
                    for x in value
                ]

    # Direct list / array
    if isinstance(
        metadata,
        (list, tuple, np.ndarray, pd.Series),
    ):
        return [
            str(x)
            for x in metadata
        ]

    raise RuntimeError(
        "Could not extract job IDs from "
        "dense_jobs_metadata.pkl."
    )


# ============================================================
# ALIGN CORPUS TO FAISS
# ============================================================

def align_jobs_to_faiss(
    jobs: pd.DataFrame,
    indexed_job_ids: List[str],
) -> pd.DataFrame:

    jobs = jobs.copy()

    jobs["job_id"] = jobs["job_id"].astype(str)

    indexed_job_ids = [
        str(x)
        for x in indexed_job_ids
    ]

    if len(indexed_job_ids) != FAISS_INDEX.ntotal:
        raise RuntimeError(
            "FAISS metadata mismatch:\n"
            f"FAISS vectors: {FAISS_INDEX.ntotal}\n"
            f"Metadata IDs: {len(indexed_job_ids)}"
        )

    job_lookup = (
        jobs.drop_duplicates("job_id")
        .set_index("job_id")
    )

    missing = [
        job_id
        for job_id in indexed_job_ids
        if job_id not in job_lookup.index
    ]

    if missing:
        raise RuntimeError(
            "Some FAISS job IDs are missing from "
            "the canonical train+gold corpus.\n"
            f"Missing count: {len(missing)}\n"
            f"Examples: {missing[:10]}"
        )

    # THIS IS CRITICAL.
    #
    # FAISS vector i must correspond to JOBS_DF row i.
    #
    jobs = job_lookup.loc[
        indexed_job_ids
    ].reset_index()

    if len(jobs) != FAISS_INDEX.ntotal:
        raise RuntimeError(
            "Final aligned corpus size mismatch."
        )

    return jobs


# ============================================================
# BM25
# ============================================================

def load_or_build_bm25():

    global BM25

    print("Building BM25 index over aligned corpus...")

    corpus = [
        tokens(text)
        for text in JOB_TEXTS
    ]

    BM25 = BM25Okapi(corpus)

    print(
        f"BM25 ready: {len(corpus)} documents"
    )


# ============================================================
# LTR MODEL
# ============================================================

def load_ltr_model():

    global LTR_MODEL

    if not LTR_FILE.exists():
        raise FileNotFoundError(
            f"LTR model not found: {LTR_FILE}"
        )

    bundle = joblib.load(LTR_FILE)

    # Expected trained bundle:
    #
    # {
    #     "model": XGBRanker,
    #     "features": [...]
    # }

    if isinstance(bundle, dict):

        if "model" in bundle:
            LTR_MODEL = bundle["model"]
        else:
            raise RuntimeError(
                "talentgraph_ltr.joblib does not "
                "contain a 'model' entry."
            )

        if "features" in bundle:
            trained_features = list(
                bundle["features"]
            )

            if trained_features != FEATURES:
                raise RuntimeError(
                    "Feature ordering mismatch between "
                    "trained model and serving code.\n"
                    f"Model: {trained_features}\n"
                    f"Serving: {FEATURES}"
                )

    else:
        # Fallback for directly serialized XGBRanker.
        LTR_MODEL = bundle

    print("Loaded trained XGBoost LTR model.")


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def get_candidate_skills(row: Dict[str, Any]) -> List[str]:

    skills = []

    skills.extend(
        parse_list(
            row.get("skills_primary", [])
        )
    )

    skills.extend(
        parse_list(
            row.get("skills_secondary", [])
        )
    )

    # Remove duplicates while preserving order.
    result = []
    seen = set()

    for skill in skills:

        normalized = normalize(skill)

        if not normalized:
            continue

        if normalized not in seen:
            seen.add(normalized)
            result.append(skill)

    return result


def skill_matches(
    skill: str,
    job_text: str,
) -> bool:

    skill = normalize(skill)
    job_text = normalize(job_text)

    if not skill or not job_text:
        return False

    # Exact phrase match.
    if skill in job_text:
        return True

    # Token-aware fallback.
    skill_tokens = tokens(skill)
    text_tokens = set(tokens(job_text))

    if len(skill_tokens) == 1:
        return skill_tokens[0] in text_tokens

    return all(
        token in text_tokens
        for token in skill_tokens
    )


def skill_coverage(row: Dict[str, Any]) -> float:

    skills = get_candidate_skills(row)

    if not skills:
        return 0.0

    job_text = normalize(
        row.get("job_text", "")
    )

    matched = sum(
        skill_matches(skill, job_text)
        for skill in skills
    )

    return matched / len(skills)


def role_family_match(
    row: Dict[str, Any],
) -> float:

    candidate_roles = normalize(
        row.get("candidate_roles", "")
    )

    job_role = normalize(
        row.get("job_role_family", "")
    )

    if not candidate_roles or not job_role:
        return 0.0

    roles = [
        x.strip()
        for x in candidate_roles.split("|")
        if x.strip()
    ]

    for role in roles:

        if role == job_role:
            return 1.0

        if role in job_role:
            return 1.0

        if job_role in role:
            return 1.0

    return 0.0


def domain_match(
    row: Dict[str, Any],
) -> float:

    candidate_domains = [
        normalize(x)
        for x in str(
            row.get("candidate_domains", "")
        ).split("|")
        if x.strip()
    ]

    job_domain = normalize(
        row.get("job_domain", "")
    )

    if not candidate_domains or not job_domain:
        return 0.0

    for domain in candidate_domains:

        if domain == job_domain:
            return 1.0

        if domain in job_domain:
            return 1.0

        if job_domain in domain:
            return 1.0

    return 0.0


def experience_compatibility(
    row: Dict[str, Any],
) -> float:

    candidate_exp = safe_float(
        row.get("candidate_experience", 0)
    )

    job_exp = safe_float(
        row.get("job_experience", 0)
    )

    if job_exp <= 0:
        return 1.0

    if candidate_exp >= job_exp:
        return 1.0

    gap = job_exp - candidate_exp

    return max(
        0.0,
        1.0 - gap / max(job_exp, 1.0),
    )


def seniority_to_level(
    value: Any,
) -> Optional[int]:

    value = normalize(value)

    mapping = {
        "intern": 0,
        "entry": 1,
        "junior": 1,
        "associate": 2,
        "mid": 2,
        "mid-level": 2,
        "senior": 3,
        "lead": 4,
        "principal": 5,
        "staff": 5,
    }

    for key, level in mapping.items():

        if key in value:
            return level

    return None


def seniority_compatibility(
    row: Dict[str, Any],
) -> float:

    candidate_level = seniority_to_level(
        row.get("candidate_seniority", "")
    )

    job_level = seniority_to_level(
        row.get("job_seniority", "")
    )

    if (
        candidate_level is None
        or job_level is None
    ):
        return 0.0

    difference = (
        candidate_level - job_level
    )

    if difference == 0:
        return 1.0

    if difference == 1:
        return 0.75

    if difference == -1:
        return 0.50

    if difference > 1:
        return 0.50

    return 0.0


def extract_locations(
    candidate_text: str,
) -> set:

    text = normalize(candidate_text)

    locations = [
        "bangalore",
        "bengaluru",
        "mumbai",
        "delhi",
        "new delhi",
        "gurgaon",
        "gurugram",
        "noida",
        "hyderabad",
        "pune",
        "chennai",
        "kolkata",
        "ahmedabad",
        "jaipur",
        "jodhpur",
        "indore",
        "chandigarh",
        "kochi",
        "remote",
    ]

    return {
        location
        for location in locations
        if location in text
    }


def location_compatibility(
    row: Dict[str, Any],
) -> float:

    candidate_locations = extract_locations(
        row.get("candidate_text", "")
    )

    job_location = normalize(
        row.get("job_location", "")
    )

    if not candidate_locations:
        return 0.0

    if "remote" in job_location:
        return 1.0

    for location in candidate_locations:

        if location in job_location:
            return 1.0

    return 0.0


# ============================================================
# CANDIDATE TEXT
# ============================================================

def build_candidate_text(
    candidate: CandidateRequest,
) -> str:

    parts = []

    if candidate.roles:
        parts.append(
            "Roles: "
            + ", ".join(candidate.roles)
        )

    if candidate.skills_primary:
        parts.append(
            "Primary skills: "
            + ", ".join(
                candidate.skills_primary
            )
        )

    if candidate.skills_secondary:
        parts.append(
            "Secondary skills: "
            + ", ".join(
                candidate.skills_secondary
            )
        )

    parts.append(
        f"Experience: "
        f"{candidate.experience_years} years"
    )

    if candidate.seniority:
        parts.append(
            "Seniority: "
            + candidate.seniority
        )

    if candidate.domains:
        parts.append(
            "Domains: "
            + ", ".join(candidate.domains)
        )

    if candidate.preferences:
        parts.append(
            "Preferences: "
            + str(candidate.preferences)
        )

    if candidate.career_intent:
        parts.append(
            "Career intent: "
            + candidate.career_intent
        )

    if candidate.dealbreakers:
        parts.append(
            "Dealbreakers: "
            + str(candidate.dealbreakers)
        )

    if candidate.location:
        parts.append(
            "Location: "
            + candidate.location
        )

    return " | ".join(parts)


# ============================================================
# FEATURE ROW
# ============================================================

def build_feature_row(
    candidate: CandidateRequest,
    job: pd.Series,
    bm25_score: float,
    dense_score: float,
) -> Dict[str, Any]:

    candidate_roles = "|".join(
        parse_list(candidate.roles)
    )

    candidate_domains = "|".join(
        parse_list(candidate.domains)
    )

    row = {
        "job_text": job["job_text"],

        "candidate_experience":
            candidate.experience_years,

        "job_experience":
            job["job_experience"],

        "job_role_family":
            job["job_role_family"],

        "job_domain":
            job["job_domain"],

        "job_seniority":
            job["job_seniority"],

        "job_location":
            job["job_location"],

        "candidate_roles":
            candidate_roles,

        "candidate_domains":
            candidate_domains,

        "candidate_seniority":
            candidate.seniority,

        "candidate_text":
            build_candidate_text(candidate),

        "skills_primary":
            candidate.skills_primary,

        "skills_secondary":
            candidate.skills_secondary,

        "bm25_score":
            float(bm25_score),

        "dense_score":
            float(dense_score),
    }

    return row


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    bm25_scores: np.ndarray,
    dense_scores: np.ndarray,
) -> List[int]:

    bm25_order = np.argsort(
        -bm25_scores
    )[:BM25_RRF_K]

    dense_order = np.argsort(
        -dense_scores
    )[:DENSE_RRF_K]

    rrf = {}

    for rank, idx in enumerate(
        bm25_order,
        start=1,
    ):
        rrf[idx] = rrf.get(idx, 0.0) + (
            1.0 / (60.0 + rank)
        )

    for rank, idx in enumerate(
        dense_order,
        start=1,
    ):
        rrf[idx] = rrf.get(idx, 0.0) + (
            1.0 / (60.0 + rank)
        )

    ordered = sorted(
        rrf,
        key=rrf.get,
        reverse=True,
    )

    return ordered


# ============================================================
# SCORE NORMALIZATION FOR UI
# ============================================================

def sigmoid(x: float) -> float:

    x = np.clip(
        x,
        -30,
        30,
    )

    return float(
        1.0 / (1.0 + np.exp(-x))
    )


def percentage_01(value: float) -> float:

    return round(
        float(
            np.clip(value, 0.0, 1.0)
            * 100.0
        ),
        1,
    )


def dense_percentage(value: float) -> float:

    # Cosine similarity -> display-only percentage.
    value = float(value)

    normalized = (
        (value + 1.0) / 2.0
    )

    return round(
        float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
            * 100.0
        ),
        1,
    )


def bm25_display_percentage(
    value: float,
    all_scores: np.ndarray,
) -> float:

    minimum = float(
        np.min(all_scores)
    )

    maximum = float(
        np.max(all_scores)
    )

    if maximum <= minimum:
        return 0.0

    normalized = (
        (value - minimum)
        / (maximum - minimum)
    )

    return round(
        float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
            * 100.0
        ),
        1,
    )


# ============================================================
# EXPLANATION
# ============================================================

def matched_skills(
    candidate: CandidateRequest,
    job_text: str,
) -> List[str]:

    result = []

    for skill in get_candidate_skills(
        candidate.model_dump()
    ):

        if skill_matches(
            skill,
            job_text,
        ):
            result.append(skill)

    return result


def build_explanation(
    candidate: CandidateRequest,
    job: pd.Series,
    features: Dict[str, float],
) -> Dict[str, Any]:

    matched = matched_skills(
        candidate,
        job["job_text"],
    )

    reasons = []

    if features["skill_coverage"] >= 0.5:
        reasons.append(
            "Strong skill alignment"
        )
    elif features["skill_coverage"] > 0:
        reasons.append(
            "Some relevant skills match"
        )

    if features["role_family_match"] == 1:
        reasons.append(
            "Role family matches"
        )

    if features["domain_match"] == 1:
        reasons.append(
            "Domain matches"
        )

    if features[
        "experience_compatibility"
    ] >= 0.75:
        reasons.append(
            "Experience is compatible"
        )

    if features[
        "seniority_compatibility"
    ] >= 0.75:
        reasons.append(
            "Seniority is compatible"
        )

    if features[
        "location_compatibility"
    ] == 1:
        reasons.append(
            "Location preference matches"
        )

    if not reasons:
        reasons.append(
            "Retrieved as a relevant candidate-job match"
        )

    return {
        "matched_skills": matched,
        "reasons": reasons,
    }


# ============================================================
# STARTUP
# ============================================================

def load_resources():

    global FAISS_INDEX
    global FAISS_METADATA
    global JOBS_DF
    global JOB_IDS
    global JOB_TEXTS
    global DENSE_MODEL

    if RESOURCE_STATUS["loaded"]:
        return

    print()
    print("=" * 70)
    print("Loading TalentGraph...")
    print("=" * 70)

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    if not FAISS_FILE.exists():
        raise FileNotFoundError(
            f"Missing FAISS index: {FAISS_FILE}"
        )

    print("Loading FAISS index...")

    FAISS_INDEX = faiss.read_index(
        str(FAISS_FILE)
    )

    print(
        f"FAISS index contains "
        f"{FAISS_INDEX.ntotal} jobs."
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if not FAISS_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing FAISS metadata: "
            f"{FAISS_METADATA_FILE}"
        )

    print("Loading FAISS metadata...")

    with open(
        FAISS_METADATA_FILE,
        "rb",
    ) as f:
        FAISS_METADATA = pickle.load(f)

    indexed_job_ids = extract_indexed_job_ids(
        FAISS_METADATA
    )

    # --------------------------------------------------------
    # Canonical corpus
    # --------------------------------------------------------

    canonical_jobs = (
        build_canonical_job_corpus()
    )

    # --------------------------------------------------------
    # CRITICAL ALIGNMENT
    # --------------------------------------------------------

    JOBS_DF = align_jobs_to_faiss(
        canonical_jobs,
        indexed_job_ids,
    )

    JOB_IDS = (
        JOBS_DF["job_id"]
        .astype(str)
        .tolist()
    )

    JOB_TEXTS = (
        JOBS_DF["job_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    if len(JOBS_DF) != FAISS_INDEX.ntotal:
        raise RuntimeError(
            "FINAL CORPUS MISMATCH:\n"
            f"FAISS={FAISS_INDEX.ntotal}\n"
            f"JOBS={len(JOBS_DF)}"
        )

    print(
        f"Aligned job corpus: "
        f"{len(JOBS_DF)} jobs."
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    load_or_build_bm25()

    # --------------------------------------------------------
    # MiniLM
    # --------------------------------------------------------

    print(
        "Loading MiniLM sentence-transformer..."
    )

    DENSE_MODEL = SentenceTransformer(
        DEFAULT_MODEL_NAME
    )

    print("MiniLM ready.")

    # --------------------------------------------------------
    # LTR
    # --------------------------------------------------------

    load_ltr_model()

    RESOURCE_STATUS["loaded"] = True
    RESOURCE_STATUS["jobs"] = len(JOBS_DF)
    RESOURCE_STATUS["faiss"] = FAISS_INDEX.ntotal

    print()
    print("=" * 70)
    print("TalentGraph is ready.")
    print(
        f"Jobs: {len(JOBS_DF)}"
    )
    print(
        f"FAISS vectors: {FAISS_INDEX.ntotal}"
    )
    print(
        f"Features: {len(FEATURES)}"
    )
    print("=" * 70)
    print()


@app.on_event("startup")
def startup_event():
    load_resources()


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "TalentGraph",
        "resources_loaded":
            RESOURCE_STATUS["loaded"],
        "jobs":
            RESOURCE_STATUS["jobs"],
        "faiss_vectors":
            RESOURCE_STATUS["faiss"],
    }


@app.get("/")
def root():

    return {
        "service": "TalentGraph",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# RECOMMEND
# ============================================================

@app.post("/recommend")
def recommend(
    candidate: CandidateRequest,
):

    if not RESOURCE_STATUS["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="TalentGraph resources are not loaded.",
        )

    try:

        # ----------------------------------------------------
        # Candidate query
        # ----------------------------------------------------

        candidate_text = (
            build_candidate_text(candidate)
        )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        query_tokens = tokens(
            candidate_text
        )

        bm25_scores = np.asarray(
            BM25.get_scores(query_tokens),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Dense retrieval
        # ----------------------------------------------------

        query_embedding = (
            DENSE_MODEL.encode(
                [candidate_text],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            .astype(np.float32)
        )

        dense_scores, dense_indices = (
            FAISS_INDEX.search(
                query_embedding,
                min(
                    RETRIEVAL_K,
                    FAISS_INDEX.ntotal,
                ),
            )
        )

        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]

        # ----------------------------------------------------
        # Hybrid RRF
        # ----------------------------------------------------

        candidate_indices = (
            reciprocal_rank_fusion(
                bm25_scores,
                np.pad(
                    dense_scores,
                    (
                        0,
                        FAISS_INDEX.ntotal
                        - len(dense_scores),
                    ),
                    constant_values=-np.inf,
                ),
            )
        )

        # Keep only RRF candidates and make sure we don't
        # accidentally include invalid FAISS padding.
        valid_indices = [
            int(i)
            for i in candidate_indices
            if 0 <= int(i) < len(JOBS_DF)
        ]

        # Limit the number sent to XGBoost.
        valid_indices = valid_indices[
            : max(
                RETRIEVAL_K,
                50,
            )
        ]

        # ----------------------------------------------------
        # Dense score lookup
        # ----------------------------------------------------

        dense_score_map = {
            int(idx): float(score)
            for idx, score in zip(
                dense_indices,
                dense_scores,
            )
            if int(idx) >= 0
        }

        # ----------------------------------------------------
        # Build LTR feature matrix
        # ----------------------------------------------------

        feature_rows = []
        metadata_rows = []

        for idx in valid_indices:

            job = JOBS_DF.iloc[idx]

            dense_score = dense_score_map.get(
                idx,
                -1.0,
            )

            row = build_feature_row(
                candidate=candidate,
                job=job,
                bm25_score=float(
                    bm25_scores[idx]
                ),
                dense_score=dense_score,
            )

            feature_values = {
                "bm25_score":
                    float(
                        row["bm25_score"]
                    ),

                "dense_score":
                    float(
                        row["dense_score"]
                    ),

                "skill_coverage":
                    skill_coverage(row),

                "role_family_match":
                    role_family_match(row),

                "domain_match":
                    domain_match(row),

                "experience_compatibility":
                    experience_compatibility(row),

                "seniority_compatibility":
                    seniority_compatibility(row),

                "location_compatibility":
                    location_compatibility(row),
            }

            feature_rows.append(
                [
                    feature_values[name]
                    for name in FEATURES
                ]
            )

            metadata_rows.append(
                (
                    idx,
                    job,
                    feature_values,
                )
            )

        if not feature_rows:
            return {
                "candidate": candidate.name,
                "results": [],
            }

        X = np.asarray(
            feature_rows,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # XGBoost LTR inference
        # ----------------------------------------------------

        ranking_scores = (
            LTR_MODEL.predict(X)
        )

        ranking_scores = np.asarray(
            ranking_scores,
            dtype=np.float32,
        )

        order = np.argsort(
            -ranking_scores
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        results = []

        for position, local_idx in enumerate(
            order[:10],
            start=1,
        ):

            global_idx, job, feature_values = (
                metadata_rows[int(local_idx)]
            )

            ranking_score = float(
                ranking_scores[int(local_idx)]
            )

            explanation = (
                build_explanation(
                    candidate,
                    job,
                    feature_values,
                )
            )

            # Ranking score is NOT a probability.
            # This sigmoid is only a bounded display score.
            match_score = round(
                sigmoid(
                    ranking_score
                ) * 100.0,
                1,
            )

            result = {
                "rank": position,

                "job_id": str(
                    job["job_id"]
                ),

                "title": str(
                    job["title"]
                ),

                "company": str(
                    job["company"]
                ),

                "location": str(
                    job["job_location"]
                ),

                "seniority": str(
                    job["job_seniority"]
                ),

                "domain": str(
                    job["job_domain"]
                ),

                "url": str(
                    job["url"]
                ),

                "match_score": match_score,

                "ranking_score": round(
                    ranking_score,
                    6,
                ),

                "matched_skills":
                    explanation[
                        "matched_skills"
                    ],

                "reasons":
                    explanation[
                        "reasons"
                    ],

                "features": {
                    "bm25_score":
                        bm25_display_percentage(
                            feature_values[
                                "bm25_score"
                            ],
                            bm25_scores,
                        ),

                    "dense_score":
                        dense_percentage(
                            feature_values[
                                "dense_score"
                            ]
                        ),

                    "skill_coverage":
                        percentage_01(
                            feature_values[
                                "skill_coverage"
                            ]
                        ),

                    "role_family_match":
                        percentage_01(
                            feature_values[
                                "role_family_match"
                            ]
                        ),

                    "domain_match":
                        percentage_01(
                            feature_values[
                                "domain_match"
                            ]
                        ),

                    "experience_compatibility":
                        percentage_01(
                            feature_values[
                                "experience_compatibility"
                            ]
                        ),

                    "seniority_compatibility":
                        percentage_01(
                            feature_values[
                                "seniority_compatibility"
                            ]
                        ),

                    "location_compatibility":
                        percentage_01(
                            feature_values[
                                "location_compatibility"
                            ]
                        ),
                },
            }

            results.append(result)

        return {
            "candidate": candidate.name,
            "query": candidate_text,
            "retrieval": {
                "bm25": True,
                "dense": True,
                "hybrid_rrf": True,
                "learning_to_rank": True,
            },
            "results": results,
        }

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )