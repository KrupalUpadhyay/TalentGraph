import json
from pathlib import Path

import pandas as pd

from src.data.contracts import (
    build_candidate_text,
    build_job_text,
    normalize_candidate,
    normalize_job,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "role-radar"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)


def load_json(filename):
    with open(RAW / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_pair_id(pair_id):
    profile_str, job_id = pair_id.split("_", 1)
    return int(profile_str), job_id


def main():

    profiles = load_json("synthetic_profiles.json")
    jobs = load_json("scraped_jobs.json")
    phase3_labels = load_json("phase3_labels.json")
    gold = load_json("gold_labels.json")

    profile_map = {
        x["profile_id"]: x
        for x in profiles
    }

    job_map = {
        x["id"]: x
        for x in jobs
    }

    # ---------------------------------------------------------
    # GOLD CANDIDATES
    # ---------------------------------------------------------

    gold_candidates = set()

    valid_gold_rows = []

    for row in gold:

        try:
            profile_id, job_id = parse_pair_id(row["pair_id"])

            if (
                profile_id in profile_map
                and job_id in job_map
            ):
                gold_candidates.add(profile_id)
                valid_gold_rows.append(row)

        except Exception:
            continue

    # ---------------------------------------------------------
    # BUILD TRAINING DATASET
    # ---------------------------------------------------------

    train_rows = []

    for row in phase3_labels:

        try:
            profile_id, job_id = parse_pair_id(row["pair_id"])

            # Invalid profile/job
            if profile_id not in profile_map:
                continue

            if job_id not in job_map:
                continue

            # Strict candidate-level gold exclusion
            if profile_id in gold_candidates:
                continue

            profile = profile_map[profile_id]
            job = job_map[job_id]

            train_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "profile_id": profile_id,
                    "job_id": job_id,

                    # Candidate fields
                    "candidate_text": build_candidate_text(profile),
                    "candidate_experience": profile.get(
                        "experience_years", 0
                    ),
                    "candidate_seniority": profile.get(
                        "seniority", ""
                    ),
                    "candidate_roles": "|".join(
                        profile.get("roles", [])
                    ),
                    "candidate_primary_skills": "|".join(
                        profile.get("skills_primary", [])
                    ),
                    "candidate_secondary_skills": "|".join(
                        profile.get("skills_secondary", [])
                    ),
                    "candidate_domains": "|".join(
                        profile.get("domains", [])
                    ),

                    # Job fields
                    "job_text": build_job_text(job),
                    "job_title": job.get("title", ""),
                    "job_location": job.get("location", ""),
                    "job_seniority": job.get(
                        "seniority_level", ""
                    ),
                    "job_role_family": job.get(
                        "role_family_hint", ""
                    ),
                    "job_domain": job.get(
                        "domain_hint", ""
                    ),
                    "job_experience": job.get(
                        "experience_years_hint", 0
                    ),

                    # Weak supervision
                    "label": row["composite"],

                    "skills_label": row["skills"],
                    "seniority_label": row["seniority"],
                    "domain_label": row["domain"],
                    "location_label": row["location"],

                    "overqualified": row.get(
                        "overqualified", False
                    ),
                }
            )

        except Exception:
            continue

    train_df = pd.DataFrame(train_rows)

    # ---------------------------------------------------------
    # BUILD GOLD TEST DATASET
    # ---------------------------------------------------------

    gold_rows = []

    for row in valid_gold_rows:

        profile_id, job_id = parse_pair_id(row["pair_id"])

        profile = profile_map[profile_id]
        job = job_map[job_id]

        gold_rows.append(
            {
                "pair_id": row["pair_id"],
                "profile_id": profile_id,
                "job_id": job_id,

                "candidate_text": build_candidate_text(profile),

                "candidate_experience": profile.get(
                    "experience_years", 0
                ),

                "candidate_seniority": profile.get(
                    "seniority", ""
                ),

                "candidate_roles": "|".join(
                    profile.get("roles", [])
                ),

                "candidate_primary_skills": "|".join(
                    profile.get("skills_primary", [])
                ),

                "candidate_secondary_skills": "|".join(
                    profile.get("skills_secondary", [])
                ),

                "candidate_domains": "|".join(
                    profile.get("domains", [])
                ),

                "job_text": build_job_text(job),

                "job_title": job.get("title", ""),
                "job_location": job.get("location", ""),
                "job_seniority": job.get(
                    "seniority_level", ""
                ),
                "job_role_family": job.get(
                    "role_family_hint", ""
                ),
                "job_domain": job.get(
                    "domain_hint", ""
                ),
                "job_experience": job.get(
                    "experience_years_hint", 0
                ),

                # Gold target
                "label": row["composite"],

                "skills_label": row["skills"],
                "seniority_label": row["seniority"],
                "domain_label": row["domain"],
                "location_label": row["location"],

                "overqualified": row.get(
                    "overqualified", False
                ),

                "gold_reasoning": row.get(
                    "reasoning", ""
                ),
            }
        )

    gold_df = pd.DataFrame(gold_rows)

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    train_path = PROCESSED / "train_pairs.parquet"
    gold_path = PROCESSED / "gold_pairs.parquet"

    train_df.to_parquet(
        train_path,
        index=False
    )

    gold_df.to_parquet(
        gold_path,
        index=False
    )

    print("=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)

    print("\nTraining:")
    print("Rows:", len(train_df))
    print("Candidates:", train_df["profile_id"].nunique())
    print("Jobs:", train_df["job_id"].nunique())

    print("\nGold:")
    print("Rows:", len(gold_df))
    print("Candidates:", gold_df["profile_id"].nunique())
    print("Jobs:", gold_df["job_id"].nunique())

    print("\nLabel statistics:")
    print(train_df["label"].describe())

    print("\nFiles:")
    print(train_path)
    print(gold_path)


if __name__ == "__main__":
    main()
