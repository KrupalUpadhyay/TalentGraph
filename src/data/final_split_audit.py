import json
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "role-radar"


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

    valid_profiles = {
        x["profile_id"]
        for x in profiles
    }

    valid_jobs = {
        x["id"]
        for x in jobs
    }

    gold_ids = {
        x["pair_id"]
        for x in gold
    }

    # ---------------------------------------------------------
    # GOLD VALIDITY
    # ---------------------------------------------------------

    print("=" * 70)
    print("GOLD VALIDITY")
    print("=" * 70)

    valid_gold = []
    invalid_gold = []

    for row in gold:

        pair_id = row["pair_id"]

        try:
            profile_id, job_id = parse_pair_id(pair_id)

            profile_ok = profile_id in valid_profiles
            job_ok = job_id in valid_jobs

            if profile_ok and job_ok:
                valid_gold.append(row)
            else:
                invalid_gold.append(
                    {
                        "pair_id": pair_id,
                        "profile_id": profile_id,
                        "job_id": job_id,
                        "profile_exists": profile_ok,
                        "job_exists": job_ok,
                    }
                )

        except Exception as e:
            invalid_gold.append(
                {
                    "pair_id": pair_id,
                    "error": str(e),
                }
            )

    print("Total gold:", len(gold))
    print("Valid gold:", len(valid_gold))
    print("Invalid gold:", len(invalid_gold))

    print("\nInvalid gold records:")

    for row in invalid_gold:
        print(row)

    # ---------------------------------------------------------
    # GOLD CANDIDATES
    # ---------------------------------------------------------

    gold_profile_ids = set()

    for row in valid_gold:
        profile_id, _ = parse_pair_id(row["pair_id"])
        gold_profile_ids.add(profile_id)

    print("\n" + "=" * 70)
    print("GOLD CANDIDATES")
    print("=" * 70)

    print("Unique valid gold candidates:", len(gold_profile_ids))
    print("Candidate IDs:", sorted(gold_profile_ids))

    # ---------------------------------------------------------
    # VALID PHASE 3 LABELS
    # ---------------------------------------------------------

    valid_phase3 = []
    invalid_phase3 = []

    for row in phase3_labels:

        pair_id = row["pair_id"]

        try:
            profile_id, job_id = parse_pair_id(pair_id)

            profile_ok = profile_id in valid_profiles
            job_ok = job_id in valid_jobs

            if profile_ok and job_ok:
                valid_phase3.append(row)
            else:
                invalid_phase3.append(row)

        except Exception:
            invalid_phase3.append(row)

    print("\n" + "=" * 70)
    print("PHASE 3 VALIDITY")
    print("=" * 70)

    print("Total Phase 3 labels:", len(phase3_labels))
    print("Valid mappings:", len(valid_phase3))
    print("Invalid mappings:", len(invalid_phase3))

    # ---------------------------------------------------------
    # REMOVE GOLD CANDIDATES
    # ---------------------------------------------------------

    train_pool = []
    excluded_gold_candidate = []

    for row in valid_phase3:

        profile_id, _ = parse_pair_id(row["pair_id"])

        if profile_id in gold_profile_ids:
            excluded_gold_candidate.append(row)
        else:
            train_pool.append(row)

    print("\n" + "=" * 70)
    print("GOLD-CANDIDATE EXCLUSION")
    print("=" * 70)

    print("Valid Phase 3 labels:", len(valid_phase3))
    print("Excluded because candidate is in gold:", len(excluded_gold_candidate))
    print("Remaining training pool:", len(train_pool))

    # ---------------------------------------------------------
    # CHECK GOLD PAIR LEAKAGE
    # ---------------------------------------------------------

    train_ids = {
        x["pair_id"]
        for x in train_pool
    }

    leakage = train_ids & gold_ids

    print("\n" + "=" * 70)
    print("GOLD PAIR LEAKAGE CHECK")
    print("=" * 70)

    print("Gold pair IDs:", len(gold_ids))
    print("Training pair IDs:", len(train_ids))
    print("Direct pair leakage:", len(leakage))

    # ---------------------------------------------------------
    # CHECK CANDIDATE LEAKAGE
    # ---------------------------------------------------------

    train_profiles = {
        parse_pair_id(x["pair_id"])[0]
        for x in train_pool
    }

    candidate_leakage = train_profiles & gold_profile_ids

    print("\n" + "=" * 70)
    print("GOLD CANDIDATE LEAKAGE CHECK")
    print("=" * 70)

    print("Training candidates:", len(train_profiles))
    print("Gold candidates:", len(gold_profile_ids))
    print("Candidate leakage:", len(candidate_leakage))

    # ---------------------------------------------------------
    # SCORE DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING POOL SCORE DISTRIBUTION")
    print("=" * 70)

    scores = [
        x["composite"]
        for x in train_pool
    ]

    print("Count:", len(scores))
    print("Mean:", sum(scores) / len(scores))
    print("Min:", min(scores))
    print("Max:", max(scores))

    hist = Counter(
        min(9, int(score // 10))
        for score in scores
    )

    for bucket in sorted(hist):
        print(
            f"{bucket * 10:02d}-{bucket * 10 + 9:02d}: "
            f"{hist[bucket]}"
        )

    # ---------------------------------------------------------
    # CANDIDATE DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CANDIDATE DISTRIBUTION")
    print("=" * 70)

    counts = Counter(
        parse_pair_id(x["pair_id"])[0]
        for x in train_pool
    )

    print("Unique candidates:", len(counts))
    print("Min pairs/candidate:", min(counts.values()))
    print("Max pairs/candidate:", max(counts.values()))
    print("Mean pairs/candidate:",
          sum(counts.values()) / len(counts))

    print("\nTop 10:")
    for candidate_id, count in counts.most_common(10):
        print(candidate_id, count)


if __name__ == "__main__":
    main()
    