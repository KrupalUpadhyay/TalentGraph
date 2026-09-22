import json
from pathlib import Path
from collections import Counter, defaultdict


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "role-radar"


def load_json(filename):
    with open(RAW / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def pair_parts(pair_id):
    """
    Pair IDs appear to follow:
        <profile_id>_<job_id>

    Example:
        19_linkedin_4374745665
    """
    profile_str, job_id = pair_id.split("_", 1)
    return int(profile_str), job_id


def main():

    profiles = load_json("synthetic_profiles.json")
    jobs = load_json("scraped_jobs.json")

    phase2_pairs = load_json("phase2_pairs.json")
    phase2_labels = load_json("phase2_labels.json")

    phase3_pairs = load_json("phase3_pairs.json")
    phase3_labels = load_json("phase3_labels.json")

    gold = load_json("gold_labels.json")

    profile_ids = {x["profile_id"] for x in profiles}
    job_ids = {x["id"] for x in jobs}

    p2_pair_map = {
        x["pair_id"]: x
        for x in phase2_pairs
    }

    p3_pair_map = {
        x["pair_id"]: x
        for x in phase3_pairs
    }

    p2_label_map = {
        x["pair_id"]: x
        for x in phase2_labels
    }

    p3_label_map = {
        x["pair_id"]: x
        for x in phase3_labels
    }

    gold_map = {
        x["pair_id"]: x
        for x in gold
    }

    # ---------------------------------------------------------
    # PHASE 3 LABEL COVERAGE
    # ---------------------------------------------------------

    print("=" * 70)
    print("PHASE 3 LABEL → PROFILE/JOB COVERAGE")
    print("=" * 70)

    invalid_phase3 = []

    phase3_profiles = set()
    phase3_jobs = set()

    for pair_id in p3_label_map:

        try:
            profile_id, job_id = pair_parts(pair_id)

            phase3_profiles.add(profile_id)
            phase3_jobs.add(job_id)

            if profile_id not in profile_ids:
                invalid_phase3.append(
                    (pair_id, "unknown_profile", profile_id)
                )

            if job_id not in job_ids:
                invalid_phase3.append(
                    (pair_id, "unknown_job", job_id)
                )

        except Exception as e:
            invalid_phase3.append(
                (pair_id, "parse_error", str(e))
            )

    print("Phase 3 labels:", len(p3_label_map))
    print("Unique profiles:", len(phase3_profiles))
    print("Unique jobs:", len(phase3_jobs))
    print("Invalid mappings:", len(invalid_phase3))

    if invalid_phase3:
        print("\nFirst 10 invalid mappings:")
        for x in invalid_phase3[:10]:
            print(x)

    # ---------------------------------------------------------
    # GOLD COVERAGE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GOLD SET COVERAGE")
    print("=" * 70)

    gold_profiles = set()
    gold_jobs = set()

    invalid_gold = []

    for pair_id in gold_map:

        try:
            profile_id, job_id = pair_parts(pair_id)

            gold_profiles.add(profile_id)
            gold_jobs.add(job_id)

            if profile_id not in profile_ids:
                invalid_gold.append(
                    (pair_id, "unknown_profile", profile_id)
                )

            if job_id not in job_ids:
                invalid_gold.append(
                    (pair_id, "unknown_job", job_id)
                )

        except Exception as e:
            invalid_gold.append(
                (pair_id, "parse_error", str(e))
            )

    print("Gold pairs:", len(gold_map))
    print("Gold profiles:", len(gold_profiles))
    print("Gold jobs:", len(gold_jobs))
    print("Invalid gold mappings:", len(invalid_gold))

    # ---------------------------------------------------------
    # GOLD ↔ PHASE 3
    # ---------------------------------------------------------

    gold_ids = set(gold_map)
    phase3_ids = set(p3_label_map)

    gold_in_phase3 = gold_ids & phase3_ids
    gold_not_in_phase3 = gold_ids - phase3_ids

    print("\n" + "=" * 70)
    print("GOLD ↔ PHASE 3")
    print("=" * 70)

    print("Gold pairs:", len(gold_ids))
    print("Gold also in Phase 3 labels:", len(gold_in_phase3))
    print("Gold absent from Phase 3 labels:", len(gold_not_in_phase3))

    if gold_not_in_phase3:
        print("\nGold pairs absent from Phase 3 labels:")
        for pair_id in sorted(gold_not_in_phase3):
            print(pair_id)

    # ---------------------------------------------------------
    # SCORE AGREEMENT
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GOLD vs PHASE 3 LABEL SCORE AGREEMENT")
    print("=" * 70)

    differences = []

    for pair_id in sorted(gold_in_phase3):

        gold_score = gold_map[pair_id]["composite"]
        phase3_score = p3_label_map[pair_id]["composite"]

        differences.append(
            phase3_score - gold_score
        )

    if differences:
        print("Pairs compared:", len(differences))
        print("Exact score matches:",
              sum(x == 0 for x in differences))
        print("Mean Phase3 - Gold difference:",
              sum(differences) / len(differences))
        print("Min difference:", min(differences))
        print("Max difference:", max(differences))

        print("\nDifference histogram:")

        hist = Counter(differences)

        for diff in sorted(hist):
            print(f"{diff:+d}: {hist[diff]}")

    # ---------------------------------------------------------
    # LABEL DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3 LABEL DISTRIBUTION")
    print("=" * 70)

    scores = [
        x["composite"]
        for x in phase3_labels
    ]

    print("Mean:", sum(scores) / len(scores))
    print("Min:", min(scores))
    print("Max:", max(scores))

    # ---------------------------------------------------------
    # PROFILE FREQUENCY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROFILE FREQUENCY IN PHASE 3 LABELS")
    print("=" * 70)

    profile_counts = Counter()

    for pair_id in p3_label_map:
        profile_id, _ = pair_parts(pair_id)
        profile_counts[profile_id] += 1

    values = list(profile_counts.values())

    print("Profiles:", len(values))
    print("Min pairs/profile:", min(values))
    print("Max pairs/profile:", max(values))
    print("Mean pairs/profile:", sum(values) / len(values))

    print("\nTop 10 profiles by pair count:")
    for profile_id, count in profile_counts.most_common(10):
        print(profile_id, count)

    # ---------------------------------------------------------
    # JOB FREQUENCY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("JOB FREQUENCY IN PHASE 3 LABELS")
    print("=" * 70)

    job_counts = Counter()

    for pair_id in p3_label_map:
        _, job_id = pair_parts(pair_id)
        job_counts[job_id] += 1

    values = list(job_counts.values())

    print("Jobs:", len(values))
    print("Min pairs/job:", min(values))
    print("Max pairs/job:", max(values))
    print("Mean pairs/job:", sum(values) / len(values))

    print("\nTop 10 jobs by pair count:")
    for job_id, count in job_counts.most_common(10):
        print(job_id, count)

    # ---------------------------------------------------------
    # PAIR TYPE → SCORE
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3 PAIR TYPE vs COMPOSITE")
    print("=" * 70)

    pair_type_scores = defaultdict(list)

    for pair_id, pair in p3_pair_map.items():

        if pair_id not in p3_label_map:
            continue

        pair_type = pair["pair_type"]
        score = p3_label_map[pair_id]["composite"]

        pair_type_scores[pair_type].append(score)

    for pair_type, values in pair_type_scores.items():

        print(
            f"{pair_type:15s}",
            "count =", len(values),
            "mean =", sum(values) / len(values),
            "min =", min(values),
            "max =", max(values)
        )

    # ---------------------------------------------------------
    # GOLD SCORE DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GOLD SCORE DISTRIBUTION")
    print("=" * 70)

    gold_scores = [
        x["composite"]
        for x in gold
    ]

    print(Counter(
        min(9, int(score // 10))
        for score in gold_scores
    ))


if __name__ == "__main__":
    main()
