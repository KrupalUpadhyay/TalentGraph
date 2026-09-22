import json
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "role-radar"


def load_json(filename):
    path = RAW / filename

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(name, data):
    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")
    print("Type:", type(data).__name__)

    if isinstance(data, list):
        print("Records:", len(data))

        if data:
            print("Fields:", list(data[0].keys()))

    elif isinstance(data, dict):
        print("Keys:", list(data.keys()))


def main():

    jobs = load_json("scraped_jobs.json")
    profiles = load_json("synthetic_profiles.json")

    phase2_pairs = load_json("phase2_pairs.json")
    phase2_labels = load_json("phase2_labels.json")

    phase3_pairs = load_json("phase3_pairs.json")
    phase3_labels = load_json("phase3_labels.json")

    gold = load_json("gold_labels.json")
    benchmarks = load_json("benchmarks.json")

    summarize("JOBS", jobs)
    summarize("PROFILES", profiles)

    summarize("PHASE 2 PAIRS", phase2_pairs)
    summarize("PHASE 2 LABELS", phase2_labels)

    summarize("PHASE 3 PAIRS", phase3_pairs)
    summarize("PHASE 3 LABELS", phase3_labels)

    summarize("GOLD LABELS", gold)
    summarize("BENCHMARKS", benchmarks)

    # ---------------------------------------------------------
    # IDs
    # ---------------------------------------------------------

    job_ids = {x["id"] for x in jobs}
    profile_ids = {x["profile_id"] for x in profiles}

    p2_pair_ids = {x["pair_id"] for x in phase2_pairs}
    p2_label_ids = {x["pair_id"] for x in phase2_labels}

    p3_pair_ids = {x["pair_id"] for x in phase3_pairs}
    p3_label_ids = {x["pair_id"] for x in phase3_labels}

    gold_ids = {x["pair_id"] for x in gold}

    print("\n" + "=" * 70)
    print("ID CONSISTENCY")
    print("=" * 70)

    print("Phase 2 pairs:", len(p2_pair_ids))
    print("Phase 2 labels:", len(p2_label_ids))
    print("Phase 2 missing labels:", len(p2_pair_ids - p2_label_ids))
    print("Phase 2 extra labels:", len(p2_label_ids - p2_pair_ids))

    print("\nPhase 3 pairs:", len(p3_pair_ids))
    print("Phase 3 labels:", len(p3_label_ids))
    print("Phase 3 pair records missing labels:",
          len(p3_pair_ids - p3_label_ids))

    print("\nGold labels:", len(gold_ids))

    print("\nPhase 2 ↔ Phase 3 pair overlap:",
          len(p2_pair_ids & p3_pair_ids))

    print("Phase 2 labels ↔ Phase 3 labels overlap:",
          len(p2_label_ids & p3_label_ids))

    print("Gold ↔ Phase 2 overlap:",
          len(gold_ids & p2_label_ids))

    print("Gold ↔ Phase 3 overlap:",
          len(gold_ids & p3_label_ids))

    # ---------------------------------------------------------
    # Candidate/job coverage
    # ---------------------------------------------------------

    def pair_stats(pairs, name):

        candidate_ids = {x["profile_id"] for x in pairs}
        job_ids_used = {x["job_id"] for x in pairs}

        print(f"\n{name}")
        print("-" * 50)
        print("Unique candidates:", len(candidate_ids))
        print("Unique jobs:", len(job_ids_used))

        unknown_candidates = candidate_ids - profile_ids
        unknown_jobs = job_ids_used - job_ids

        print("Unknown candidate IDs:", len(unknown_candidates))
        print("Unknown job IDs:", len(unknown_jobs))

    pair_stats(phase2_pairs, "PHASE 2")
    pair_stats(phase3_pairs, "PHASE 3")

    # ---------------------------------------------------------
    # Pair types
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PAIR TYPES")
    print("=" * 70)

    print("Phase 2:")
    print(Counter(x.get("pair_type") for x in phase2_pairs))

    print("\nPhase 3:")
    print(Counter(x.get("pair_type") for x in phase3_pairs))

    # ---------------------------------------------------------
    # Label distributions
    # ---------------------------------------------------------

    def label_stats(labels, name):

        composites = [
            x["composite"]
            for x in labels
            if "composite" in x
        ]

        print(f"\n{name}")
        print("-" * 50)
        print("Labels:", len(labels))

        if composites:
            print("Composite min:", min(composites))
            print("Composite max:", max(composites))
            print("Composite mean:",
                  sum(composites) / len(composites))

            bins = Counter(
                min(9, int(score // 10))
                for score in composites
            )

            print("Composite histogram:")
            for bucket in sorted(bins):
                print(
                    f"{bucket * 10:02d}-{bucket * 10 + 9:02d}: "
                    f"{bins[bucket]}"
                )

    label_stats(phase2_labels, "PHASE 2 LABELS")
    label_stats(phase3_labels, "PHASE 3 LABELS")
    label_stats(gold, "GOLD LABELS")

    # ---------------------------------------------------------
    # Benchmark inspection
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("BENCHMARK CONTENT")
    print("=" * 70)

    for key, value in benchmarks.items():

        print(f"\n[{key}]")
        print("Type:", type(value).__name__)

        if isinstance(value, list):
            print("Length:", len(value))
            print("Sample:", value[:2])

        elif isinstance(value, dict):
            print("Keys:", list(value.keys()))

        else:
            print("Value:", value)


if __name__ == "__main__":
    main()