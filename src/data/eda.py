import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)


def main():

    train = pd.read_parquet(DATA / "train_pairs.parquet")
    gold = pd.read_parquet(DATA / "gold_pairs.parquet")

    print("=" * 70)
    print("TALENTGRAPH EDA")
    print("=" * 70)

    # ---------------------------------------------------------
    # BASIC DATASET STATISTICS
    # ---------------------------------------------------------

    print("\nTRAINING DATA")
    print("-" * 50)

    print("Rows:", len(train))
    print("Candidates:", train["profile_id"].nunique())
    print("Jobs:", train["job_id"].nunique())

    print("\nGOLD DATA")
    print("-" * 50)

    print("Rows:", len(gold))
    print("Candidates:", gold["profile_id"].nunique())
    print("Jobs:", gold["job_id"].nunique())

    # ---------------------------------------------------------
    # LABEL DISTRIBUTION
    # ---------------------------------------------------------

    print("\nLABEL DISTRIBUTION")
    print("-" * 50)

    print(train["label"].describe())

    # ---------------------------------------------------------
    # PAIRS PER CANDIDATE
    # ---------------------------------------------------------

    candidate_counts = (
        train
        .groupby("profile_id")
        .size()
        .sort_values(ascending=False)
    )

    print("\nPAIRS PER CANDIDATE")
    print("-" * 50)

    print(candidate_counts.describe())

    # ---------------------------------------------------------
    # PAIRS PER JOB
    # ---------------------------------------------------------

    job_counts = (
        train
        .groupby("job_id")
        .size()
        .sort_values(ascending=False)
    )

    print("\nPAIRS PER JOB")
    print("-" * 50)

    print(job_counts.describe())

    # ---------------------------------------------------------
    # GOLD LABEL DISTRIBUTION
    # ---------------------------------------------------------

    print("\nGOLD LABEL DISTRIBUTION")
    print("-" * 50)

    print(gold["label"].describe())

    # ---------------------------------------------------------
    # LABEL HISTOGRAM
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        train["label"],
        bins=range(
            int(train["label"].min()),
            int(train["label"].max()) + 2
        ),
        edgecolor="black"
    )

    plt.xlabel("Composite Match Score")
    plt.ylabel("Number of Candidate–Job Pairs")
    plt.title("TalentGraph Training Label Distribution")
    plt.tight_layout()

    plt.savefig(
        RESULTS / "label_distribution.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------------------------------
    # PAIRS PER CANDIDATE
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        candidate_counts,
        bins=30,
        edgecolor="black"
    )

    plt.xlabel("Number of Candidate–Job Pairs")
    plt.ylabel("Number of Candidates")
    plt.title("Candidate Pair Coverage")

    plt.tight_layout()

    plt.savefig(
        RESULTS / "candidate_pair_coverage.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------------------------------
    # PAIRS PER JOB
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        job_counts,
        bins=30,
        edgecolor="black"
    )

    plt.xlabel("Number of Candidate–Job Pairs")
    plt.ylabel("Number of Jobs")
    plt.title("Job Pair Coverage")

    plt.tight_layout()

    plt.savefig(
        RESULTS / "job_pair_coverage.png",
        dpi=200
    )

    plt.close()

    # ---------------------------------------------------------
    # LABEL COMPONENTS
    # ---------------------------------------------------------

    components = [
        "skills_label",
        "seniority_label",
        "domain_label",
        "location_label",
    ]

    print("\nLABEL COMPONENT STATISTICS")
    print("-" * 50)

    print(
        train[components].describe()
    )

    # ---------------------------------------------------------
    # CORRELATION
    # ---------------------------------------------------------

    print("\nLABEL COMPONENT CORRELATION")
    print("-" * 50)

    print(
        train[
            components + ["label"]
        ].corr()["label"].sort_values(
            ascending=False
        )
    )

    # ---------------------------------------------------------
    # COMPONENT VS COMPOSITE
    # ---------------------------------------------------------

    for component in components:

        plt.figure(figsize=(7, 5))

        plt.scatter(
            train[component],
            train["label"],
            alpha=0.25,
            s=8
        )

        plt.xlabel(component)
        plt.ylabel("Composite Score")
        plt.title(
            f"{component} vs Composite Score"
        )

        plt.tight_layout()

        plt.savefig(
            RESULTS / f"{component}_vs_composite.png",
            dpi=200
        )

        plt.close()

    # ---------------------------------------------------------
    # GOLD VS TRAINING DISTRIBUTION
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        train["label"],
        bins=20,
        alpha=0.6,
        label="Training"
    )

    plt.hist(
        gold["label"],
        bins=20,
        alpha=0.6,
        label="Gold"
    )

    plt.xlabel("Composite Match Score")
    plt.ylabel("Count")
    plt.title(
        "Training vs Gold Match Score Distribution"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS / "train_vs_gold_distribution.png",
        dpi=200
    )

    plt.close()

    print("\nEDA plots saved to:")
    print(RESULTS)


if __name__ == "__main__":
    main()