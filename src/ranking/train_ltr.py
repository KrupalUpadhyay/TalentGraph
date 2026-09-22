from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker


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

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


# Weak-label -> ordinal relevance

def relevance_grade(score):
    """
    Convert the dataset's composite score into a stable
    ordinal relevance grade.

    0: < 60
    1: 60-69
    2: 70-79
    3: 80-89
    4: >= 90
    """

    if score < 60:
        return 0

    if score < 70:
        return 1

    if score < 80:
        return 2

    if score < 90:
        return 3

    return 4


# Ranking metrics

def ranking_metrics(
    df,
    score_col,
    label_col="relevance_grade",
    k=10,
):
    """
    Evaluate ranking within each candidate group.

    Used for candidate-disjoint validation where every
    candidate-job pair has a weak relevance label.
    """

    rows = []

    for profile_id, group in df.groupby(
        "profile_id",
        sort=False,
    ):

        ranked = group.sort_values(
            score_col,
            ascending=False,
        )

        top_k = ranked.head(k)

        labels = top_k[label_col].to_numpy()

        # nDCG
        
        gains = (2 ** labels) - 1

        discounts = np.log2(
            np.arange(
                2,
                len(labels) + 2,
            )
        )

        dcg = float(
            np.sum(
                gains / discounts
            )
        )

        ideal_labels = np.sort(
            group[label_col].to_numpy()
        )[::-1][:k]

        ideal_gains = (
            2 ** ideal_labels
        ) - 1

        ideal_discounts = np.log2(
            np.arange(
                2,
                len(ideal_labels) + 2,
            )
        )

        idcg = float(
            np.sum(
                ideal_gains /
                ideal_discounts
            )
        )

        ndcg = (
            dcg / idcg
            if idcg > 0
            else 0.0
        )

        # Hit@K
        
        hit = float(
            np.any(
                labels > 0
            )
        )

        # Reciprocal Rank

        reciprocal_rank = 0.0

        for rank, label in enumerate(
            ranked[label_col].to_numpy()[:k],
            start=1,
        ):

            if label > 0:
                reciprocal_rank = 1.0 / rank
                break

        rows.append(
            (
                ndcg,
                hit,
                reciprocal_rank,
            )
        )

    if not rows:
        return {
            f"nDCG@{k}": 0.0,
            f"hit@{k}": 0.0,
            "MRR": 0.0,
            "candidates": 0,
        }

    values = np.asarray(
        rows,
        dtype=float,
    )

    return {
        f"nDCG@{k}": float(
            values[:, 0].mean()
        ),

        f"hit@{k}": float(
            values[:, 1].mean()
        ),

        "MRR": float(
            values[:, 2].mean()
        ),

        "candidates": int(
            len(values)
        ),
    }


# Group sizes for XGBoost

def make_groups(df):

    return (
        df.groupby(
            "profile_id",
            sort=False,
        )
        .size()
        .to_numpy()
        .astype(int)
    )


def make_ranker():
    """Keep full and ablation models on identical XGBoost settings."""
    return XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )


def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        DATA_DIR /
        "ranking_train.parquet"
    )

    df = pd.read_parquet(path)

    print("=" * 70)
    print("TALENTGRAPH — XGBOOST LEARNING-TO-RANK")
    print("=" * 70)

    print(
        f"\nLoaded rows: {len(df)}"
    )

    required_columns = (
        FEATURES
        + [
            "profile_id",
            "job_id",
            "label",
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

   # Convert weak labels to ordinal grades
    
    df["relevance_grade"] = (
        df["label"]
        .apply(relevance_grade)
        .astype(int)
    )

     # Candidate-disjoint split
    
    # A candidate can NEVER appear in both
    # train and validation.
    
    candidates = (
        df["profile_id"]
        .drop_duplicates()
        .to_numpy()
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, validation_indices = (
        next(
            splitter.split(
                candidates,
                groups=candidates,
            )
        )
    )

    train_candidates = set(
        candidates[train_indices]
    )

    validation_candidates = set(
        candidates[validation_indices]
    )

    train = df[
        df["profile_id"]
        .isin(train_candidates)
    ].copy()

    validation = df[
        df["profile_id"]
        .isin(validation_candidates)
    ].copy()

    # Ranking requires at least two documents per group
    
    train_counts = (
        train
        .groupby("profile_id")
        .size()
    )

    valid_train_candidates = (
        train_counts[
            train_counts >= 2
        ]
        .index
    )

    train = train[
        train["profile_id"]
        .isin(valid_train_candidates)
    ].copy()

    # Sort by candidate because XGBoost expects
    # contiguous ranking groups.
   
    train = (
        train
        .sort_values("profile_id")
        .reset_index(drop=True)
    )

    validation = (
        validation
        .sort_values("profile_id")
        .reset_index(drop=True)
    )

    
    X_train = train[FEATURES]
    y_train = (
        train["relevance_grade"]
    )

    X_validation = validation[
        FEATURES
    ]

    y_validation = (
        validation["relevance_grade"]
    )

    train_groups = make_groups(
        train
    )

    validation_groups = make_groups(
        validation
    )

    
    print("\nTraining XGBoost Ranker...")

    model = make_ranker()

    model.fit(
        X_train,
        y_train,

        group=train_groups,

        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],

        eval_group=[
            validation_groups
        ],

        verbose=False,
    )

    
    validation[
        "ltr_score"
    ] = model.predict(
        X_validation
    )

    # Baselines
    
    print("\nEvaluating validation ranking...")

    bm25_metrics = ranking_metrics(
        validation,
        "bm25_score",
    )

    dense_metrics = ranking_metrics(
        validation,
        "dense_score",
    )

    ltr_metrics = ranking_metrics(
        validation,
        "ltr_score",
    )

    # Train deployment models on all weakly labelled, gold-disjoint candidates.
    # Validation above remains the only weak-label generalization estimate.
    full_training = df.sort_values("profile_id").reset_index(drop=True)
    final_model = make_ranker()
    final_model.fit(
        full_training[FEATURES],
        full_training["relevance_grade"],
        group=make_groups(full_training),
        verbose=False,
    )

    no_location_features = [feature for feature in FEATURES if feature != "location_compatibility"]
    no_location_model = make_ranker()
    no_location_model.fit(
        full_training[no_location_features],
        full_training["relevance_grade"],
        group=make_groups(full_training),
        verbose=False,
    )

    # Save deployment models.  The validation-only model is intentionally not
    # served because it leaves 20% of the weak training data unused.

    model_path = (
        MODELS_DIR /
        "talentgraph_ltr.joblib"
    )

    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "relevance_scheme": (
                "0:<60, "
                "1:60-69, "
                "2:70-79, "
                "3:80-89, "
                "4:>=90"
            ),
        },
        model_path,
    )

    joblib.dump(
        {
            "model": no_location_model,
            "features": no_location_features,
            "ablation": "location_compatibility removed",
        },
        MODELS_DIR / "talentgraph_ltr_no_location.joblib",
    )

    # Save validation metrics
    
    metadata = {

        "train_rows": int(
            len(train)
        ),

        "validation_rows": int(
            len(validation)
        ),

        "train_candidates": int(
            train["profile_id"]
            .nunique()
        ),

        "validation_candidates": int(
            validation["profile_id"]
            .nunique()
        ),

        "features": FEATURES,

        "baseline_validation": {

            "BM25": bm25_metrics,

            "Dense-MiniLM": dense_metrics,
        },

        "XGBoost-LTR": ltr_metrics,
    }

    metrics_path = (
        RESULTS_DIR /
        "ltr_validation.json"
    )

    metrics_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "BM25",
                **bm25_metrics,
            },

            {
                "model": "Dense-MiniLM",
                **dense_metrics,
            },

            {
                "model": "XGBoost-LTR",
                **ltr_metrics,
            },
        ]
    )

    comparison.to_csv(
        RESULTS_DIR /
        "ltr_validation_metrics.csv",
        index=False,
    )

   # Feature importance
    
    importance = pd.DataFrame(
        {
            "feature": FEATURES,

            "importance":
                final_model.feature_importances_,
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance.to_csv(
        RESULTS_DIR /
        "feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nTraining rows: "
        f"{len(train)}"
    )

    print(
        f"Training candidates: "
        f"{train.profile_id.nunique()}"
    )

    print(
        f"Validation rows: "
        f"{len(validation)}"
    )

    print(
        f"Validation candidates: "
        f"{validation.profile_id.nunique()}"
    )

    print("\nValidation results:")

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nFeature importance:"
    )

    print(
        importance.to_string(
            index=False
        )
    )

    print(
        f"\nSaved model:\n"
        f"{model_path}"
    )


if __name__ == "__main__":
    main()
