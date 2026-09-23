# TalentGraph

### Intelligent Talent Discovery & Workforce Intelligence Engine

TalentGraph is an ML-powered **candidate–job recommendation system** that combines lexical retrieval, semantic similarity, structured candidate–job features, and Learning-to-Rank to recommend relevant jobs for a candidate profile.

Instead of relying purely on keyword matching or purely on semantic similarity, TalentGraph combines multiple signals and uses an **XGBoost Learning-to-Rank model** to produce the final ranking.

---

## Overview

Matching candidates to jobs is an information-retrieval problem involving both exact and semantic relationships.

For example, a candidate may explicitly mention:

> Python, PyTorch, Machine Learning

while a relevant job may describe:

> Deep Learning Engineer with Python and PyTorch experience.

Simple keyword search can miss semantic relationships, while semantic search alone can lose important exact-skill and structured compatibility signals.

TalentGraph therefore combines:

* **BM25** for lexical retrieval
* **MiniLM sentence embeddings** for semantic retrieval
* **FAISS** for dense vector search
* **Candidate–job compatibility features**
* **XGBoost Learning-to-Rank**
* **Deterministic recommendation explanations**

The result is a lightweight end-to-end recommendation system exposed through **FastAPI** and **Streamlit**.

---

## Architecture

```text
                  Candidate Profile
                         │
                         ▼
                Candidate Preprocessing
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
          BM25                  MiniLM Encoder
      Lexical Retrieval         Dense Embedding
             │                       │
             └───────────┬───────────┘
                         ▼
                  Retrieval Signals
                         │
                         ▼
              Candidate–Job Features
                         │
                         ▼
                 XGBoost LTR Model
                         │
                         ▼
                    Top-K Jobs
                         │
                         ▼
             Deterministic Explanation
                         │
                ┌────────┴────────┐
                ▼                 ▼
             FastAPI          Streamlit
```

---

## Key Features

* 🔎 BM25 lexical retrieval
* 🧠 MiniLM semantic embeddings
* ⚡ FAISS dense retrieval
* 🔀 Hybrid retrieval with Reciprocal Rank Fusion
* 📊 Candidate–job feature engineering
* 🏆 XGBoost Learning-to-Rank
* 💡 Deterministic recommendation explanations
* 🚀 FastAPI inference API
* 🖥️ Streamlit interactive dashboard
* 🐳 Docker deployment
* 📈 Ranking evaluation with Hit@K, MRR and nDCG

---

## Feature Engineering

The ranking model uses eight candidate–job features:

| Feature                    | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| `bm25_score`               | Lexical similarity between candidate and job              |
| `dense_score`              | Semantic embedding similarity                             |
| `skill_coverage`           | Coverage of candidate skills in the job                   |
| `role_family_match`        | Compatibility between candidate roles and job role family |
| `domain_match`             | Domain compatibility                                      |
| `experience_compatibility` | Compatibility of experience requirements                  |
| `seniority_compatibility`  | Candidate/job seniority compatibility                     |
| `location_compatibility`   | Location compatibility                                    |

The final ranking model is an **XGBoost `XGBRanker`** using the `rank:ndcg` objective.

---

## Dataset

TalentGraph uses the **Role Radar** dataset.

Approximate dataset size:

* ~2,500 jobs
* ~640 synthetic candidate profiles
* sparse human-reviewed gold judgments

The project distinguishes between:

### Training labels

Weak supervision is used to train the Learning-to-Rank model.

### Gold evaluation

A separate set of sparse human-reviewed judgments is used for final evaluation.

The final benchmark contains:

* **78 gold candidates**
* **2,082 jobs**
* **106 gold judgments**

Because the gold judgments are sparse rather than exhaustive, the results are reported as a **sparse judged gold evaluation**.

---

## Evaluation

### Sparse Judged Gold Evaluation

| Method                 |  Hit@5 |     Hit@10 |    MRR | nDCG@5 |    nDCG@10 |
| ---------------------- | -----: | ---------: | -----: | -----: | ---------: |
| BM25                   | 0.1026 |     0.1282 | 0.0929 | 0.0790 |     0.0826 |
| Dense-MiniLM           | 0.0256 |     0.0385 | 0.0195 | 0.0120 |     0.0163 |
| RRF                    | 0.0256 |     0.0897 | 0.0455 | 0.0229 |     0.0390 |
| XGBoost-LTR            | 0.0897 |     0.1026 | 0.0666 | 0.0596 |     0.0633 |
| XGBoost-LTR − Location | 0.1026 | **0.1538** | 0.0851 | 0.0720 | **0.0872** |

The ranking experiments show that lexical retrieval is a strong baseline for this dataset, while the Learning-to-Rank model provides a framework for combining lexical, semantic, and structured compatibility signals.

An ablation removing the location feature produced higher gold-set Hit@10 and nDCG@10 than the full feature set.

### Weak-Label Validation

The trained XGBoost-LTR model achieved:

**nDCG@10 = 0.8028**

on the candidate-disjoint weak-label validation split.

This number should not be directly compared with the sparse gold benchmark because the two evaluations use different labels and evaluation settings.

---

## Retrieval Methods

### BM25

BM25 provides the lexical retrieval baseline.

It is effective when exact skills, technologies, titles, and domain terminology overlap between candidates and jobs.

### Dense Retrieval

TalentGraph uses:

`sentence-transformers/all-MiniLM-L6-v2`

to encode candidate and job text into dense vectors.

FAISS is used for vector similarity search.

### RRF

Reciprocal Rank Fusion combines rankings from lexical and dense retrieval.

### Learning-to-Rank

The final ranking stage uses XGBoost to combine:

* lexical similarity
* semantic similarity
* skills
* role family
* domain
* experience
* seniority
* location

into a final ranking score.

---

## API

Start the FastAPI server:

```bash
uvicorn src.api.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

The recommendation endpoint accepts a candidate profile and returns the top matching jobs with ranking scores and deterministic compatibility information.

---

## Dashboard

Start the Streamlit interface:

```bash
streamlit run dashboard/app.py
```

The dashboard allows users to enter a candidate profile and inspect recommended jobs, including:

* job title
* company
* location
* ranking score
* matched skills
* compatibility signals

---

## Docker

Build and start the application with:

```bash
docker compose up --build
```

Services:

```text
FastAPI      → http://localhost:8000
Streamlit    → http://localhost:8501
```

---

## Project Structure

```text
TalentGraph/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── talentgraph_ltr.joblib
│   ├── bm25_index.pkl
│   └── dense_jobs.faiss
│
├── results/
│
├── src/
│   ├── data/
│   ├── retrieval/
│   ├── ranking/
│   ├── evaluation/
│   └── api/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Tech Stack

**Machine Learning**

* Python
* XGBoost
* scikit-learn
* Sentence Transformers
* PyTorch

**Information Retrieval**

* BM25
* FAISS
* Dense embeddings
* Reciprocal Rank Fusion
* Learning-to-Rank

**Backend**

* FastAPI
* Uvicorn

**Frontend**

* Streamlit

**Deployment**

* Docker
* Docker Compose

**Data**

* Pandas
* NumPy
* Parquet

---

## Running Locally

Create an environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then start the API:

```bash
uvicorn src.api.main:app --reload
```

Or launch the dashboard:

```bash
streamlit run dashboard/app.py
```

---

## Limitations

TalentGraph is a portfolio-scale recommendation system rather than a production hiring platform.

Current limitations include:

* sparse gold judgments
* relatively small evaluation set
* synthetic candidate profiles
* lightweight MiniLM semantic encoder
* inference performance that can be further optimized
* recommendation quality depends on job-corpus coverage

The system should be treated as a recommendation and information-retrieval tool, not as an automated hiring decision system.

---

## Future Improvements

Potential engineering improvements include:

* faster feature computation
* larger real-world job corpora
* richer candidate feedback signals
* online Learning-to-Rank
* vector database integration
* production monitoring
* incremental model updates

---

## Summary

TalentGraph demonstrates an end-to-end approach to intelligent talent discovery by combining **lexical search, semantic retrieval, structured compatibility features, and Learning-to-Rank** in a single deployable ML system.

The project covers the complete workflow:

```text
Data
 ↓
EDA
 ↓
Retrieval
 ↓
Feature Engineering
 ↓
Learning-to-Rank
 ↓
Evaluation
 ↓
FastAPI
 ↓
Streamlit
 ↓
Docker
```

Built as a practical **Information Retrieval + Recommendation + Machine Learning** system.