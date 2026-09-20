# TalentGraph Data

This directory contains datasets and generated data artifacts used by TalentGraph.

## Structure

- `raw/` — Original downloaded datasets. Never modify these files.
- `processed/` — Cleaned and normalized datasets.
- `embeddings/` — Generated embedding matrices and vector indexes.

## Reproducibility

Raw datasets are not committed to Git.

The data preparation pipeline should transform raw data into the canonical TalentGraph schema.

## Data leakage policy

Candidate-job pairs belonging to validation/test evaluation must not be used during training or feature fitting.

All transformations that learn parameters from data must be fitted only on the training split.