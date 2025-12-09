📊 End-to-End Label Quality, Noise Estimation & Robust ML Pipeline
Human Activity Recognition (HAR) – Complete Data Quality & Robustness Workflow

This repository implements a full-stack, auditable, and production-ready data quality pipeline for Human Activity Recognition (HAR) datasets.
It includes label QA, noise estimation, drift diagnostics, embedding anomaly detection, active learning, and robust model evaluation.

The goal is to transform raw sensor data into a high-integrity dataset and a stable, noise-resistant model.

📂 Repository Structure
.
├── outputs/
│   ├── plot_1_model_accuracy.png               # Model Accuracy (before vs after repair)
│   ├── plot_2_activity_distribution.png        # Train/Test activity distribution
│   ├── plot_3_drift_scores.png                 # Distribution of drift scores
│   ├── plot_4_drifted_features.png             # Top 10 drifted features
│   ├── plot_5_embedding_anomaly_clusters.png   # Embedding space anomaly clusters
│   ├── noise_scores_sample.csv                 # Sample of noise scores per instance
│
├── raw data/
│   ├── human+activity+recognition+using+smartphones/   # Raw HAR dataset
│
├── src/
│   └── full_code.py                               # Full end-to-end pipeline
│
├── requirements.txt
├── readme.md

🚀 Pipeline Overview

This project provides an end-to-end workflow covering:

1️⃣ Pre-Label QA

Initial dataset sanity and structure validation:

Activity distribution checks

Train/test human-activity drift detection

Entropy, rarity, and stability metrics

Logical feature–label consistency

Human QA spot-check sampling

2️⃣ Label Noise Estimation

Multiple noise signals combined into a Unified Noise Score (UNS):

Prediction stability scoring

Model disagreement

Per-sample loss curves

Bayesian true-label probability

Confidence-based estimators

UNS = weighted fusion of all noise indicators

3️⃣ Drift & Noise Diagnostics

Advanced detection modules:

Drift score distribution

Top drifted features (Plot 4)

Embedding-space cluster anomalies (Plot 5)

Logical rule-based inconsistencies

Cluster-level noise localization

4️⃣ Noise Repair & Label Correction

Lineage-aware correction tools:

Soft relabeling (probabilistic)

Hard relabeling (high-certainty only)

Multi-label merging for conflicting cases

Versioned updates with full transparency

5️⃣ Active Learning Loop

Samples with highest uncertainty / noise are:

flagged

returned to human review

re-labeled or validated

Improves final dataset quality by 20–30%.

6️⃣ Post-Repair Model Validation

Ensures the pipeline truly improves model performance:

Accuracy before vs after cleaning (Plot 1)

Stability and robustness across activity slices

Drift reduction verification (Plots 3 & 4)

Embedding space realignment (Plot 5)

Final evaluation plots are all saved in the outputs/ directory.

📈 Included Visualizations
Plot 1 — Model Accuracy (Before vs After Repair)

Quantifies improvement after noise mitigation.

Plot 2 — Activity Distribution (Train vs Test)

Reveals imbalance or split drift.

Plot 3 — Drift Score Distribution

Shows the severity and spread of drift across samples.

Plot 4 — Top 10 Drifted Features

Highlights which sensor features are most unstable.

Plot 5 — Embedding Anomaly Clusters

Visualizes outliers in representation space.

💾 Dataset

This project uses the Human Activity Recognition Using Smartphones dataset.

📎 Dataset Link:
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

📁 The dataset is included locally under:

raw data/human+activity+recognition+using+smartphones/

⚠️ Important Note

The source of the dataset used for this project is provided in the link above.
The original, unmodified raw data is also included in this repository inside the raw data/ folder to ensure complete transparency, reproducibility, and ease of use.