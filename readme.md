## Label Quality & Noise Estimation Pipeline for HAR Datasets
### Why this project exists

In many data projects, label quality is often assumed to be correct, especially when using datasets like Human Activity Recognition (HAR). However, label errors and inconsistencies can significantly impact model performance.

While working on this project, I asked myself a fundamental question:

Can we trust the labels in this dataset enough to use them for training a model?

This project focuses on answering that question before any modeling happens. The goal is not just model performance, but label quality, noise estimation, and data integrity.

### What problem does this project address?

When labels are assumed to be correct without proper validation:

Label noise can go unnoticed

Data drift can distort model predictions over time

Mislabeling can lead to poor model performance

Label inconsistencies can be hard to trace

This project aims to:

Evaluate label quality using multiple noise detection methods

Identify and estimate the impact of label errors

Ensure high-quality labels before proceeding with model training

### High-level approach

The pipeline is organized into distinct stages, each focused on improving dataset integrity and robustness:

Pre-Label QA

Validate dataset structure and sanity

Detect activity distribution shifts and data drift

Ensure feature-label consistency

Label Noise Estimation

Compute Unified Noise Score (UNS) using multiple signals (model disagreement, per-sample loss)

Bayesian true-label probability estimation

Confidence-based estimators for noise detection

Drift & Noise Diagnostics

Analyze drift in data and features

Detect embedding space anomalies

Identify logical inconsistencies in the dataset

Noise Repair & Label Correction

Use soft and hard relabeling techniques for noise correction

Versioned updates to maintain full transparency

Active Learning Loop

Flag and return uncertain/ambiguous samples for human review

Improve dataset quality by re-labeling and validation

Post-Repair Model Validation

Assess model stability and accuracy after label repair

Track improvements in performance and reduce drift

Validate label corrections with final evaluation plots

### Outcome

After running the pipeline on the Human Activity Recognition dataset, it was clear that label noise and data drift were present, but the pipeline successfully mitigated these issues. The final model showed an improvement in accuracy and stability, even in the presence of noisy labels and drift.

Stopping at this stage, with a high-quality dataset and robust model, was considered a success.

### What I learned

Label quality is critical for model success, and noise can undermine performance if not addressed early.

Drift detection and active learning loops can significantly improve the final model.

Transparency in the process of correcting label noise is essential for reproducibility and trust.

Data integrity must be prioritized over rushing to build a model.

### Scope and limitations

This is a learning-focused project designed for educational purposes, not a production-ready system.

The pipeline focuses on label quality and data drift, and other factors such as bias or sensor errors may need additional attention.

The methods used are tailored to HAR datasets; adaptations may be needed for other domains.

### Background

I am a Statistics graduate with a focus on data integrity, noise handling, and model robustness. I am working on projects that address real-world data issues and improve machine learning workflows, especially in terms of dataset quality and trust.

Feedback and suggestions are always welcome.

### Dataset

This project uses the Human Activity Recognition Using Smartphones dataset. The dataset can be found here:

👉 Dataset Link https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The dataset is included locally in the raw data/human+activity+recognition+using+smartphones/ folder for full transparency.
