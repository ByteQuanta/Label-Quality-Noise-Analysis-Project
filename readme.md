# Label Quality & Noise Estimation Pipeline for HAR Datasets
## Why this project exists

In many machine learning projects, label correctness is often assumed without explicit validation.
However, even small amounts of label noise can significantly affect model stability, interpretability, and downstream decisions.

This project focuses on evaluating label quality and estimating label noise before relying on the dataset for modeling.

Rather than optimizing model performance, the primary goal is to understand whether the labels can be trusted.

## What problem does this project address?

When label quality is not explicitly assessed:

- Label noise can remain hidden

- Model performance may appear unstable or misleading

- Errors in labeling are difficult to diagnose

- Downstream modeling decisions become unreliable

This project aims to make label quality visible, measurable, and analyzable.

## High-level approach

The pipeline is organized into analytical stages focused on diagnosing label quality:

### Pre-label sanity checks

Dataset structure and consistency validation

Class distribution inspection

Feature–label relationship checks

### Label noise analysis

Noise estimation using multiple signals (e.g. model disagreement, per-sample loss)

Confidence-based indicators for potentially noisy samples

### Drift & inconsistency diagnostics

Detection of distribution shifts

Identification of anomalous samples in feature space

Logical consistency checks across activities

### Label repair experiments

Exploration of soft and hard relabeling strategies

Versioned label updates to preserve traceability

### Post-analysis validation

Comparison of model behavior before and after label adjustments

Stability-focused evaluation rather than performance optimization

## Outcome

The analysis revealed the presence of label noise and distributional inconsistencies in the dataset.

After applying label repair strategies, model behavior became more stable and interpretable.
At this stage, stopping with a cleaner and better-understood dataset was considered a successful outcome.

## Key takeaways

- Label quality has a direct impact on model reliability

- Noise analysis is valuable even before advanced modeling

- Transparent diagnostics improve trust in data-driven workflows

- Improving dataset integrity can be more important than optimizing models

## Scope and limitations

- This project is designed for learning and analysis purposes

- Methods are exploratory rather than production-ready

- Techniques are tailored to HAR-style datasets and may require adaptation elsewhere

## Dataset
Human Activity Recognition Using Smartphones Dataset
Source: UCI Machine Learning Repository [👉 Dataset Link https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones]

The dataset is included locally for transparency and reproducibility.
