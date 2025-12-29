## Label Quality & Noise Estimation Pipeline for HAR Datasets
## Why this project exists

In many machine learning projects, label correctness is often assumed without explicit validation.
However, even small amounts of label noise can significantly affect model stability, interpretability, and downstream decisions.

This project focuses on evaluating label quality and estimating label noise before relying on the dataset for modeling.

Rather than optimizing model performance, the primary goal is to understand whether the labels can be trusted.

### What problem does this project address?

When label quality is not explicitly assessed:

- Label noise can remain hidden

- Model performance may appear unstable or misleading

- Errors in labeling are difficult to diagnose

- Downstream modeling decisions become unreliable

This project aims to make label quality visible, measurable, and analyzable.

### High-level approach

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
