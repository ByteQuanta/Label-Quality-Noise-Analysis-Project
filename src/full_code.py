# ================
# A) PRE LABEL QA
# ================
# ================
# 1) Data Import
# ================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import random

# ================
# 1. Global Settings
# ================
random.seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')

# ================
# 2. Path Setup
# ================
base_path = os.path.expanduser('~') + '/Desktop/UCI HAR Dataset'

paths = {
    "X_train":  os.path.join(base_path, 'train', 'X_train.txt'),
    "X_test":   os.path.join(base_path, 'test',  'X_test.txt'),
    "y_train":  os.path.join(base_path, 'train', 'y_train.txt'),
    "y_test":   os.path.join(base_path, 'test',  'y_test.txt'),
    "features": os.path.join(base_path, 'features.txt'),
    "subject_train": os.path.join(base_path, 'train', 'subject_train.txt'),
    "subject_test":  os.path.join(base_path, 'test',  'subject_test.txt')
}

# ================
# 3. Load Features
# ================
features_raw = pd.read_csv(paths["features"], header=None, delim_whitespace=True)[1].values

# Duplicate feature names correction
feature_counts = {}
unique_features = []

for feat in features_raw:
    if feat in feature_counts:
        feature_counts[feat] += 1
        unique_features.append(f"{feat}_{feature_counts[feat]}")
    else:
        feature_counts[feat] = 0
        unique_features.append(feat)

# ================
# 4. Load Main Data
# ================
X_train = pd.read_csv(paths["X_train"], header=None, delim_whitespace=True, names=unique_features)
X_test  = pd.read_csv(paths["X_test"],  header=None, delim_whitespace=True, names=unique_features)

y_train = pd.read_csv(paths["y_train"], header=None, names=['Activity'])
y_test  = pd.read_csv(paths["y_test"],  header=None, names=['Activity'])

# ================
# 5. Load Subject IDs
# ================
subject_train = pd.read_csv(paths["subject_train"], header=None, names=['Subject'])
subject_test  = pd.read_csv(paths["subject_test"], header=None, names=['Subject'])

X_train['Subject'] = subject_train['Subject']
X_test['Subject']  = subject_test['Subject']

# ================
# 6. Basic Validation (Pre-QA entry check)
# ================
assert X_train.shape[1] == X_test.shape[1], "❌ Train & Test feature mismatch!"
assert len(unique_features) == X_train.shape[1] - 1, "❌ Feature count mismatch after Subject merge!"
assert y_train.shape[0] == X_train.shape[0], "❌ y_train length mismatch!"
assert y_test.shape[0]  == X_test.shape[0],  "❌ y_test length mismatch!"

print("✅ Data successfully loaded.")
print(f"Train Shape: {X_train.shape}")
print(f"Test Shape : {X_test.shape}")
print(f"# Features : {len(unique_features)}")



# ============================================================
# 2) EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
import numpy as np
import pandas as pd

print("\n==============================")
print(" EXPLORATORY DATA ANALYSIS")
print("==============================\n")

# ============================================================
# 1. Dataset Structure Overview
# ============================================================
print("=== Dataset Overview ===")
print("Train shape:", X_train.shape, " | Test shape:", X_test.shape)
print("y_train shape:", y_train.shape, " | y_test shape:", y_test.shape)

print("\nFeature count:", X_train.shape[1])
print("Unique activity count (train):", y_train['Activity'].nunique())
print("Unique activity count (test):", y_test['Activity'].nunique())


# ============================================================
# 2. Null / Missing / Zeroed Row Check
# ============================================================
print("\n=== Null / Missing / Zero Row Check ===")

# Missing values (True EDA requirement)
print("Train missing values:\n", X_train.isnull().sum().sum())
print("Test missing values:\n", X_test.isnull().sum().sum())

# Existing check (your code)
print("Train null-row-like (sum==0):", (X_train.sum(axis=1) == 0).sum())
print("Test null-row-like  (sum==0):", (X_test.sum(axis=1) == 0).sum())


# ============================================================
# 3. Activity Label Quality Check
# ============================================================
print("\n=== Activity Label Distribution ===")
print("Train Activity Counts:\n", y_train['Activity'].value_counts().sort_index())
print("\nTest Activity Counts:\n", y_test['Activity'].value_counts().sort_index())

print("\n=== Invalid Activity IDs ===")
valid_ids = [1, 2, 3, 4, 5, 6]
invalid_train = y_train[~y_train['Activity'].isin(valid_ids)]
invalid_test  = y_test[~y_test['Activity'].isin(valid_ids)]

print("Invalid Activity IDs (train):", len(invalid_train))
print("Invalid Activity IDs (test) :", len(invalid_test))


# ============================================================
# 4. Subject ID Quality Control
# ============================================================
print("\n=== Subject ID Integrity ===")

# Basic invalid subject detection
print("Train invalid Subject IDs:", X_train[X_train['Subject'] <= 0].shape[0])
print("Test invalid Subject IDs :", X_test[X_test['Subject'] <= 0].shape[0])

# Subject distribution analysis (industry-standard part of EDA)
print("\nTrain subject distribution:\n", X_train['Subject'].value_counts().sort_index())
print("\nTest subject distribution:\n", X_test['Subject'].value_counts().sort_index())


# ============================================================
# 5. All-Zero / Low-Variance Row Detection
# ============================================================
print("\n=== Zero-Norm / Low-Variance Row Detection ===")

# Calculate L2 norms excluding subject
row_norm_train = np.linalg.norm(X_train.drop(columns=['Subject']), axis=1)
row_norm_test  = np.linalg.norm(X_test.drop(columns=['Subject']), axis=1)

print("Train norm=0 rows:", (row_norm_train == 0).sum())
print("Test norm=0 rows :", (row_norm_test == 0).sum())

# Low-variance detection (true EDA check)
feature_variances = X_train.drop(columns=["Subject"]).var()
low_var_cols = feature_variances[feature_variances < 1e-6]

print("\nLow-variance feature count:", len(low_var_cols))
print("Low-variance features:\n", low_var_cols)


# ============================================================
# 6. Basic Statistical Summary (industry EDA requirement)
# ============================================================
print("\n=== Summary Statistics ===")
print(X_train.describe().T)


# ============================================================
# 7. Train–Test Feature Distribution Comparison (EDA only)
# ============================================================
print("\n=== Train–Test Feature Mean Comparison ===")

train_means = X_train.drop(columns=['Subject']).mean()
test_means  = X_test.drop(columns=['Subject']).mean()

mean_diff = (train_means - test_means).abs().sort_values(ascending=False)

print("Top 10 feature mean differences:\n", mean_diff.head(10))

print("\nEDA Completed.\n")



# ================
# 3) Missing Value Detection
# ================
# ================
# 1. Checking Missing Value Ratio
# ================
print("=== Checking Missing Value ===")
print("X_train missing values:\n", X_train.isnull().sum().sum())
print("X_test missing values:\n",  X_test.isnull().sum().sum())
print("y_train missing values:",   y_train.isnull().sum().sum())
print("y_test missing values:",    y_test.isnull().sum().sum())

# ================
# 2. Numeric type check
# ================
print("\n=== Format / Type Control ===")
non_numeric_cols_train = X_train.select_dtypes(exclude=[np.number]).columns
non_numeric_cols_test  = X_test.select_dtypes(exclude=[np.number]).columns

print("Train non-numeric columns:", list(non_numeric_cols_train))
print("Test non-numeric columns :", list(non_numeric_cols_test))

# ================
# 3. Inf value control
# ================
print("\n=== Inf / -Inf Control ===")
print("Train:", np.isinf(X_train).sum().sum())
print("Test :", np.isinf(X_test).sum().sum())



# ============================================================
# 4) Outlier Analysis (WITHOUT SUBJECT COLUMN)
# ============================================================
import numpy as np
import pandas as pd
from scipy.stats import zscore, chi2
import numpy.linalg as LA

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler

# ============================================================
# 1. Prepare data without Subject column
# ============================================================
Xtr_no_subj = X_train.drop(columns=["Subject"])
Xte_no_subj = X_test.drop(columns=["Subject"])

numeric_columns = Xtr_no_subj.select_dtypes(include=[np.number]).columns

print("Shape (train no Subject):", Xtr_no_subj.shape)
print("Shape (test no Subject):", Xte_no_subj.shape)

# ============================================================
# 2. Z-Score Outlier Detection
# ============================================================
z_train_ns = np.abs(zscore(Xtr_no_subj[numeric_columns]))
z_test_ns  = np.abs(zscore(Xte_no_subj[numeric_columns]))

Xtr_no_subj["zscore_outliers"] = (z_train_ns > 3).sum(axis=1)
Xte_no_subj["zscore_outliers"] = (z_test_ns  > 3).sum(axis=1)

# ============================================================
# 3. IQR Outlier Detection
# ============================================================
Q1 = Xtr_no_subj[numeric_columns].quantile(0.25)
Q3 = Xtr_no_subj[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

Xtr_no_subj["iqr_outliers"] = ((Xtr_no_subj[numeric_columns] < lower) | 
                               (Xtr_no_subj[numeric_columns] > upper)).sum(axis=1)

Xte_no_subj["iqr_outliers"] = ((Xte_no_subj[numeric_columns] < lower) | 
                               (Xte_no_subj[numeric_columns] > upper)).sum(axis=1)

# ============================================================
# 4. Mahalanobis Outlier Detection
# ============================================================
cov = np.cov(Xtr_no_subj[numeric_columns].values, rowvar=False)
cov_inv = LA.pinv(cov)
mean_vec = Xtr_no_subj[numeric_columns].mean().values

def mahalanobis_distance(x):
    diff = x - mean_vec
    return np.sqrt(diff @ cov_inv @ diff.T)

md_train = Xtr_no_subj[numeric_columns].apply(lambda row: mahalanobis_distance(row.values), axis=1)
md_test  = Xte_no_subj[numeric_columns].apply(lambda row: mahalanobis_distance(row.values), axis=1)

Xtr_no_subj["mahalanobis_score"] = md_train
Xte_no_subj["mahalanobis_score"] = md_test

# ============================================================
# 5. MODEL-BASED OUTLIERS (Isolation Forest + One-Class SVM)
# ============================================================

# ---- Isolation Forest ----
iso = IsolationForest(
    n_estimators=300,
    contamination='auto',
    random_state=42
)

iso.fit(Xtr_no_subj[numeric_columns])

Xtr_no_subj["iforest_outliers"] = (iso.predict(Xtr_no_subj[numeric_columns]) == -1).astype(int)
Xte_no_subj["iforest_outliers"] = (iso.predict(Xte_no_subj[numeric_columns]) == -1).astype(int)

# ---- One-Class SVM ----
oc = OneClassSVM(
    kernel="rbf",
    nu=0.05,         # typical value
    gamma="scale"
)
oc.fit(Xtr_no_subj[numeric_columns])

Xtr_no_subj["ocsvm_outliers"] = (oc.predict(Xtr_no_subj[numeric_columns]) == -1).astype(int)
Xte_no_subj["ocsvm_outliers"] = (oc.predict(Xte_no_subj[numeric_columns]) == -1).astype(int)

# ============================================================
# 6. ROBUST NORMALIZATION
# ============================================================
scaler = RobustScaler()

Xtr_scaled = pd.DataFrame(scaler.fit_transform(Xtr_no_subj[numeric_columns]), columns=numeric_columns)
Xte_scaled = pd.DataFrame(scaler.transform(Xte_no_subj[numeric_columns]), columns=numeric_columns)

# ============================================================
# 7. ADD OUTLIER SCORES BACK (FINAL CLEAN SET)
# ============================================================
add_cols = [
    "zscore_outliers",
    "iqr_outliers",
    "mahalanobis_score",
    "iforest_outliers",
    "ocsvm_outliers"
]

Xtr_clean = pd.concat([Xtr_scaled, Xtr_no_subj[add_cols]], axis=1)
Xte_clean = pd.concat([Xte_scaled, Xte_no_subj[add_cols]], axis=1)

print("\nFinal cleaned train shape:", Xtr_clean.shape)
print("Final cleaned test shape:", Xte_clean.shape)



# ================
# 5) Duplicate Analysis
# ================
# ============================================================
# 1. Deleting Outlier Marking Columns
# ============================================================
Xtr_clean_no_outliers = Xtr_clean.drop(columns=[
    "zscore_outliers", "iqr_outliers", "mahalanobis_score",
    "iforest_outliers", "ocsvm_outliers"
])

Xte_clean_no_outliers = Xte_clean.drop(columns=[
    "zscore_outliers", "iqr_outliers", "mahalanobis_score",
    "iforest_outliers", "ocsvm_outliers"
])

Xtr_clean_no_outliers['Subject'] = X_train.loc[Xtr_clean_no_outliers.index, 'Subject']
Xtr_clean_no_outliers['Activity'] = y_train.loc[Xtr_clean_no_outliers.index, 'Activity']

Xte_clean_no_outliers['Subject'] = X_test.loc[Xte_clean_no_outliers.index, 'Subject']
Xte_clean_no_outliers['Activity'] = y_test.loc[Xte_clean_no_outliers.index, 'Activity']


# ============================================================
# 2. Check the New Data Set
# ============================================================
print("\nNew Xtr_clean (Outliers removed) shape:", Xtr_clean_no_outliers.shape)
print("New Xte_clean (Outliers removed) shape:", Xte_clean_no_outliers.shape)

# ============================================================
# 3. Continue Duplicate Analysis Without Outlier Marking Columns
# ============================================================
# Duplicate Row Pre-Detection
print("\n=== Duplicate Row Pre-Detection (Outliers Removed) ===")
print("count of unique row in train set:", Xtr_clean_no_outliers.drop(columns=['Subject', 'Activity']).duplicated().sum())
print("count of unique row in test set :", Xte_clean_no_outliers.drop(columns=['Subject', 'Activity']).duplicated().sum())

# Subject + Activity based duplicate control
print("=== Subject + Activity based duplicate (Outliers Removed) ===")
dup_count_train = Xtr_clean_no_outliers.duplicated(subset=list(Xtr_clean_no_outliers.columns), keep=False).sum()
dup_count_test  = Xte_clean_no_outliers.duplicated(subset=list(Xte_clean_no_outliers.columns), keep=False).sum()

print("Train SA duplicates (Outliers Removed):", dup_count_train)
print("Test SA duplicates (Outliers Removed):", dup_count_test)

# Feature-only duplicate control (Without Subject)
print("\n=== Feature-only duplicate kontrolü (Outliers Removed) ===")
train_feats = Xtr_clean_no_outliers.drop(columns=['Subject', 'Activity'])
test_feats  = Xte_clean_no_outliers.drop(columns=['Subject', 'Activity'])

print("Train feature duplicates (Outliers Removed):", train_feats.duplicated(keep=False).sum())
print("Test feature duplicates (Outliers Removed):", test_feats.duplicated(keep=False).sum())

# ============================================================
# 4) Near-duplicate Control (Corrected – Full Similarity Only)
# ============================================================
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
from datetime import datetime

# ============================================================
# 1) Representation Normalization
# ============================================================
def normalize_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# ============================================================
# 2) PCA Compression
# ============================================================
def compressed_representation(X, dim=32):
    pca = PCA(n_components=min(dim, X.shape[1]))
    return pca.fit_transform(X)

# ============================================================
# 3) FULL cosine similarity duplicate detection  (Correct)
# ============================================================
def count_near_duplicates_full(X, threshold=0.9999):
    sim = cosine_similarity(X)

    # Upper triangular part WITHOUT diagonal
    mask = np.triu(sim > threshold, k=1)

    dup_count = np.sum(mask)

    severity_score = np.sum((sim[mask] - threshold) / (1 - threshold))

    return dup_count, severity_score, sim


# ============================================================
# 4) Duplicate Grouping (Full matrix)
# ============================================================
def duplicate_groups_full(X, threshold=0.9999, index_values=None):
    sim = cosine_similarity(X)
    n = sim.shape[0]
    visited = set()
    groups = []

    for i in range(n):
        if i in visited:
            continue

        group = set([i])
        for j in range(i + 1, n):
            if sim[i, j] > threshold:
                group.add(j)

        if len(group) > 1:
            visited |= group
            if index_values is not None:
                groups.append([index_values[g] for g in group])
            else:
                groups.append(sorted(list(group)))

    return groups


# ============================================================
# 5) Merge groups
# ============================================================
def merge_groups(groups):
    merged = []
    used = set()

    for g in groups:
        if any(x in used for x in g):
            continue

        comp = set(g)
        changed = True

        while changed:
            changed = False
            for h in groups:
                if any(x in comp for x in h):
                    old_len = len(comp)
                    comp |= set(h)
                    if len(comp) != old_len:
                        changed = True

        used |= comp
        merged.append(sorted(list(comp)))

    return merged


# ============================================================
# 6) Representative Selection (same)
# ============================================================
def select_representative_duplicate(X, group_indices, drift_scores=None):
    df = X.loc[group_indices].copy()

    nan_rate = df.isna().mean(axis=1)
    row_variance = df.var(axis=1)

    if drift_scores is not None:
        drift_vals = df.columns.map(lambda c: drift_scores.get(c, 0))
        drift_vals = np.mean(drift_vals)
        drift = pd.Series([drift_vals] * len(df), index=df.index)
    else:
        drift = pd.Series([0] * len(df), index=df.index)

    n_comp = max(1, min(8, df.shape[1], len(df)-1))
    pca = PCA(n_components=n_comp)
    norms = np.linalg.norm(pca.fit_transform(df), axis=1)
    norm_dist = pd.Series(norms, index=df.index)

    score_df = pd.DataFrame({
        "nan_rate": nan_rate,
        "variance": row_variance,
        "drift": drift,
        "pca_norm": norm_dist
    })

    for col in score_df.columns:
        score_df[col] = (score_df[col] - score_df[col].min()) / (score_df[col].max() - score_df[col].min() + 1e-9)

    score_df["final_score"] = (
        -0.25 * score_df["nan_rate"]
        -0.20 * score_df["pca_norm"]
        + 0.20 * score_df["variance"]
    )

    best_idx = score_df["final_score"].idxmax()
    return best_idx, score_df


# ============================================================
# 7) Lineage Logging
# ============================================================
def duplicate_lineage_log(merged_groups, representative_map, severity_score,
                          path_csv="duplicate_lineage.csv",
                          path_json="duplicate_lineage.json"):

    lineage_records = []
    ts = datetime.now().isoformat()

    for gid, group in enumerate(merged_groups, start=1):
        rep = representative_map[gid]

        lineage_records.append({
            "group_id": gid,
            "timestamp": ts,
            "representative": int(rep),
            "members": list(map(int, group)),
            "group_size": len(group),
            "severity_score": float(severity_score)
        })

    df_lineage = pd.DataFrame(lineage_records)
    df_lineage.to_csv(path_csv, index=False)

    with open(path_json, "w") as f:
        json.dump(lineage_records, f, indent=4)

    print("\n=== Duplicate Lineage Saved ===")
    print("CSV :", path_csv)
    print("JSON:", path_json)

    return df_lineage


# ============================================================
# 8) Main Pipeline (FULL MATRIX VERSION — CORRECT)
# ============================================================
def detect_near_duplicates(X, sample_size=500, threshold=0.9999):

    # Sampling
    Xs = X.sample(sample_size, random_state=42)
    X_values = Xs.values

    # Normalize + PCA
    X_norm = normalize_features(X_values)
    X_pca = compressed_representation(X_norm, dim=32)
    X_pca = StandardScaler().fit_transform(X_pca)

    # FULL sim matrix
    dup_count, severity_score, sim_full = count_near_duplicates_full(X_pca, threshold)

    # Grouping
    groups = duplicate_groups_full(X_pca, threshold, index_values=list(Xs.index))
    merged_groups = merge_groups(groups)

    # Representatives
    representative_map = {}
    for gid, group in enumerate(merged_groups, start=1):
        best_idx, _ = select_representative_duplicate(X.loc[group], group)
        representative_map[gid] = best_idx

    print("\n=== Near-duplicate Detection (FULL MATRIX – CORRECT) ===")
    print(f"Total duplicate collisions:        {dup_count}")
    print(f"Duplicate groups (raw):            {len(groups)}")
    print(f"Merged groups:                     {len(merged_groups)}")
    print(f"Severity-weighted score:           {severity_score:.4f}")

    return dup_count, severity_score, groups, merged_groups, representative_map

# For Run Functions and Lineage
dup_count, severity_score, groups, merged_groups, representative_map = \
    detect_near_duplicates(train_feats, sample_size=500, threshold=0.9999)

duplicate_lineage_log(merged_groups, representative_map, severity_score)



# ============================================================
# 6) Correlation Analysis (WITHOUT SUBJECT COLUMN)
# ============================================================
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# PREPARE SUBJECT-FREE TRAINING DATA
# ============================================================
Xtr_no_subj = Xtr_clean_no_outliers.drop(columns=['Subject', 'Activity'])
Xte_no_subj = Xte_clean_no_outliers.drop(columns=['Subject', 'Activity'])

print("\nShape (train no Subject):", Xtr_no_subj.shape)
print("Shape (test  no Subject):", Xte_no_subj.shape)

# =========================================================================================
# 0) FUNCTION TO COMPUTE VIF (Variance Inflation Factor)
# =========================================================================================
def calculate_vif(X_data):
    """
    Calculates the Variance Inflation Factor (VIF) for each feature in the dataset.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_data.columns
    vif_data["VIF"] = [variance_inflation_factor(X_data.values, i) for i in range(len(X_data.columns))]
    return vif_data.set_index("feature")

# =========================================================================================
# 1) FUNCTION TO COMPUTE DRIFT BETWEEN TRAIN AND TEST DATA
# =========================================================================================
def calculate_drift(X_train, X_test):
    """
    Calculates the drift between the train and test datasets based on the mean difference.
    """
    drift_scores = {}
    for col in X_train.columns:
        drift_scores[col] = abs(X_train[col].mean() - X_test[col].mean())
    return drift_scores

# =========================================================================================
# 2) FEATURE SELECTION WITH IMPROVED METRICS (IMPORTANCE, VIF, TARGET CORRELATION, DRIFT)
# =========================================================================================
def select_representative_feature(group, scores):
    """
    Selects the best feature from a correlated group using scoring metrics.
    Priority: importance > target_corr > low drift > low VIF > low NaN rate > variance
    """
    df = pd.DataFrame(scores, index=group)

    # Sorting logic (descending for positive metrics, ascending for negative metrics)
    df = df.sort_values(
        by=[
            "importance",
            "target_corr",
            # reverse order for metrics where *lower is better*
            "drift",
            "vif",
            "nan_rate",
            "variance"
        ],
        ascending=[False, False, True, True, True, False]
    )

    return df.index[0]

# =========================================================================================
# 3) MAIN FUNCTION TO SELECT CORRELATED FEATURES
# =========================================================================================
def correlation_feature_selector(
    X_train,
    X_test,
    importance_scores=None,
    target_corr=None,
    threshold=0.9
):
    """
    Feature selection with clustering + representative feature selection,
    following Google/Meta/Amazon/Stanford conventions.
    """

    # Default scores if not provided
    if importance_scores is None:
        importance_scores = {col: 0 for col in X_train.columns}
    
    if target_corr is None:
        target_corr = {col: 0 for col in X_train.columns}

    # Calculate additional metrics: Drift, VIF, NaN rate, Variance
    drift_scores = calculate_drift(X_train, X_test)
    vif_scores = calculate_vif(X_train)
    vif_scores = vif_scores['VIF'].to_dict()
    nan_rate = {col: X_train[col].isna().mean() for col in X_train.columns}
    variance = {col: X_train[col].var() for col in X_train.columns}

    # Build correlation groups (clusters)
    corr = X_train.corr().abs()
    groups = []
    visited = set()

    for col in corr.columns:
        if col in visited:
            continue

        group = set([col])
        for col2 in corr.columns:
            if col != col2 and corr.loc[col, col2] > threshold:
                group.add(col2)

        visited |= group
        groups.append(list(group))

    # Select the best feature from each group
    keep = []

    for group in groups:
        if len(group) == 1:
            keep.append(group[0])
            continue

        # Collect scores for each feature in the group
        scores = {
            "importance": {f: importance_scores[f] for f in group},
            "target_corr": {f: target_corr[f] for f in group},
            "drift": {f: drift_scores.get(f, 0) for f in group},
            "vif": {f: vif_scores.get(f, 0) for f in group},
            "nan_rate": {f: nan_rate[f] for f in group},
            "variance": {f: variance[f] for f in group},
        }

        representative = select_representative_feature(group, scores)
        keep.append(representative)

    # Return reduced dataset
    return X_train[keep], groups



# ================
# 7) Class Distribution Analysis
# ================
# ================
# 1. Basic analysis of Class Distribution 
# ================
print("=== Class Distribution Analysis ===")

print("\nTrain activity distribution:")
print(y_train['Activity'].value_counts(normalize=True) * 100)

print("\nTest activity distribution:")
print(y_test['Activity'].value_counts(normalize=True) * 100)

# ================
# 2. Plot for Class Distribution
# ================
train_class_dist = y_train['Activity'].value_counts(normalize=True) * 100
test_class_dist = y_test['Activity'].value_counts(normalize=True) * 100

plt.figure(figsize=(12, 6))

# Görselleştirme
plt.figure(figsize=(12, 6))

# Eğitim seti sınıf dağılımı
plt.subplot(1, 2, 1)
sns.barplot(x=train_class_dist.index, y=train_class_dist.values, palette="Blues_d")
plt.title("Train Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Percentage")

# Test seti sınıf dağılımı
plt.subplot(1, 2, 2)
sns.barplot(x=test_class_dist.index, y=test_class_dist.values, palette="Blues_d")
plt.title("Test Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Percentage")

plt.tight_layout()
plt.show()



# ============================================================
# 8) Data Drift & Source Drift Analysis  (Full Feature Scan)
# ============================================================
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

# =========================================================================================
# 0) FUNCTION TO COMPUTE VIF (Variance Inflation Factor)
# =========================================================================================
def calculate_vif_full(X_data):
    cols = [c for c in X_data.columns if X_data[c].var() > 0]
    X_mat = X_data[cols].fillna(0.0).values
    vif_list = []
    for i in range(len(cols)):
        try:
            vif = variance_inflation_factor(X_mat, i)
        except Exception:
            vif = np.nan
        vif_list.append(vif)
    return pd.DataFrame({"feature": cols, "VIF": vif_list}).set_index("feature")

# =========================================================================================
# 1) FUNCTION TO COMPUTE DRIFT BETWEEN TRAIN AND TEST DATA
# =========================================================================================
def calculate_drift(X_train, X_test):
    """
    Calculates the drift between the train and test datasets based on the mean difference.
    """
    drift_scores = {}
    for col in X_train.columns:
        drift_scores[col] = abs(X_train[col].mean() - X_test[col].mean())
    
    # Debugging: Print drift scores to trace
    print("Drift Scores (Mean Difference):", drift_scores)
    return drift_scores

# =========================================================================================
# 2) FUNCTION FOR COLLECTING ADVANCED DRIFT MEASURES (KS-TEST, PSI, VARIANCE, MEAN)
# =========================================================================================
def calculate_psi(expected, actual, buckets=10, eps=1e-6):
    """PSI using train-based bin edges and smoothing."""
    # compute bin edges from expected (train)
    quantiles = np.linspace(0, 1, buckets + 1)
    try:
        bin_edges = np.unique(np.quantile(expected, quantiles))
        if len(bin_edges) <= 1:
            return 0.0
    except Exception:
        bin_edges = np.linspace(np.min(expected), np.max(expected), buckets + 1)
    
    # histogram counts (use same bins)
    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)
    
    exp_perc = exp_counts / exp_counts.sum()
    act_perc = act_counts / act_counts.sum()
    
    # smoothing to avoid zeros
    exp_perc = np.clip(exp_perc, eps, None)
    act_perc = np.clip(act_perc, eps, None)
    
    psi = np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc))
    return float(psi)

def compute_advanced_drift(X_train, X_test):
    drift_report = []
    ks_stats = []
    for col in X_train.columns:
        train_vals = X_train[col].dropna()
        test_vals  = X_test[col].dropna()
        if len(train_vals) < 2 or len(test_vals) < 2:
            ks_stat, ks_p = 0.0, 1.0
        else:
            ks_stat, ks_p = ks_2samp(train_vals, test_vals)
        psi_val = calculate_psi(train_vals.values, test_vals.values)
        mean_drift = abs(train_vals.mean() - test_vals.mean())
        var_drift  = abs(train_vals.var()  - test_vals.var())
        drift_report.append({"feature": col, "ks_stat": ks_stat, "ks_p": ks_p,
                             "psi": psi_val, "mean_drift": mean_drift, "var_drift": var_drift})
        ks_stats.append(ks_p)
    drift_df = pd.DataFrame(drift_report)
    # multiple testing correction on ks p-values
    _, p_adj, _, _ = multipletests(drift_df["ks_p"].values, method='fdr_bh')
    drift_df["ks_p_adj"] = p_adj

    # normalize psi/mean/var/ks_stat to 0-1
    for c in ["psi", "mean_drift", "var_drift", "ks_stat"]:
        drift_df[c + "_norm"] = (drift_df[c] - drift_df[c].min()) / (drift_df[c].max() - drift_df[c].min() + 1e-9)
    
    # create a composite drift score (tunable weights)
    w = {"ks":0.4, "psi":0.3, "mean":0.2, "var":0.1}
    drift_df["composite_drift"] = (w["ks"]*drift_df["ks_stat_norm"] +
                                   w["psi"]*drift_df["psi_norm"] +
                                   w["mean"]*drift_df["mean_drift_norm"] +
                                   w["var"]*drift_df["var_drift_norm"])
    return drift_df.sort_values("composite_drift", ascending=False)

# =========================================================================================
# 3) FEATURE SELECTION WITH IMPROVED METRICS (IMPORTANCE, VIF, TARGET CORRELATION, DRIFT)
# =========================================================================================
def select_representative_feature(group, scores):
    """
    Selects the best feature from a correlated group using scoring metrics.
    Prioritizes importance, target correlation, low drift, low VIF, low NaN rate, variance.
    """
    df = pd.DataFrame(scores, index=group)

    # Sorting features based on different metrics, prioritizing higher importance and lower drift
    df = df.sort_values(
        by=["importance", "target_corr", "drift", "vif", "nan_rate", "variance"],
        ascending=[False, False, True, True, True, False]
    )
    
    return df.index[0]

# =========================================================================================
# 4) MAIN FUNCTION TO SELECT CORRELATED FEATURES WITH ADVANCED METRICS
# =========================================================================================
def correlation_feature_selector_with_drift(
    X_train, X_test, y_train, y_test, importance_scores=None, target_corr=None, threshold_corr=0.9, threshold_drift=0.5
):
    """
    Feature selection with clustering + representative feature selection,
    including drift evaluation and importance scores.
    """
    if importance_scores is None:
        importance_scores = {col: 0 for col in X_train.columns}
    
    if target_corr is None:
        target_corr = {col: 0 for col in X_train.columns}

    # Compute additional metrics
    drift_scores = calculate_drift(X_train, X_test)
    vif_scores = calculate_vif(X_train)
    vif_scores = vif_scores['VIF'].to_dict()
    nan_rate = {col: X_train[col].isna().mean() for col in X_train.columns}
    variance = {col: X_train[col].var() for col in X_train.columns}

    # Build correlation groups (clusters) based on the correlation matrix
    corr = X_train.corr().abs()
    groups = []
    visited = set()

    for col in corr.columns:
        if col in visited:
            continue

        group = set([col])
        for col2 in corr.columns:
            if col != col2 and corr.loc[col, col2] > threshold_corr:
                group.add(col2)

        visited |= group
        groups.append(list(group))

    # Select the representative feature from each group
    keep = [] 

    for group in groups:
        if len(group) == 1:
            keep.append(group[0])
            continue

        # Collect scores for each feature in the group
        scores = {
            "importance": {f: importance_scores[f] for f in group},
            "target_corr": {f: target_corr[f] for f in group},
            "drift": {f: drift_scores.get(f, 0) for f in group},
            "vif": {f: vif_scores.get(f, 0) for f in group},
            "nan_rate": {f: nan_rate[f] for f in group},
            "variance": {f: variance[f] for f in group},
        }

        # Select the most representative feature from the group
        representative = select_representative_feature(group, scores)
        keep.append(representative)

    # Return reduced dataset
    return X_train[keep], groups


# =======================================================
# EXAMPLE USAGE WITH Xtr_no_subj AND Xte_no_subj
# =======================================================
# Ensure Xtr_no_subj and Xte_no_subj are defined before running this
drift_df = compute_advanced_drift(Xtr_no_subj, Xte_no_subj)

# Select representative features from correlated groups based on drift and VIF
X_train_selected, correlation_groups = correlation_feature_selector_with_drift(
    Xtr_no_subj, Xte_no_subj, y_train, y_test
)

print("Selected Features after Drift & Correlation Analysis:", X_train_selected.columns)

# =======================================================
# Checking Outputs
# =======================================================
drift_scores = calculate_drift(Xtr_no_subj, Xte_no_subj)
print(drift_scores)

# Filter features with drift scores greater than 1
high_drift_features = {feature: score for feature, score in drift_scores.items() if score > 1}

# Print features with high drift scores
print("Features with High Drift (Drift > 1):")
print(high_drift_features)

# =======================================================
# Drift Analysis – Extended Inspection
# =======================================================
# =======================================================
# 1) Mean drift scores
# =======================================================
drift_scores = calculate_drift(Xtr_no_subj, Xte_no_subj)

print("\n==============================")
print("📊 Mean Drift Score Summary")
print("==============================")
print(f"Total number of features: {len(drift_scores)}")

# =======================================================
# 2) Sort drift scores descending
# =======================================================
sorted_drift = dict(sorted(drift_scores.items(), key=lambda x: x[1], reverse=True))

# =======================================================
# 3) Count high-drift features
# =======================================================
high_drift_features = {f: s for f, s in drift_scores.items() if s > 1}
print(f"Number of high-drift features (drift > 1): {len(high_drift_features)}")

print("\nHigh Drift Features (Drift > 1):")
for f, s in high_drift_features.items():
    print(f" - {f}: {s:.4f}")

# =======================================================
# 4) Show top N drifted features
# =======================================================
TOP_N = 10
print(f"\nTop {TOP_N} Features with Highest Drift:")
for idx, (f, s) in enumerate(list(sorted_drift.items())[:TOP_N]):
    print(f"{idx+1}. {f}: drift={s:.4f}")

# =======================================================
# 5) Text-based histogram visualization
# =======================================================
print("\n==============================")
print("📉 Drift Distribution (Text Histogram)")
print("==============================")

max_score = max(drift_scores.values())
for f, s in list(sorted_drift.items())[:TOP_N]:
    bar = "█" * int((s / max_score) * 40)
    print(f"{f:25} | {bar} ({s:.3f})")

# =======================================================
# 6) If advanced drift (PSI, KS) is available
# =======================================================
try:
    if 'drift_df' in locals():
        print("\n==============================")
        print("🔎 Integrating Advanced Drift Metrics (PSI, KS-Drift)")
        print("==============================")
        
        merged = drift_df.set_index("feature").copy()
        merged["mean_drift"] = merged.index.map(drift_scores)
        merged = merged.sort_values(by="ks_drift", ascending=False)

        print("\nTop features by KS Drift:")
        print(merged[["ks_drift", "psi", "mean_drift"]].head(10))

except Exception as e:
    print("\nAdvanced drift metrics could not be merged:", e)

# =======================================================
# Aligning X_test with the selected features
# =======================================================
final_features = X_train_selected.columns

X_test_selected = Xte_no_subj[final_features]

print("X_test_selected shape:", X_test_selected.shape)
print("X_train_selected shape:", X_train_selected.shape)
print("Final Feature List:", list(final_features))



# =======================================================
# B) Label Quality
# =======================================================
# 1) Evaluating Label Quality
# =======================================================

import numpy as np
import pandas as pd
from scipy.stats import entropy
from collections import Counter

# =======================================================
# Ensure labels are 1D
# =======================================================
y_train = np.ravel(y_train)
y_test  = np.ravel(y_test)

# =======================================================
# 1. Entropy Analysis
# =======================================================
def label_entropy(y):
    counts = np.array(list(Counter(y).values()))
    probs = counts / counts.sum()
    return entropy(probs)

print("Train Label Entropy:", label_entropy(y_train))
print("Test  Label Entropy:", label_entropy(y_test))


# =======================================================
# 2. Rare-Class Noise Risk
# =======================================================
rare_threshold = 0.01  # Less than 1% risk
class_freq = pd.Series(y_train).value_counts(normalize=True)
rare_classes = class_freq[class_freq < rare_threshold]

print("\nRare classes (high noise risk):")
print(rare_classes)


# =======================================================
# 3. Train vs Test Unique Label Consistency
# =======================================================
train_unique = set(y_train)
test_unique  = set(y_test)

print("\nClasses in TRAIN but not TEST:", train_unique - test_unique)
print("Classes in TEST but not TRAIN:", test_unique - train_unique)


# =======================================================
# 4. Label Adjacency Anomaly Detection
# =======================================================
unique_classes = np.unique(y_train)
class_positions = {c: i for i, c in enumerate(unique_classes)}

diffs = []
for i in range(1, len(y_train)):
    diffs.append(abs(class_positions[y_train[i]] - class_positions[y_train[i-1]]))

print("\nAverage label adjacency jump:", np.mean(diffs))



# =======================================================
# 2) Label Noise Estimation (Final Version)
# =======================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold

# =======================================================
# 1. Model Train + Probability Predictions
# =======================================================
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_selected, y_train)

train_proba = clf.predict_proba(X_train_selected)
train_pred  = clf.predict(X_train_selected)

test_proba = clf.predict_proba(X_test_selected)
test_pred  = clf.predict(X_test_selected)

# =======================================================
#  IMPORTANT: map labels → model class indices
# =======================================================
class_to_index = {c: i for i, c in enumerate(clf.classes_)}
y_train_mapped = np.array([class_to_index[y] for y in y_train])

# =======================================================
# 2. Prediction Stability
# =======================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
stability_scores = np.zeros(len(y_train))

for t_idx, v_idx in kf.split(X_train_selected):
    X_tr = X_train_selected.iloc[t_idx]
    X_val = X_train_selected.iloc[v_idx]
    y_tr  = y_train[t_idx]

    clf_cv = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_cv.fit(X_tr, y_tr)

    val_pred = clf_cv.predict_proba(X_val)
    stability_scores[v_idx] = np.max(val_pred, axis=1)

Prediction_Stability_noise_score = 1 - stability_scores

# =======================================================
# 3. Model Disagreement
# =======================================================
models = [
    RandomForestClassifier(n_estimators=200, random_state=42),
    ExtraTreesClassifier(n_estimators=200, random_state=42),
    GradientBoostingClassifier()
]

proba_list = []
for m in models:
    m.fit(X_train_selected, y_train)
    proba_list.append(m.predict_proba(X_train_selected))

proba_stack = np.stack(proba_list)
Model_Disagreement_noise_score = proba_stack.var(axis=0).max(axis=1)

# =======================================================
# 4. Loss-Based Noise
# =======================================================
loss_per_sample = -np.log(train_proba[np.arange(len(y_train)), y_train_mapped])
Loss_Based_noise_score = (loss_per_sample - loss_per_sample.min()) / (loss_per_sample.max() - loss_per_sample.min())

# =======================================================
# 5. Bayesian Confidence
# =======================================================
p_true = train_proba[np.arange(len(y_train)), y_train_mapped]
bayesian_confidence_noise_score = 1 - p_true

# =======================================================
# 6. Flag + Continuous Score
# =======================================================
confidence_threshold = 0.5
cleanlab_flag = (p_true < confidence_threshold).astype(int)

# === CleanLab continuous noise score ===
try:
    from cleanlab.internal.multiclass_utils import get_normalized_probs
    probs_norm = get_normalized_probs(train_proba)
    cleanlab_continuous = 1 - probs_norm[np.arange(len(y_train)), y_train_mapped]

except Exception:
    # fallback: treat the binary flag as continuous
    cleanlab_continuous = cleanlab_flag.astype(float)

# =======================================================
# 7. Final Unified Noise Score
# =======================================================
noise_df = pd.DataFrame({
    "Prediction_Stability_noise_score": Prediction_Stability_noise_score,
    "Model_Disagreement_score": Model_Disagreement_noise_score,
    "Loss_Based_noise_score": Loss_Based_noise_score,
    "bayesian_confidence_noise_score": bayesian_confidence_noise_score,
    "cleanlab_flag": cleanlab_flag,
    "cleanlab_continuous_score": cleanlab_continuous
})

# **Including Continuous CleanLab, average**
noise_df["final_noise_score"] = noise_df[
    [
        "Prediction_Stability_noise_score",
        "Model_Disagreement_score",
        "Loss_Based_noise_score",
        "bayesian_confidence_noise_score",
        "cleanlab_continuous_score"
    ]
].mean(axis=1)

# =======================================================
# 8. FINAL NOISE REPORT
# =======================================================
noise_df_formatted = noise_df.copy()
noise_df_formatted["sample_id"] = noise_df_formatted.index

# Columns order
noise_df_formatted = noise_df_formatted[
    [
        "sample_id",
        "Prediction_Stability_noise_score",
        "Model_Disagreement_score",
        "Loss_Based_noise_score",
        "bayesian_confidence_noise_score",
        "cleanlab_flag",
        "cleanlab_continuous_score",
        "final_noise_score"
    ]
]

# Normalize final score
noise_df_formatted["final_noise_score"] = (
    (noise_df_formatted["final_noise_score"] - noise_df_formatted["final_noise_score"].min())
    / (noise_df_formatted["final_noise_score"].max() - noise_df_formatted["final_noise_score"].min())
)

# Most noisy 20 sample
top_noisy_samples = noise_df_formatted.sort_values("final_noise_score", ascending=False).head(20)

top_noisy_samples



# =======================================================
# 3) NOISE DIAGNOSTICS
# =======================================================
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Existing from Section B
X = X_train_selected.copy()
y = y_train.copy()
noise_scores = noise_df["final_noise_score"].values


# =======================================================
# 1. Class-Level Noise Concentration
# =======================================================
class_noise_df = pd.DataFrame({
    "label": y,
    "noise": noise_scores
})

class_noise_summary = class_noise_df.groupby("label")["noise"].mean().sort_values(ascending=False)

print("\n===== CLASS-LEVEL NOISE CONCENTRATION =====")
print(class_noise_summary)


# =======================================================
# 2. Feature-Space Noise Clustering
# =======================================================
kmeans = KMeans(n_clusters=8, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X)

cluster_noise = (
    pd.DataFrame({"cluster": cluster_labels, "noise": noise_scores})
    .groupby("cluster")["noise"]
    .mean()
    .sort_values(ascending=False)
)

print("\n===== CLUSTER-LEVEL NOISE =====")
print(cluster_noise)


# =======================================================
# 3. Embedding-Space Outlier Detection
# =======================================================
# dynamic PCA components:
n_components = min(10, X.shape[1] - 1)

pca = PCA(n_components=n_components, random_state=42)
X_embed = pca.fit_transform(X)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
lof.fit(X_embed)

# Correct outlier score
lof_scores = -lof.negative_outlier_factor_

embedding_noise_df = pd.DataFrame({
    "lof_score": lof_scores,
    "noise_score": noise_scores
})

print("\n===== PCA–EMBEDDING OUTLIER OVERLAP =====")
print(embedding_noise_df.corr())


# =======================================================
# 4. Label–Feature Rule Violation Detector
# =======================================================
from sklearn.tree import DecisionTreeClassifier

rule_model = DecisionTreeClassifier(
    max_depth=6,       # dynamic, more expressive
    min_samples_leaf=5,
    random_state=42
)
rule_model.fit(X, y)

rule_pred = rule_model.predict(X)
rule_violations = (rule_pred != y).astype(int)

rule_violation_df = pd.DataFrame({
    "label": y,
    "rule_violation": rule_violations,
    "noise_score": noise_scores
})

print("\n===== RULE-BASED LABEL VIOLATIONS =====")
print(
    rule_violation_df.groupby("label")[["rule_violation", "noise_score"]].mean()
)


# =======================================================
# 5. Combine Diagnostics into a Unified Dashboard
# =======================================================
diagnostics_df = pd.DataFrame({
    "true_label": y,
    "cluster": cluster_labels,
    "noise_score": noise_scores,
    "rule_violation": rule_violations,
    "lof_outlier": lof_scores,
})


# Normalize all components
scaler = MinMaxScaler()
diagnostics_df[["noise_score", "rule_violation", "lof_outlier"]] = scaler.fit_transform(
    diagnostics_df[["noise_score", "rule_violation", "lof_outlier"]]
)

# FINAL SUSPICIOUS SCORE
diagnostics_df["suspicious_score"] = (
    diagnostics_df["noise_score"] * 0.5 +
    diagnostics_df["lof_outlier"] * 0.3 +
    diagnostics_df["rule_violation"] * 0.2
)

print("\n===== TOP 20 SUSPICIOUS SAMPLES (Unified Diagnostics) =====")
print(
    diagnostics_df.sort_values("suspicious_score", ascending=False).head(20)
)



# ============================================================
# 4) DATA LINEAGE & SENSOR PROVENANCE
# ============================================================
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
import json

# ============================================================
# 1) UCI HAR SENSOR METADATA
# ============================================================

UCI_SENSOR_METADATA = {
    "sensor_type": "accelerometer + gyroscope",
    "sensor_brand": "Samsung",
    "sensor_model": "Galaxy S II embedded IMU",
    "sampling_rate_hz": 50,
    "window_size_samples": 128,
    "window_length_seconds": 2.56,
    "data_source": "UCI HAR Smartphone Dataset",
    "collection_protocol": "Fixed waist-mounted smartphone capturing body acceleration & gyro",
}

# ============================================================
# 2) LINEAGE CONFIG (Annotator / Label Info)
# ============================================================

LINEAGE_CONFIG = {
    "annotator_id": "UCI_AUTO_LABEL",              # dataset'te insan annotator yok
    "annotation_tool_version": "HAR-v1.0",
    "guideline_version": "UCI-HAR-STANDARD",
    "revision_count": 0,
    "agreement_score": 1.0,                        # ground truth verified
    "annotation_source": "pre-verified activity labels"
}

# ============================================================
# 3) DEVICE ID HASHING (No device ID in UCI HAR)
# ============================================================
def hash_string(x: str):
    return hashlib.sha256(x.encode()).hexdigest()[:16]

device_hash = hash_string("uci_har_device")  # sabit, anonim cihaz ID

# ============================================================
# 4) Build Lineage Table for ALL SAMPLES
# ============================================================
def build_lineage_table(X, y, noise_scores=None):
    """
    X  → feature dataframe (train or test)
    y  → labels (np array or pd series)
    noise_scores → B bölümündeki final_noise_score (opsiyonel)
    """
    if noise_scores is None:
        noise_scores = np.zeros(len(X))

    sample_ids = list(X.index)

    lineage_records = []

    timestamp_now = datetime.now().isoformat()

    for i, sid in enumerate(sample_ids):
        lineage_records.append({
            # -------------------------------------------------------
            # BASIC IDENTIFICATION
            # -------------------------------------------------------
            "sample_id": int(sid),
            "label": int(y[i]),

            # -------------------------------------------------------
            # SENSOR METADATA
            # -------------------------------------------------------
            "sensor_type": UCI_SENSOR_METADATA["sensor_type"],
            "sensor_brand": UCI_SENSOR_METADATA["sensor_brand"],
            "sensor_model": UCI_SENSOR_METADATA["sensor_model"],
            "sampling_rate_hz": UCI_SENSOR_METADATA["sampling_rate_hz"],
            "window_size_samples": UCI_SENSOR_METADATA["window_size_samples"],
            "window_length_seconds": UCI_SENSOR_METADATA["window_length_seconds"],
            "data_source": UCI_SENSOR_METADATA["data_source"],
            "collection_protocol": UCI_SENSOR_METADATA["collection_protocol"],

            # -------------------------------------------------------
            # LINEAGE + LABEL METADATA
            # -------------------------------------------------------
            "annotator_id": LINEAGE_CONFIG["annotator_id"],
            "annotation_timestamp": timestamp_now,
            "annotation_tool_version": LINEAGE_CONFIG["annotation_tool_version"],
            "guideline_version": LINEAGE_CONFIG["guideline_version"],
            "revision_count": LINEAGE_CONFIG["revision_count"],
            "agreement_score": LINEAGE_CONFIG["agreement_score"],
            "annotation_source": LINEAGE_CONFIG["annotation_source"],

            # -------------------------------------------------------
            # NOISE + QUALITY (From Section B)
            # -------------------------------------------------------
            "noise_score": float(noise_scores[i]),

            # -------------------------------------------------------
            # DEVICE HASH (anonymized)
            # -------------------------------------------------------
            "device_hash": device_hash
        })

    return pd.DataFrame(lineage_records)

# ============================================================
# 5) CREATE TRAIN & TEST LINEAGE TABLES
# ============================================================

train_lineage = build_lineage_table(
    X_train_selected, 
    y_train, 
    noise_scores=noise_df["cleanlab_continuous_score"]  # best noise metric
)

test_lineage = build_lineage_table(
    X_test_selected,
    y_test,
    noise_scores=np.zeros(len(y_test))            # noise is unknown in the test
)

# ============================================================
# 6) SAVE ARTIFACTS
# ============================================================

train_lineage.to_csv("train_lineage.csv", index=False)
test_lineage.to_csv("test_lineage.csv", index=False)

with open("sensor_metadata.json", "w") as f:
    json.dump(UCI_SENSOR_METADATA, f, indent=4)

print("=== LINEAGE + SENSOR PROVENANCE TABLES SAVED ===")
print("train_lineage.csv")
print("test_lineage.csv")
print("sensor_metadata.json")


# ============================================================
# E) GUIDELINE DRIFT ANALYSIS
# ============================================================
# ============================================================
# Create lineage_train when missing
# ============================================================

import numpy as np
import pandas as pd

# If noise_df exists, use its final noise score
if "noise_df" in globals():
    noise_scores = noise_df["final_noise_score"].values
else:
    noise_scores = np.zeros(len(y_train))

# ============================================================
# 1) Synthetic annotators (UCI HAR has no annotators)
# ============================================================
num_annotators = 6
annotators = [f"annotator_{i}" for i in range(1, num_annotators+1)]

annotator_ids = np.random.choice(
    annotators, 
    size=len(y_train), 
    p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
)

# ============================================================
# 2) Guideline version simulation
# ============================================================

guidelines = ["v1.0", "v2.0", "v3.0"]

# Time drift: early samples → v1, mid → v2, late → v3
guideline_version = np.where(
    np.arange(len(y_train)) < len(y_train)*0.33, "v1.0",
    np.where(
        np.arange(len(y_train)) < len(y_train)*0.66, "v2.0", 
        "v3.0"
    )
)

# ============================================================
# 3) Fake annotation timestamps (chronological)
# ============================================================
timestamps = pd.date_range(
    start="2020-01-01", 
    periods=len(y_train), 
    freq="H"
)

# ============================================================
# 4) Revision IDs (none for HAR, so zeros)
# ============================================================
revision_id = np.zeros(len(y_train), dtype=int)

# ============================================================
# 5) Device / Sensor metadata (UCI HAR style)
# ============================================================
device_ids = np.random.choice(
    ["SamsungS2", "SamsungS3"], 
    size=len(y_train),
    p=[0.6, 0.4]
)

sampling_rate = np.random.choice(
    [50, 100], 
    size=len(y_train),
    p=[0.8, 0.2]
)

axis_config = np.random.choice(
    ["standard", "rotated_5deg", "rotated_10deg"],
    size=len(y_train),
    p=[0.7, 0.2, 0.1]
)

sensor_noise = np.random.uniform(0.0, 0.15, size=len(y_train))

# ============================================================
# 6) Combine into lineage table
# ============================================================
lineage_train = pd.DataFrame({
    "sample_id": np.arange(len(y_train)),
    "guideline_version": guideline_version,
    "annotator_id": annotator_ids,
    "timestamp": timestamps,
    "true_label": y_train,
    "revision_id": revision_id,
    "device_id": device_ids,
    "sampling_rate_hz": sampling_rate,
    "axis_config": axis_config,
    "sensor_noise_std": sensor_noise,
    "noise_score": noise_scores
})

print("✓ lineage_train created successfully")
print(lineage_train.head())



# ===========================
# F) ACTIVE LEARNING LOOP
# ===========================

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# ---------------------------------------
# 0) Choose model from models list
# ---------------------------------------

if "models" not in globals():
    raise RuntimeError("The list of models could not be found. Please load the models.")

# Ilk modeli (RandomForest) aktif öğrenme modeli olarak kullanıyoruz
model = models[0]

# Eğer model daha önce fit edilmediyse, fit et
if not hasattr(model, "classes_"):
    print("Model daha önce eğitilmemiş — şimdi eğitiliyor...")
    model.fit(X_train_selected, y_train)
else:
    print("Eğitilmiş model bulundu — yeniden eğitim yapılmadı.")

# ---------------------------------------
# 1) UNCERTAINTY SCORES
# ---------------------------------------
probs = model.predict_proba(X_train_selected)

least_conf = 1 - probs.max(axis=1)
entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)

losses = np.array([
    log_loss([y_train[i]], [probs[i]], labels=model.classes_)
    for i in range(len(y_train))
])

# ---------------------------------------
# 2) ANNOTATOR DISAGREEMENT
# ---------------------------------------
if "annotator_id" not in lineage_train.columns:
    lineage_train["annotator_id"] = np.random.choice(
        [f"annotator_{i}" for i in range(1,6)],
        size=len(lineage_train)
    )

annotator_ids = lineage_train["annotator_id"]
annotator_counts = annotator_ids.map(annotator_ids.value_counts())
annotator_disagreement = annotator_counts / annotator_counts.max()

# ---------------------------------------
# 3) CLEANLAB SCORE
# ---------------------------------------
if "noise_df" in globals() and "cleanlab_continuous" in noise_df.columns:
    cleanlab_score = noise_df["cleanlab_continuous"].values
else:
    cleanlab_score = np.zeros(len(y_train))

# ---------------------------------------
# 4) UNIFIED ACTIVE SCORE
# ---------------------------------------
active_score = (
    0.40 * (least_conf / (least_conf.max() + 1e-12)) +
    0.30 * (losses / (losses.max() + 1e-12)) +
    0.20 * annotator_disagreement +
    0.10 * (cleanlab_score / (cleanlab_score.max() + 1e-12))
)

active_df = pd.DataFrame({
    "sample_id": np.arange(len(y_train)),
    "true_label": y_train,
    "least_conf": least_conf,
    "entropy": entropy,
    "loss": losses,
    "annotator_disagreement": annotator_disagreement,
    "cleanlab_score": cleanlab_score,
    "active_score": active_score
})

# ---------------------------------------
# 5) TOP-1% SAMPLE CHOOSING
# ---------------------------------------
TOP_RATIO = 0.01
top_k = max(1, int(len(active_df) * TOP_RATIO))

human_review = active_df.sort_values("active_score", ascending=False).head(top_k)

print("\n=== Samples Selected for Human Review ===")
print(human_review[["sample_id", "active_score", "least_conf", "loss"]])

# ---------------------------------------
# 6) SIMULATE HUMAN ANNOTATION
# ---------------------------------------
np.random.seed(42)
new_labels_sim = []

for _, row in human_review.iterrows():
    old_label = row["true_label"]
    if np.random.rand() < 0.85:
        new_labels_sim.append(old_label)
    else:
        choices = [c for c in model.classes_ if c != old_label]
        new_labels_sim.append(np.random.choice(choices))

# ---------------------------------------
# 7) LABEL UPDATE
# ---------------------------------------
for df_idx, corrected_label in zip(human_review.index, new_labels_sim):
    y_train[df_idx] = corrected_label
    lineage_train.loc[df_idx, "true_label"] = corrected_label
    lineage_train.loc[df_idx, "revision_id"] = lineage_train.get("revision_id", pd.Series(0)).iloc[df_idx] + 1
    lineage_train.loc[df_idx, "annotation_timestamp"] = pd.Timestamp.now()

print(f"\nActive learning loop completed — {top_k} sample updated.")
print("Modeli yeniden eğitmek için: model.fit(X_train_selected, y_train)")



# =========================
# G) NOISE REPAIR (AUTO-CORRECTION)
# =========================
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.metrics import accuracy_score
from copy import deepcopy

# ---------- Safety checks ----------
required = ["X_train_selected", "y_train", "X_test_selected", "y_test", "lineage_train"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(f"Missing required objects in environment: {missing} - please prepare them before running repair.")

# ---------- Configurable thresholds ----------
NOISE_CANDIDATE_THRESHOLD = 0.6   # final_noise_score above -> candidate for repair
HARD_RELABEL_PROB = 0.95          # model must be >= this to hard relabel
SOFT_ALPHA = 0.7                  # weighting for soft relabel: alpha * model_proba + (1-alpha) * one_hot(original)
MULTI_TOPK = 3                    # keep top-K probabilities for multi-label merging

# ---------- Resolve model(s) / ensemble ----------
# Prefer 'models' list (ensemble). If not present, try single 'model'.
ensemble_models = None
if "models" in globals() and isinstance(models, (list, tuple)) and len(models) > 0:
    ensemble_models = models
elif "model" in globals():
    ensemble_models = [model]
else:
    raise RuntimeError("No model(s) found. Ensure `models` or `model` exist in the environment.")

# ---------- Build ensemble probability matrix on train ----------
def ensemble_proba_matrix(X):
    # returns (n_samples, n_classes) average prob across ensemble
    proba_list = []
    for m in ensemble_models:
        # if model not fitted, fit quickly on current labels
        if not hasattr(m, "classes_"):
            m.fit(X_train_selected, y_train)
        proba_list.append(m.predict_proba(X))
    stacked = np.stack(proba_list, axis=0)  # (n_models, n_samples, n_classes)
    mean_proba = np.mean(stacked, axis=0)
    return mean_proba

print("Computing ensemble probabilities on training set...")
train_ensemble_proba = ensemble_proba_matrix(X_train_selected)
train_pred_labels = np.array([m.predict(X_train_selected) for m in ensemble_models]).T  # (n_samples, n_models)

# map model.classes_ ordering (assume same across ensemble; if not, align)
classes = ensemble_models[0].classes_
class_to_index = {c: i for i, c in enumerate(classes)}
n_classes = len(classes)

# ---------- Identify candidate noisy samples ----------
if "final_noise_score" in globals().get("noise_df", pd.DataFrame()).columns:
    scores_all = noise_df["final_noise_score"].values
else:
    # fallback: if no noise_df, compute simple uncertainty by 1-max_proba
    scores_all = 1.0 - train_ensemble_proba.max(axis=1)

candidates_mask = scores_all > NOISE_CANDIDATE_THRESHOLD
candidate_indices = np.where(candidates_mask)[0]
print(f"Found {len(candidate_indices)} candidate samples (threshold {NOISE_CANDIDATE_THRESHOLD}).")

# ---------- Backup original labels for before/after comparison ----------
y_train_before = deepcopy(y_train)

# ---------- Prepare lineage updates collector ----------
if "revision_id" not in lineage_train.columns:
    lineage_train["revision_id"] = 0

repairs = []  # collect records of repair actions

# ---------- 1) HARD RELABELING ----------
# Replace label when ensemble strongly prefers a different class with high confidence
print("Running HARD relabeling step...")
hard_relabels = []
for idx in candidate_indices:
    orig_label = y_train[idx]
    proba = train_ensemble_proba[idx]
    top_idx = int(np.argmax(proba))
    top_prob = float(proba[top_idx])
    pred_label = classes[top_idx]
    # only relabel if model is confident and disagrees with original
    if (top_prob >= HARD_RELABEL_PROB) and (pred_label != orig_label):
        # apply hard relabel
        y_train[idx] = pred_label
        lineage_train.at[idx, "true_label"] = pred_label
        lineage_train.at[idx, "revision_id"] = lineage_train.at[idx, "revision_id"] + 1
        lineage_train.at[idx, "annotation_timestamp"] = pd.Timestamp.now()
        repairs.append({
            "sample_id": int(lineage_train.at[idx, "sample_id"]) if "sample_id" in lineage_train.columns else int(idx),
            "method": "hard_relabel",
            "original_label": orig_label,
            "new_label": pred_label,
            "confidence": top_prob,
            "timestamp": datetime.now().isoformat()
        })
        hard_relabels.append(idx)

print(f"Hard relabels applied: {len(hard_relabels)}")

# ---------- 2) SOFT RELABELING ----------
# For remaining candidates, build soft targets as weighted mixture
print("Running SOFT relabeling step...")
soft_updated = []
for idx in candidate_indices:
    if idx in hard_relabels:
        continue
    orig_label = y_train_before[idx]  # use original pre-hard label as anchor
    # ensemble proba vector
    proba = train_ensemble_proba[idx]  # length n_classes
    one_hot_orig = np.zeros_like(proba)
    if orig_label in class_to_index:
        one_hot_orig[class_to_index[orig_label]] = 1.0
    # build soft target
    soft_target = SOFT_ALPHA * proba + (1.0 - SOFT_ALPHA) * one_hot_orig
    # store soft target in lineage (as JSON compact) and in memory mapping
    lineage_train.at[idx, "soft_target"] = json.dumps({
        "classes": list(map(int, classes)),
        "probs": soft_target.tolist()
    })
    lineage_train.at[idx, "revision_id"] = lineage_train.at[idx, "revision_id"] + 1
    lineage_train.at[idx, "annotation_timestamp"] = pd.Timestamp.now()
    repairs.append({
        "sample_id": int(lineage_train.at[idx, "sample_id"]) if "sample_id" in lineage_train.columns else int(idx),
        "method": "soft_relabel",
        "original_label": orig_label,
        "soft_target_top3": [(classes[i], float(p)) for i, p in enumerate(soft_target.argsort()[-3:][::-1])],
        "timestamp": datetime.now().isoformat()
    })
    soft_updated.append(idx)

print(f"Soft relabels prepared (soft targets stored): {len(soft_updated)}")

# ---------- 3) PROBABILISTIC MULTI-LABEL MERGING ----------
# Keep ensemble prob distribution and save top-K for each candidate
print("Running PROBABILISTIC multi-label merging (storing top-K distributions)...")
multi_updated = []
for idx in candidate_indices:
    proba = train_ensemble_proba[idx]
    topk_idx = np.argsort(proba)[-MULTI_TOPK:][::-1]
    multi = [(classes[i], float(proba[i])) for i in topk_idx]
    lineage_train.at[idx, "probabilistic_labels"] = json.dumps(multi)
    lineage_train.at[idx, "revision_id"] = lineage_train.at[idx, "revision_id"] + 1
    lineage_train.at[idx, "annotation_timestamp"] = pd.Timestamp.now()
    repairs.append({
        "sample_id": int(lineage_train.at[idx, "sample_id"]) if "sample_id" in lineage_train.columns else int(idx),
        "method": "probabilistic_merge",
        "topk": multi,
        "timestamp": datetime.now().isoformat()
    })
    multi_updated.append(idx)

print(f"Probabilistic multi-label records written: {len(multi_updated)}")

# ---------- Save repair log artifact ----------
repairs_df = pd.DataFrame(repairs)
repairs_df.to_csv("repair_actions_log.csv", index=False)
print("Repair actions saved to repair_actions_log.csv")

# ---------- POST-REPAIR VALIDATION ----------
print("Running post-repair validation: retrain model and compare on test set...")

# Use a fresh copy of the first model in ensemble for retrain
retrain_model = deepcopy(ensemble_models[0])
retrain_model.fit(X_train_selected, y_train)
y_test_pred_before = ensemble_models[0].predict(X_test_selected)  # before using retrain - original model[0]
acc_before = accuracy_score(y_test, y_test_pred_before)

y_test_pred_after = retrain_model.predict(X_test_selected)
acc_after = accuracy_score(y_test, y_test_pred_after)

print(f"Test Accuracy BEFORE repair (model[0]): {acc_before:.4f}")
print(f"Test Accuracy AFTER repair  (retrained): {acc_after:.4f}")

# Save lineage table with repairs
lineage_train.to_csv("lineage_train_post_repair.csv", index=False)
repairs_df.to_csv("repair_actions_log.csv", index=False)

print("Lineage table saved to lineage_train_post_repair.csv")
print("G) Noise repair completed.")



# ===============================================================
# H) POST-REPAIR VALIDATION (Google/Amazon Production QA)
# ===============================================================

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

print("\n================ POST-REPAIR VALIDATION ================\n")

# -----------------------------------------------------------
# 1) BEFORE METRICS (Repair öncesi)
# -----------------------------------------------------------
try:
    before_acc
    before_f1
except NameError:
    print("Saving BEFORE metrics first time...")

    before_acc  = accuracy_score(y_test, model.predict(X_test_selected))
    before_f1   = f1_score(y_test, model.predict(X_test_selected), average="macro")
    before_cm   = confusion_matrix(y_test, model.predict(X_test_selected))

    print(f"BEFORE Accuracy: {before_acc:.4f}")
    print(f"BEFORE F1-macro: {before_f1:.4f}")


# -----------------------------------------------------------
# 2) MODEL RETRAIN (after repair)
# -----------------------------------------------------------

print("\nRetraining model on repaired labels...")

# y_train_repaired zaten G modülünden geliyor
# Eğer yoksa hata fırlatmasın diye fallback:
try:
    _ = y_train_repaired
except NameError:
    print("WARNING: y_train_repaired not found → using original y_train")
    y_train_repaired = y_train.copy()

model.fit(X_train_selected, y_train_repaired)

print("Model retrained on repaired labels.")


# -----------------------------------------------------------
# 3) AFTER METRICS (Repair sonrası)
# -----------------------------------------------------------

y_pred_after = model.predict(X_test_selected)

after_acc = accuracy_score(y_test, y_pred_after)
after_f1  = f1_score(y_test, y_pred_after, average="macro")
after_cm  = confusion_matrix(y_test, y_pred_after)

print(f"\nAFTER Accuracy: {after_acc:.4f}")
print(f"AFTER F1-macro: {after_f1:.4f}")

# -----------------------------------------------------------
# 4) METRIC LIFT (Kurumsal Standart)
# -----------------------------------------------------------

acc_lift = after_acc - before_acc
f1_lift  = after_f1  - before_f1

print("\n========== METRIC LIFT ==========")
print(f"Accuracy Lift: {acc_lift:+.4f}")
print(f"F1-macro Lift: {f1_lift:+.4f}")

# -----------------------------------------------------------
# 5) Stability / Regression Check
# Amazon & Google’da zorunlu
# -----------------------------------------------------------

print("\n========== STABILITY / REGRESSION CHECK ==========")

def check_regression(before, after, name, tolerance=0.01):
    """
    Eğer performans düşüşü %1’den fazlaysa alarm ver.
    """
    drop = before - after
    if drop > tolerance:
        print(f"❌ REGRESSION DETECTED in {name}: drop={drop:.4f}")
        return False
    else:
        print(f"✅ {name} stable (drop={drop:.4f})")
        return True

stable_acc = check_regression(before_acc, after_acc, "Accuracy")
stable_f1  = check_regression(before_f1,  after_f1,  "F1-macro")

pipeline_stable = stable_acc and stable_f1

# -----------------------------------------------------------
# 6) Human-in-the-Loop Approval Report
# -----------------------------------------------------------

qa_report = {
    "before_accuracy": before_acc,
    "after_accuracy": after_acc,
    "accuracy_lift": acc_lift,
    
    "before_f1_macro": before_f1,
    "after_f1_macro": after_f1,
    "f1_lift": f1_lift,

    "regression_free": bool(pipeline_stable),
}

qa_report_df = pd.DataFrame([qa_report])
print("\n========== QA APPROVAL REPORT ==========")
print(qa_report_df)

print("\nPOST-REPAIR VALIDATION completed.\n")



# ===============================================================
# I) SEVERITY-WEIGHTED NOISE SCORING (Noise Priority Score - NPS)
# ===============================================================
print("\n================ I) SEVERITY-WEIGHTED NOISE SCORING (UCI HAR) ================\n")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

# ---------------------------
# 1. Prediction & Loss
# ---------------------------

y_pred_proba = model.predict_proba(X_train_selected)

# HAR labels 1-6 olduğu için -1 kaydırıyoruz
true_class_probs = y_pred_proba[np.arange(len(y_train)), y_train - 1]
y_pred_train = np.argmax(y_pred_proba, axis=1) + 1

sample_loss = -np.log(true_class_probs + 1e-9)

df_noise = pd.DataFrame()
df_noise["sample_id"] = np.arange(len(y_train))
df_noise["true_label"] = y_train
df_noise["pred_label"] = y_pred_train
df_noise["confidence"] = true_class_probs
df_noise["est_model_loss"] = sample_loss

# ---------------------------
# 2. Embedding (Sensor space) Anomaly - Mahalanobis
# ---------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_selected)

mean = np.mean(X_scaled, axis=0)
cov = np.cov(X_scaled, rowvar=False)
inv_cov = np.linalg.pinv(cov)

def mahalanobis_distance(x):
    return mahalanobis(x, mean, inv_cov)

df_noise["embedding_anomaly"] = np.apply_along_axis(mahalanobis_distance, 1, X_scaled)

# Normalize anomaly
df_noise["embedding_anomaly"] = (
    (df_noise["embedding_anomaly"] - df_noise["embedding_anomaly"].min()) /
    (df_noise["embedding_anomaly"].max() - df_noise["embedding_anomaly"].min() + 1e-9)
)

# ---------------------------
# 3. Class Distance (Near vs Far mismatch)
# ---------------------------

cm = confusion_matrix(y_train, y_pred_train)
cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

df_noise["class_distance"] = [
    1 - cm_norm[int(t)-1][int(p)-1]
    for t, p in zip(df_noise["true_label"], df_noise["pred_label"])
]

# ---------------------------
# 4. Logical contradiction
# ---------------------------

df_noise["logical_flag"] = (df_noise["true_label"] != df_noise["pred_label"]).astype(int)

# ---------------------------
# 5. Movement-sensor magnitude instability (HAR-specific)
# ---------------------------

sensor_magnitude = np.linalg.norm(X_train_selected, axis=1)

df_noise["sensor_magnitude"] = (
    (sensor_magnitude - sensor_magnitude.min()) /
    (sensor_magnitude.max() - sensor_magnitude.min() + 1e-9)
)

# ---------------------------
# 6. Normalize numerical scores
# ---------------------------

for col in ["est_model_loss", "class_distance"]:
    df_noise[col] = (
        (df_noise[col] - df_noise[col].min()) /
        (df_noise[col].max() - df_noise[col].min() + 1e-9)
    )

# ---------------------------
# 7. FINAL NOISE PRIORITY SCORE (Meta FAIR Weighted)
# ---------------------------

w1, w2, w3, w4, w5 = 0.30, 0.25, 0.20, 0.15, 0.10

df_noise["NPS"] = (
    w1 * df_noise["est_model_loss"] +
    w2 * df_noise["class_distance"] +
    w3 * df_noise["embedding_anomaly"] +
    w4 * (df_noise["logical_flag"] * 2) +
    w5 * df_noise["sensor_magnitude"]
)

df_noise = df_noise.sort_values("NPS", ascending=False)

print("\n✅ Noise Priority Score calculated successfully\n")

# ---------------------------
# 8. Top priority noisy samples
# ---------------------------

top_5_percent = int(len(df_noise) * 0.05)

priority_samples = df_noise.head(top_5_percent)

print("Top 10 highest-priority noisy samples:\n")
print(priority_samples[["sample_id", "true_label", "pred_label", "NPS"]].head(10))

# Save for Active Learning (F loop)
priority_samples.to_csv("Noise_Priority_List_UCI_HAR.csv", index=False)

print("\n✅ Noise Priority List saved as: Noise_Priority_List_UCI_HAR.csv")
print("\nI) Severity-Weighted Noise Scoring COMPLETED.\n")



# ================================================================
# J) FULL BIAS AUDITING (UCI HAR) - Full Script
# ================================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

print("\n=== START: FULL BIAS AUDIT (UCI HAR) ===\n")

# -------------------------
# 0) CHECK REQUIRED OBJECTS
# -------------------------
_required = ["model", "X_train_selected", "X_test_selected", "y_train", "y_test"]
_missing = [o for o in _required if o not in globals()]
if _missing:
    raise NameError(f"Required objects missing from workspace: {_missing}\nPlease define these before running the audit.")

# shorten names
model = globals()["model"]
X_train = globals()["X_train_selected"]
X_test  = globals()["X_test_selected"]
y_train = np.array(globals()["y_train"])
y_test  = np.array(globals()["y_test"])

# Ensure shapes consistent
if len(y_test) != len(X_test):
    raise ValueError(f"Length mismatch: len(y_test)={len(y_test)} vs len(X_test)={len(X_test)}")

# Output folder
OUT_DIR = "bias_audit_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# 1) PREDICTIONS + alignment for HAR labels (1-6)
# -------------------------
# Try predict_proba if available for possible calibration checks, else predict
has_proba = hasattr(model, "predict_proba")
if has_proba:
    y_proba_test = model.predict_proba(X_test)  # shape (N, C) with C likely = 6 (0..5)
    # Convert to 1-based predicted labels if y_test labels are 1..6
    # Check label base (if y_test min >=1 assume 1-based)
    if y_test.min() >= 1:
        y_pred_test = np.argmax(y_proba_test, axis=1) + 1
    else:
        y_pred_test = np.argmax(y_proba_test, axis=1)
else:
    y_pred_test = model.predict(X_test)
    y_proba_test = None
    # Align to 1-based if necessary
    if y_test.min() >= 1 and y_pred_test.min() == 0:
        y_pred_test = np.array(y_pred_test) + 1

print("Predictions computed. Has predict_proba:", has_proba)

# -------------------------
# 2) BASIC OVERALL METRICS
# -------------------------
overall_acc = accuracy_score(y_test, y_pred_test)
prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred_test, average=None, labels=np.unique(y_test))
class_report = classification_report(y_test, y_pred_test, digits=4)

print("Overall Accuracy: {:.4f}".format(overall_acc))
print("\nClassification report (per class):\n")
print(class_report)

# save classification report text
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n")
    f.write(class_report)

# -------------------------
# 3) CONFUSION MATRIX (visual + csv)
# -------------------------
labels_sorted = np.unique(np.concatenate([y_test, y_pred_test]))
cm = confusion_matrix(y_test, y_pred_test, labels=labels_sorted)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_sorted, yticklabels=labels_sorted)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (UCI HAR)")
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved -> {cm_path}")

# Save as CSV
pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))

# -------------------------
# 4) SUBJECT / SENSOR (surrogate geographic/device) BIAS
# -------------------------

subject_ids = None

possible_subject_vars = [
    "subject_test",
    "subjects_test",
    "subject_ids",
    "subject"
]

for cand in possible_subject_vars:
    if cand in globals():

        obj = globals()[cand]

        if isinstance(obj, (list, np.ndarray, pd.Series)):

            arr = np.array(obj)

            if arr.ndim == 1 and len(arr) == len(y_test):
                subject_ids = arr
                print(f"✅ Using '{cand}' as subject id source.")
                break


# fallback: if not available
if subject_ids is None:
    print("⚠️ subject_id array not found. Using auto-binned surrogate subject groups (UCI HAR safe).")

    n_subjects = 30  # HAR has 30 subjects

    subject_ids = np.repeat(
        np.arange(1, n_subjects + 1),
        repeats=int(np.ceil(len(y_test) / n_subjects))   # <-- FIX HERE
    )[:len(y_test)]


# compute per-subject accuracy
unique_subjects = np.unique(subject_ids)

sub_accs = {}
sub_counts = {}

for s in unique_subjects:
    mask = (subject_ids == s)

    if mask.sum() >= 5:
        acc = accuracy_score(y_test[mask], y_pred_test[mask])
    else:
        acc = np.nan

    sub_accs[int(s)] = float(acc)
    sub_counts[int(s)] = int(mask.sum())


sub_acc_df = pd.DataFrame({
    "subject_id": list(sub_accs.keys()),
    "accuracy": list(sub_accs.values()),
    "n_samples": list(sub_counts.values())
})

sub_acc_df.to_csv(os.path.join(OUT_DIR, "per_subject_accuracy.csv"), index=False)


# --- plot ---
plt.figure(figsize=(14,5))
plt.bar(sub_acc_df["subject_id"].astype(str), sub_acc_df["accuracy"])
plt.xlabel("Subject ID")
plt.ylabel("Accuracy")
plt.title("Per-Subject Accuracy (UCI HAR)")
plt.xticks(rotation=90)
plt.grid(alpha=0.3)
plt.tight_layout()

sub_plot_path = os.path.join(OUT_DIR, "per_subject_accuracy.png")
plt.savefig(sub_plot_path)
plt.close()

print(f"Per-subject accuracy saved -> {sub_plot_path}")


sub_std = sub_acc_df["accuracy"].std()
sub_mean = sub_acc_df["accuracy"].mean()

subject_bias_risk = "LOW"
if sub_std > 0.08:
    subject_bias_risk = "HIGH"
elif sub_std > 0.04:
    subject_bias_risk = "MEDIUM"

print(f"Per-subject accuracy mean={sub_mean:.4f}, std={sub_std:.4f} → risk={subject_bias_risk}")

# -------------------------
# 5) TIME-DRIFT (early vs late test split)
# -------------------------
mid = len(X_test) // 2
acc_early = accuracy_score(y_test[:mid], y_pred_test[:mid])
acc_late  = accuracy_score(y_test[mid:], y_pred_test[mid:])
time_drift_abs = abs(acc_early - acc_late)

print(f"Time split accuracies -> early: {acc_early:.4f}, late: {acc_late:.4f}, abs diff: {time_drift_abs:.4f}")
time_drift_risk = "LOW" if time_drift_abs < 0.02 else ("MEDIUM" if time_drift_abs < 0.05 else "HIGH")

# -------------------------
# 6) CLASS-LEVEL SKEW & PERFORMANCE
# -------------------------
class_counts = pd.Series(y_train).value_counts().sort_index()
class_perf = []
for cls in labels_sorted:
    mask = (y_test == cls)
    n = mask.sum()
    acc_cls = accuracy_score(y_test[mask], y_pred_test[mask]) if n > 0 else np.nan
    class_perf.append({"class": int(cls), "n_test": int(n), "accuracy": float(acc_cls)})

class_perf_df = pd.DataFrame(class_perf).sort_values("class")
class_perf_df.to_csv(os.path.join(OUT_DIR, "class_level_performance.csv"), index=False)

# plot class-level accuracy & train distribution
plt.figure(figsize=(8,4))
ax = plt.bar(class_perf_df["class"].astype(str), class_perf_df["accuracy"])
plt.xlabel("Class (Activity Label)")
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.tight_layout()
cls_plot_path = os.path.join(OUT_DIR, "per_class_accuracy.png")
plt.savefig(cls_plot_path)
plt.close()
print(f"Per-class accuracy saved -> {cls_plot_path}")

# also save class distribution
pd.DataFrame({"class": class_counts.index.astype(int), "train_count": class_counts.values}).to_csv(os.path.join(OUT_DIR, "train_class_distribution.csv"), index=False)

# -------------------------
# 7) ANNOTATOR BIAS CHECK (NOT APPLICABLE)
# -------------------------
annotator_bias = {
    "exists": False,
    "reason": "UCI HAR standard release has no per-sample annotator_id metadata available in this workspace.",
    "recommendation": "If annotator metadata becomes available, compute per-annotator JS divergence vs global label distribution and confusion matrices."
}

# -------------------------
# 8) SENSOR-LEVEL PHYSICS CHECK (HAR-specific heuristics)
#    (Check for abnormal sensor magnitude distributions per class)
# -------------------------
try:
    # if X_test is 2D with sensor channels across columns
    sensor_magnitude = np.linalg.norm(X_test, axis=1)
    mag_df = pd.DataFrame({"mag": sensor_magnitude, "true": y_test, "pred": y_pred_test})
    agg = mag_df.groupby("true")["mag"].agg(["mean","std","count"]).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, "sensor_magnitude_by_true_class.csv"), index=False)

    # quick check: classes with unusually low count or extreme std
    extreme_mag = agg[(agg["count"] < 50) | (agg["std"] > (agg["std"].median() * 2))]
    sensor_bias_notes = extreme_mag.to_dict(orient="records")
except Exception as e:
    sensor_bias_notes = [{"error": str(e)}]

# -------------------------
# 9) ANOMALY: per-class confusion-based 'near/far' metric
# -------------------------
# compute for each sample whether predicted class is commonly confused with true class
cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
# map label->row index
label_to_idx = {int(l): i for i, l in enumerate(labels_sorted)}
near_far = []
for t, p in zip(y_test, y_pred_test):
    i = label_to_idx[int(t)]
    j = label_to_idx[int(p)]
    conf_prob = cm_norm[i, j]
    # near if conf_prob > median conf for that true class
    near_far.append(float(conf_prob))

np_test_nearfar = np.array(near_far)
# attach per-class median conf
median_conf_by_class = pd.DataFrame({"class": labels_sorted, "median_conf": np.median(cm_norm, axis=1)})

# -------------------------
# 10) AGGREGATE REPORT OBJECT
# -------------------------
report = {
    "dataset": "UCI HAR (surrogate)",
    "overall_accuracy": float(overall_acc),
    "per_subject_mean_accuracy": float(sub_mean),
    "per_subject_std_accuracy": float(sub_std),
    "per_subject_bias_risk": subject_bias_risk,
    "time_drift_early_acc": float(acc_early),
    "time_drift_late_acc": float(acc_late),
    "time_drift_abs": float(time_drift_abs),
    "time_drift_risk": time_drift_risk,
    "class_counts_train": class_counts.to_dict(),
    "annotator_bias": annotator_bias,
    "sensor_bias_notes": sensor_bias_notes,
    "notes": "Demographic bias (gender/age) not applicable in this dataset."
}

# write master json + csv summary
with open(os.path.join(OUT_DIR, "bias_audit_summary.json"), "w") as f:
    json.dump(report, f, indent=2)

# summary csv row
summary_row = {
    "overall_accuracy": overall_acc,
    "per_subject_mean_accuracy": sub_mean,
    "per_subject_std_accuracy": sub_std,
    "time_drift_abs": time_drift_abs,
    "time_drift_risk": time_drift_risk,
    "per_subject_bias_risk": subject_bias_risk
}
pd.DataFrame([summary_row]).to_csv(os.path.join(OUT_DIR, "bias_audit_report.csv"), index=False)

# -------------------------
# 11) PRINT EXECUTIVE SUMMARY (console)
# -------------------------
print("\n=== EXECUTIVE SUMMARY ===")
print(f"Dataset: UCI HAR (surrogate metadata)")
print(f"Overall accuracy: {overall_acc:.4f}")
print(f"Per-subject accuracy: mean={sub_mean:.4f}, std={sub_std:.4f} -> risk={subject_bias_risk}")
print(f"Time drift (early vs late) abs diff: {time_drift_abs:.4f} -> risk={time_drift_risk}")
print("Top per-class accuracies (class : accuracy):")
print(class_perf_df[["class", "accuracy"]].to_string(index=False))
print("\nNotes:")
print("- Demographic bias: Not applicable (no gender/age in dataset).")
print("- Annotator bias: Not available (no annotator IDs).")
print(f"- Sensor-level notes (saved): {os.path.join(OUT_DIR, 'sensor_magnitude_by_true_class.csv')}")
print("\nSaved all outputs to folder:", OUT_DIR)

print("\n=== END: FULL BIAS AUDIT ===\n")



# ============================================================
# K) MODEL ROBUSTIFICATION & DEBIASING - UCI HAR Implementation
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os

print("\n================= K) MODEL ROBUSTIFICATION & DEBIASING =================\n")

OUT_DIR = "robustification_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# 1) LOAD / CHECK DATA
# =========================================================

env = globals()

Xtr = X_train_selected if "X_train_selected" in env else X_train
Xte = X_test_selected  if "X_test_selected" in env  else X_test

ytr = y_train.copy()
yte = y_test.copy()

# ✅ Fix label indexing (1..6  →  0..5)
encoder = LabelEncoder()
ytr = encoder.fit_transform(ytr)
yte = encoder.transform(yte)

print("Train :", Xtr.shape, "Classes:", np.unique(ytr))
print("Test  :", Xte.shape)

# =========================================================
# 2) SENSOR-FRAME NORMALIZATION
# =========================================================

scaler = StandardScaler()
Xtr_norm = scaler.fit_transform(Xtr)
Xte_norm = scaler.transform(Xte)

print("✅ Normalization done")

# =========================================================
# 3) ROTATION AUGMENTATION
# =========================================================

def rotation_augment(X, y, angle_noise=0.03):
    noise = np.random.normal(0, angle_noise, X.shape)
    X_aug = X + noise
    X_comb = np.vstack((X, X_aug))
    y_comb = np.hstack((y, y))
    return X_comb, y_comb

Xtr_aug, ytr_aug = rotation_augment(Xtr_norm, ytr)

print("✅ Augmented:", Xtr_aug.shape)

# =========================================================
# 4) ORIENTATION-INVARIANT PROJECTION
# =========================================================

Xtr_inv = np.abs(Xtr_aug)
Xte_inv = np.abs(Xte_norm)

print("✅ Orientation-invariant applied")

# =========================================================
# 5) HARD / UNSTABLE SAMPLE DETECTION
# =========================================================

baseline = RandomForestClassifier(n_estimators=150, random_state=42)
baseline.fit(Xtr_inv, ytr_aug)

probs = baseline.predict_proba(Xtr_inv)

# ✅ GUARANTEED SAFE INDEXING
true_probs = np.take_along_axis(
    probs, 
    ytr_aug.reshape(-1,1), 
    axis=1
).flatten()

loss = -np.log(true_probs + 1e-9)

threshold = np.percentile(loss, 85)
hard_idx = loss >= threshold

print("Unstable samples:", np.sum(hard_idx))

# =========================================================
# 6) SLICE-AWARE + HARD REWEIGHT
# =========================================================

base_weights = compute_sample_weight("balanced", ytr_aug)
hard_boost = np.where(hard_idx, 3.0, 1.0)

sample_weights = base_weights * hard_boost

print("✅ Reweighting finished")

# =========================================================
# 7) FINAL ROBUST MODEL
# =========================================================

robust_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=42
)

robust_model.fit(Xtr_inv, ytr_aug, sample_weight=sample_weights)

# =========================================================
# 8) EVALUATION
# =========================================================

y_pred_robust = robust_model.predict(Xte_inv)

robust_acc = accuracy_score(yte, y_pred_robust)
robust_f1  = f1_score(yte, y_pred_robust, average="macro")

print("\n========== ROBUST RESULTS ==========")
print("Accuracy :", round(robust_acc, 4))
print("F1 Macro :", round(robust_f1, 4))

df = pd.DataFrame({
    "metric": ["accuracy", "f1_macro"],
    "value": [robust_acc, robust_f1]
})

df.to_csv(f"{OUT_DIR}/robust_metrics.csv", index=False)

print("\n✅ K) ROBUSTIFICATION COMPLETED WITHOUT ERRORS ✅")
