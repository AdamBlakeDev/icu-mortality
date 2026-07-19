# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model Development
#
# <span style="color:red">**Note:** </span>This notebook is paired with a Python script using **Jupytext**.  
#  - The `.ipynb` is the primary artifact for exploration and explanation.  
#  - A synchronized `.py` file is maintained for reproducibility and review.  
#  - To manually sync all notebooks, run the alias `syncnb`.  
#  - The project does not require Jupytext to run; the paired `.py` files are already committed.
#

# %% [markdown]
# <h2 style="color:cyan">1. Feature Engineering</h2>
#
#  1. Setup: load data via DuckDB, load feature set from exploration
#  2. Define offset map for 24h temporal filtering
#  3. Build target variable (binary label)
#  4. Inspect each table before aggregation
#  5. Extract and aggregate text with bucket-specific logic
#  6. Join all tables onto target labels
#  7. Map text columns to API input fields (diagnosis, admission, history)
#  8. Add and clean demographics for encoding during training (age, gender, unittype)
#  9. Train/test split (stratified)

# %% [markdown]
# ### Setup
#
#  - Setting base paths
#  - DuckDB connection using SQLite file
#  - Load Feature Set from data_exploration.ipynb

# %%
from pathlib import Path
import duckdb
import pandas as pd

# View all rows and full column content when displaying DataFrames
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Setting base paths
PROJECT_ROOT = Path("/root/icu_mortality_takehome")
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "eicu"
SQLITE_DB = DATA_DIR / "eicu_v2_0_1.sqlite3"

# Set up DuckDB connection and load SQLite extension
con = duckdb.connect()
con.execute("INSTALL sqlite;")
con.execute("LOAD sqlite;")

# Confirm the database file exists
print(f'Path:   {SQLITE_DB}\nExists: {SQLITE_DB.exists()}\n')

# Load feature set exported from data_exploration.ipynb.
# Contains tables with ≥ 50% coverage and their classified text columns.
feature_set = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "feature_set.csv")

# Display the feature set to verify it was loaded correctly
feature_set.head()

# %% [markdown]
# ### Define primary offset columns for temporal filtering
#
# Each table has one or more offset columns representing minutes from
# ICU admission. The primary clinical event offset is manually selected for each
# table. System-entry timestamps ("entered" or "entry" variants) are
# excluded when a clinical event offset exists, but are used as a
# fallback for tables where they are the only offset available (e.g.,
# admissiondx with admitdxenteredoffset). Tables without any offset
# column contain static data (e.g., patient demographics) and do not
# require temporal filtering.

# %%
# Map each table to its primary clinical event offset column.
# Selected based on offset analysis in data_exploration.ipynb.
# Entry/entered offsets excluded (represent when data was logged,
# not when the clinical event occurred).
OFFSET_MAP = {
    "admissiondx": "admitdxenteredoffset",
    "careplancareprovider": "careprovidersaveoffset",
    "careplangeneral": "cplitemoffset",
    "diagnosis": "diagnosisoffset",
    "intakeoutput": "intakeoutputoffset",
    "lab": "labresultoffset",
    "medication": "drugorderoffset",
    "note": "noteoffset",
    "nursecharting": "nursingchartoffset",
    "pasthistory": "pasthistoryoffset",
    "physicalexam": "physicalexamoffset",
    "respiratorycharting": "respchartoffset",
    "treatment": "treatmentoffset",
}

# 24-hour cutoff in minutes
OFFSET_CUTOFF = 1440

# %% [markdown]
# ### Build target variable
#
# Pull patientunitstayid and hospitaldischargestatus from the patient
# table. Drop rows with null or empty labels, then encode as binary
# (Alive=0, Expired=1). This dataframe is the base table that all feature 
# tables join onto by patientunitstayid.

# %%
# Query non-null, non-empty discharge status labels
labels = con.execute(f"""
    SELECT 
        patientunitstayid,
        hospitaldischargestatus AS label
    FROM sqlite_scan('{SQLITE_DB}', 'patient')
    WHERE hospitaldischargestatus IS NOT NULL
      AND TRIM(hospitaldischargestatus) <> ''
""").df()

# Binary encoding: Alive=0, Expired=1
labels["y"] = (labels["label"] == "Expired").astype(int)

print(f"Usable stays: {len(labels)}")
print(f"\nClass distribution:")
print(labels["y"].value_counts().to_string())
labels.head()


# %% [markdown]
# ### Extract text features by table
#
# Each table is queried individually with the 24-hour temporal filter
# applied. This allows inspection of the raw text before aggregation
# and ensures each table's offset column and text columns are handled
# correctly.

# %%
# Reusable inspection function for any table in the feature set.
# Pulls text columns and offset column with the 24h filter applied.
def inspect_table(table_name, n=20):
    """Display feature set columns and sample rows for a table."""
    
    # Show which columns and buckets are included
    cols_info = feature_set[feature_set["table_name"] == table_name]
    print("Check what columns are in the feature set for this table:")
    display(cols_info)
    
    # Get text column names for the query
    text_cols = cols_info["column_name"].tolist()
    offset_col = OFFSET_MAP.get(table_name)
    
    # Build SELECT clause: always include stay ID and offset
    select_cols = ["patientunitstayid"]
    if offset_col:
        select_cols.append(offset_col)
    select_cols.extend(text_cols)
    
    # Build WHERE clause: apply temporal filter if offset exists
    where = f"WHERE {offset_col} <= {OFFSET_CUTOFF}" if offset_col else ""
    
    print(f"\nSample raw rows from {table_name} before aggregation to inspect text content:")
    return con.execute(f"""
        SELECT {', '.join(select_cols)}
        FROM sqlite_scan('{SQLITE_DB}', '{table_name}')
        {where}
        LIMIT {n}
    """).df()


# %% [markdown]
# <h4 style="color:yellow">admissiondx</h4>

# %%
inspect_table("admissiondx")

# %% [markdown]
# <h4 style="color:yellow">careplancareprovider</h4>

# %%
inspect_table("careplancareprovider")

# %% [markdown]
# <span style="color:red">**Note:** </span>
# It seems that "Unknown" values in `interventioncategory` are system-level placeholders
# for missing data, not clinical observations. These will be filtered out during aggregation 
# alongside nulls and empty strings.

# %% [markdown]
# <h4 style="color:yellow">careplangeneral</h4>

# %%
inspect_table("careplangeneral")

# %% [markdown]
# <h4 style="color:yellow">diagnosis</h4>

# %%
inspect_table("diagnosis")

# %% [markdown]
# <h4 style="color:yellow">intakeoutput</h4>

# %%
inspect_table("intakeoutput")

# %% [markdown]
# <h4 style="color:yellow">lab</h4>

# %%
inspect_table("lab")

# %% [markdown]
# <h4 style="color:yellow">medication</h4>

# %%
inspect_table("medication")

# %% [markdown]
# <h4 style="color:yellow">note</h4>

# %%
inspect_table("note")

# %% [markdown]
# <h4 style="color:yellow">nursecharting</h4>

# %%
inspect_table("nursecharting")

# %% [markdown]
# <h4 style="color:yellow">pasthistory</h4>

# %%
inspect_table("pasthistory")

# %% [markdown]
# <h4 style="color:yellow">physicalexam</h4>

# %%
inspect_table("physicalexam")

# %% [markdown]
# <h4 style="color:yellow">respiratorycharting</h4>

# %%
inspect_table("respiratorycharting")

# %% [markdown]
# <h4 style="color:yellow">treatment</h4>

# %%
inspect_table("treatment")

# %% [markdown]
# <h4 style="color:yellow">patient</h4>

# %%
inspect_table("patient")

# %% [markdown]
# ### Text Extraction and Aggregation
#
# Each table's text columns are extracted with a 24-hour temporal filter,
# then aggregated to one row per patientunitstayid. Aggregation logic
# is routed by the final_bucket classification from exploration:
#
#  - **categorical_text**: `STRING_AGG(DISTINCT ...)` to collect unique labels
#  - **hierarchical_structured_text**: replace delimiters (| and /) with spaces,
#    then `STRING_AGG(DISTINCT ...)` to collect unique paths
#  - **mixed_text**: `STRING_AGG(...)` ordered by offset to preserve sequence
#  - **numeric_text_artifact**: skipped (already excluded from feature set)
#  - **REMOVE**: skipped (already excluded from feature set)

# %% [markdown]
# ### Verify feature set before aggregation

# %%
print(f"Tables: {feature_set['table_name'].nunique()}")
print(f"Columns: {len(feature_set)}")
print(f"\nBuckets:")
print(feature_set["final_bucket"].value_counts().to_string())


# %% [markdown]
# ### Aggregation function

# %%
# Aggregation function to extract and combine text columns for a given table
def extract_and_aggregate(table_name: str, feature_set: pd.DataFrame) -> pd.DataFrame:
    """
    Extract text columns from a single table, apply temporal filtering,
    and aggregate to one row per patientunitstayid.

    Aggregation is routed by final_bucket:
      - categorical_text: distinct values, space-separated
      - hierarchical_structured_text: delimiters replaced with spaces,
        then distinct values collected
      - mixed_text: all values concatenated, ordered by offset since they
        may represent a sequence of events

    Placeholder values ('Unknown', '') are filtered out during aggregation.

    Returns a dataframe with patientunitstayid and one column per text
    column, containing the aggregated text for that stay.
    """
    # Get columns and buckets for this table
    table_cols = feature_set[feature_set["table_name"] == table_name]

    if table_cols.empty:
        return pd.DataFrame(columns=["patientunitstayid"])

    # Look up offset column for temporal filtering
    offset_col = OFFSET_MAP.get(table_name)

    # Build SELECT expressions based on bucket type
    agg_exprs = []
    for _, row in table_cols.iterrows():
        col = row["column_name"]
        bucket = row["final_bucket"]

        if bucket == "hierarchical_structured_text":
            # Replace pipe and slash delimiters with spaces, then deduplicate
            clean = f"REPLACE(REPLACE({col}, '|', ' '), '/', ' ')"
            agg_exprs.append(
                f"STRING_AGG(DISTINCT CASE WHEN TRIM(CAST({col} AS VARCHAR)) NOT IN ('', 'Unknown') "
                f"THEN {clean} END, ' ') AS {col}"
            )
        elif bucket == "categorical_text":
            # Collect distinct non-empty values
            agg_exprs.append(
                f"STRING_AGG(DISTINCT CASE WHEN TRIM(CAST({col} AS VARCHAR)) NOT IN ('', 'Unknown') "
                f"THEN CAST({col} AS VARCHAR) END, ' ') AS {col}"
            )
        elif bucket == "mixed_text":
            # Concatenate all values ordered by offset (if available)
            agg_exprs.append(
                f"STRING_AGG(CASE WHEN TRIM(CAST({col} AS VARCHAR)) NOT IN ('', 'Unknown') "
                f"THEN CAST({col} AS VARCHAR) END, ' ' "
                f"ORDER BY {offset_col}) AS {col}" if offset_col else
                f"STRING_AGG(CASE WHEN TRIM(CAST({col} AS VARCHAR)) NOT IN ('', 'Unknown') "
                f"THEN CAST({col} AS VARCHAR) END, ' ') AS {col}"
            )

    # Build WHERE clause for temporal filtering
    where = f"WHERE {offset_col} <= {OFFSET_CUTOFF}" if offset_col else ""

    query = f"""
        SELECT
            patientunitstayid,
            {', '.join(agg_exprs)}
        FROM sqlite_scan('{SQLITE_DB}', '{table_name}')
        {where}
        GROUP BY patientunitstayid
    """

    return con.execute(query).df()


# %% [markdown]
# ### Run extraction and aggregation for all tables

# %%
# Aggregate text from each table into one row per stay
aggregated_tables = {}

for table_name in feature_set["table_name"].unique():
    print(f"Processing {table_name}...")
    df = extract_and_aggregate(table_name, feature_set)
    aggregated_tables[table_name] = df
    print(f"  → {len(df)} stays, {df.shape[1] - 1} text columns")

print(f"\nDone. {len(aggregated_tables)} tables processed.")

# %% [markdown]
# ### Inspect aggregated output

# %%
# Spot check a few tables to verify aggregation looks correct
for table_name in ["admissiondx", "patient", "diagnosis"]:
    print(f"\n--- {table_name} ---")
    display(aggregated_tables[table_name].head(3))

# %% [markdown]
# <span style="color:red">**Note:** </span> <br>
#  - **Potential Improvement:** concatenate diagnosis priority with each
#    diagnosis string before aggregation (e.g., "Primary acute renal failure")
#    to capture interactions between severity ranking and diagnosis type.
#    Deferred for baseline model where individual tokens carry most signal.

# %% [markdown]
# ### Join all aggregated text onto labels

# %%
# Normalize column names to lowercase across all aggregated tables
# (DuckDB returns mixed case for some tables)
for table_name in aggregated_tables:
    aggregated_tables[table_name].columns = (
        aggregated_tables[table_name].columns.str.lower()
    )

# Start with the labels dataframe as the base table.
# Left join each aggregated table by patientunitstayid.
# Stays with no data in a given table will have NaN, filled with empty strings.
model_df = labels[["patientunitstayid", "y"]].copy()

for table_name, agg_df in aggregated_tables.items():
    model_df = model_df.merge(agg_df, on="patientunitstayid", how="left")

# Fill NaN with empty strings (patient has no data in that table)
model_df = model_df.fillna("")

print(f"Shape: {model_df.shape}")
print(f"Columns: {list(model_df.columns)}")
model_df.head(1)

# %% [markdown]
# ### Map text columns to API input fields
#
# The API accepts three text fields: `diagnosis_text`, `admission_text` and `history_text`. Each aggregated column is mapped to the API field
# it would populate at serving time. This ensures the training representation matches what the model receives at inference.
#
# **diagnosis_text:** What is wrong with the patient and what treatments they are receiving. Includes diagnosis strings, diagnosis priority,
# and treatment strings. Treatments are grouped here because they directly reflect the underlying diagnoses.
#
#  - `diagnosisstring`: the actual diagnoses (e.g., "cardiovascular arrhythmias syncope")
#  - `diagnosispriority`: severity ranking of each diagnosis (Primary, Major, Other)
#  - `treatmentstring`: treatments being applied (e.g., "pulmonary medications bronchodilator inhaled")
#
# **admission_text:** What happened at admission and the patient's current clinical state. This is the broadest field, capturing
# admission diagnosis, progress notes, vitals and assessments being monitored, care plan items, care providers and specialties, lab
# orders, medications, respiratory monitoring, intake/output and physical exam findings.
#
#  - `admitdxpath`: why the patient was admitted
#  - `notepath`, `notetype`: clinical progress notes
#  - `nursingchartcelltypecat`, `nursingchartcelltypevallabel`: what vitals and assessments are being monitored
#  - `cplgroup`, `cplitemvalue`: care plan items (ventilation, airway, sedation)
#  - `interventioncategory`, `managingphysician`, `providertype`, `specialty`: who is caring for the patient
#  - `labname`: what labs were ordered
#  - `drugname`, `frequency`, `routeadmin`: medications administered
#  - `respcharttypecat`, `respchartvaluelabel`: respiratory monitoring
#  - `cellpath`: intake/output tracking
#  - `physicalexampath`: physical exam findings
#
# **history_text:** What happened before this admission. Contains past medical history note type and the full hierarchical history
# path (e.g., Cardiovascular/Myocardial Infarction/MI - remote).
#
#  - `pasthistorynotetype`: type of history note
#  - `pasthistorypath`: the actual past medical history (e.g., "Cardiovascular/Myocardial Infarction/MI - remote")

# %%
API_TEXT_MAP = {
    "diagnosis_text": [
        "diagnosisstring", "diagnosispriority",
        "treatmentstring",
    ],
    "admission_text": [
        "admitdxpath", "notetype", "notepath",
        "nursingchartcelltypecat", "nursingchartcelltypevallabel",
        "cplgroup", "cplitemvalue",
        "interventioncategory", "managingphysician", "providertype", "specialty",
        "labname", "drugname", "frequency", "routeadmin",
        "respcharttypecat", "respchartvaluelabel",
        "cellpath", "physicalexampath",
    ],
    "history_text": [
        "pasthistorynotetype", "pasthistorypath",
    ],
}

# Build the three API text fields by concatenating mapped columns
for api_field, cols in API_TEXT_MAP.items():
    available = [c for c in cols if c in model_df.columns]
    model_df[api_field] = model_df[available].apply(
        lambda row: " ".join(row.values), axis=1
    )

# Combine all three into a single text input for TF-IDF
model_df["text"] = (
    model_df["diagnosis_text"] + " " +
    model_df["admission_text"] + " " +
    model_df["history_text"]
)

print("API field text lengths (characters):")
for field in ["diagnosis_text", "admission_text", "history_text", "text"]:
    print(f"  {field}: {model_df[field].str.len().describe()[['mean', '50%', 'max']].to_dict()}")

# View the final dataframe structure
print(f"\nFinal dataframe shape: {model_df.shape}\n")
model_df.head(1)

# %% [markdown]
# ### Add age from patient table
#
# Age is a `numeric_text_artifact` field not captured by the text feature set.
# Pulled directly from the patient table to match the API demographics specification.

# %%
# Pull age from patient table
age_df = con.execute(f"""
    SELECT patientunitstayid, age
    FROM sqlite_scan('{SQLITE_DB}', 'patient')
""").df()

model_df = model_df.merge(age_df, on="patientunitstayid", how="left")
model_df["age"] = model_df["age"].fillna("")

print(f"Age sample values: {model_df['age'].value_counts().head(10).to_dict()}")

# %%
# Checking cardinality of demographic fields to confirm they are suitable for one-hot encoding
print("gender:", model_df["gender"].nunique(), model_df["gender"].unique())
print("")
print("unittype:", model_df["unittype"].nunique(), model_df["unittype"].unique())

# %% [markdown]
# ### Clean demographics for encoding
#
#  - Age: convert to numeric, handle "> 89" as 90 (standard convention for HIPAA de-identified eICU data), fill missing with median.
#  - Gender and unittype: one-hot encoded in the model pipeline.

# %%
# Convert age to numeric
model_df["age_numeric"] = (
    model_df["age"]
    .replace("> 89", "90")
    .replace("", None)
    .astype(float)
)

# Fill missing age with median
median_age = model_df["age_numeric"].median()
model_df["age_numeric"] = model_df["age_numeric"].fillna(median_age)

print(f"Age stats:")
print(model_df["age_numeric"].describe().to_string())

# %% [markdown]
# ### Train/test split
#
# Stratified split to preserve class distribution across train and test
# sets. The 90/10 imbalance makes stratification essential to ensure
# the minority class (Expired) is represented in both sets.

# %%
from sklearn.model_selection import train_test_split

# Columns to keep for modeling
MODEL_COLS = [
    "text", "diagnosis_text", "admission_text", "history_text",
    "age_numeric", "gender", "unittype"
]

# Prepare features and target for modeling
X = model_df[MODEL_COLS]
y = model_df["y"]

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Check class distribution in train and test sets
print(f"Train: {len(X_train)} stays ({y_train.sum()} expired, {y_train.mean():.1%})")
print(f"Test:  {len(X_test)} stays ({y_test.sum()} expired, {y_test.mean():.1%})")

# Final check of feature dataframe before modeling
print(f"\nShape: {X.shape}")
print(f"\nDtypes:\n{X.dtypes.to_string()}")
print(f"\nNulls:\n{X.isnull().sum().to_string()}\n")
X.head(1)

# %% [markdown]
# <h2 style="color:cyan">2. Model Training</h2>
#
#  1. Baseline: single TF-IDF + Logistic Regression
#  2. Feature iteration: separate TF-IDF per API field + Logistic Regression
#  3. Classifier iteration: separate TF-IDF + XGBoost
#  4. Classifier iteration: separate TF-IDF + Calibrated Linear SVM
#  5. Ensemble: soft voting over LR + Calibrated SVM
#  6. Model comparison and final selection

# %% [markdown]
# ### Baseline: single TF-IDF + Logistic Regression
#
# The simplest viable model: all text concatenated into one field,
# vectorized with TF-IDF, combined with encoded demographics, and
# classified with logistic regression. This establishes a performance
# floor to iterate on.
#
# **Key parameter choices:**
#
#  - `max_features=10000`: caps vocabulary to prevent overfitting on ~2,500 stays
#  - `stop_words="english"`: removes common English words that add noise to clinical text
#  - `handle_unknown="ignore"`: prevents errors if test set contains unseen categories
#  - `class_weight="balanced"`: automatically upweights the minority class (Expired) to address the 90/10 imbalance
#  - `max_iter=1000`: logistic regression may need more iterations to converge on high-dimensional sparse data

# %%
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Define preprocessing: TF-IDF for text, one-hot for categoricals, passthrough for age
preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english"), "text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["gender", "unittype"]),
        ("age", "passthrough", ["age_numeric"]),
    ]
)

# Full pipeline: preprocessing + classifier
baseline_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )),
])

# Train
baseline_pipeline.fit(X_train, y_train)

# %% [markdown]
# ### Baseline Evaluation

# %%
from sklearn.metrics import (
    classification_report, roc_auc_score, 
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Predicted probabilities and labels
y_pred = baseline_pipeline.predict(X_test)
y_prob = baseline_pipeline.predict_proba(X_test)[:, 1]

# AUROC
auroc = roc_auc_score(y_test, y_prob)
print(f"AUROC: {auroc:.3f}\n")

# Classification report (precision, recall, F1 per class)
print(classification_report(y_test, y_pred, target_names=["Alive", "Expired"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Alive", "Expired"], ax=ax, cmap="Blues"
)
ax.set_title(f"Baseline: TF-IDF + Logistic Regression (AUROC={auroc:.3f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color:red">**Baseline Results:** </span>
# AUROC confirms the clinical text contains meaningful mortality
# signal. The model achieves ~64% recall on expired patients but at the
# cost of low precision (~19%), producing many false positives. This is
# expected behavior from `class_weight="balanced"` which aggressively
# upweights the minority class. The next iteration tests whether
# separate TF-IDF vectorizers per API field improve discrimination
# by preserving field-level context.

# %% [markdown]
# ### Feature Iteration: Separate TF-IDF per API field + Logistic Regression
#
# Instead of a single TF-IDF over all text, each API field gets its own
# vectorizer. This allows the model to learn that the same token (e.g.,
# "cardiovascular") may carry different predictive weight depending on
# whether it appears in a diagnosis, admission note, or past history.
#
# The classifier and all other parameters remain identical to the
# baseline to isolate the effect of the representation change.

# %%
# Define preprocessing: separate TF-IDF per API text field
preprocessor_separate = ColumnTransformer(
    transformers=[
        ("tfidf_dx", TfidfVectorizer(max_features=5000, stop_words="english"), "diagnosis_text"),
        ("tfidf_adm", TfidfVectorizer(max_features=5000, stop_words="english"), "admission_text"),
        ("tfidf_hx", TfidfVectorizer(max_features=5000, stop_words="english"), "history_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["gender", "unittype"]),
        ("age", "passthrough", ["age_numeric"]),
    ]
)

# Full pipeline: preprocessing + classifier
separate_pipeline = Pipeline([
    ("preprocessor", preprocessor_separate),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )),
])

# Train
separate_pipeline.fit(X_train, y_train)

# %% [markdown]
# ### Separate Vectorizer Evaluation

# %%
# Predicted probabilities and labels
y_pred_sep = separate_pipeline.predict(X_test)
y_prob_sep = separate_pipeline.predict_proba(X_test)[:, 1]

# AUROC
auroc_sep = roc_auc_score(y_test, y_prob_sep)
print(f"AUROC: {auroc_sep:.3f}\n")

# Classification report (precision, recall, F1 per class)
print(classification_report(y_test, y_pred_sep, target_names=["Alive", "Expired"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_sep, display_labels=["Alive", "Expired"], ax=ax, cmap="Blues"
)
ax.set_title(f"Separate TF-IDF + Logistic Regression (AUROC={auroc_sep:.3f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color:red">**Separate Vectorizer Results:** </span>
# AUROC improves from ~78% to ~81%. The major gain is in precision
# for the Expired class (~19% → ~29%), cutting false positives nearly
# in half (~117 → ~64) while maintaining similar recall (~64% → ~62%).
# This confirms that field-level context matters: the same clinical
# term carries different predictive weight depending on whether it
# appears in a diagnosis, admission note, or past history.
# The separate vectorizer representation is used going forward.

# %% [markdown]
# ### Classifier Iteration: separate TF-IDF + XGBoost
#
# The separate vectorizer representation is kept since it outperformed
# the single vectorizer. Now the classifier is swapped from logistic
# regression to XGBoost to test whether a nonlinear model can better
# exploit the feature interactions in the sparse TF-IDF matrix.
#
# The representation and preprocessing are identical to isolate the
# effect of the classifier change.

# %%
from xgboost import XGBClassifier
import numpy as np

# Calculate scale_pos_weight for class imbalance (equivalent to class_weight="balanced")
scale_pos = int(np.sum(y_train == 0) / np.sum(y_train == 1))

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor_separate),
    ("classifier", XGBClassifier(
        scale_pos_weight=scale_pos,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )),
])

# Train
xgb_pipeline.fit(X_train, y_train)

# %%
# Evaluate
y_pred_xgb = xgb_pipeline.predict(X_test)
y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

# AUROC
auroc_xgb = roc_auc_score(y_test, y_prob_xgb)
print(f"AUROC: {auroc_xgb:.3f}\n")

# Classification report (precision, recall, F1 per class)
print(classification_report(y_test, y_pred_xgb, target_names=["Alive", "Expired"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_xgb, display_labels=["Alive", "Expired"], ax=ax, cmap="Blues"
)
ax.set_title(f"Separate TF-IDF + XGBoost (AUROC={auroc_xgb:.3f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color:red">**XGBoost Results:** </span>
# AUROC drops to ~77% and expired recall collapses to ~17%, catching
# only ~7 of 42 expired patients. Tree-based models struggle with
# high-dimensional sparse TF-IDF features on small datasets because
# individual tree splits cannot find reliable patterns across 15,000
# sparse columns with only 170 expired training examples. Logistic
# regression is better suited to this feature space as it learns a
# global weight per feature rather than searching for split points.

# %% [markdown]
# ### Classifier Iteration: Separate TF-IDF + Calibrated Linear SVM
#
# Linear SVM is a natural alternative to logistic regression for
# high-dimensional sparse text classification. It maximizes the margin
# between classes rather than modeling probabilities directly, which
# can improve discrimination on small datasets.
#
# `CalibratedClassifierCV` wraps the SVM with Platt scaling to produce
# calibrated probability estimates. This is necessary because the API
# must return a mortality probability, not just a binary prediction.
# Raw SVM outputs only decision function scores, not probabilities.

# %%
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# LinearSVC does not provide probabilities, so we wrap it in CalibratedClassifierCV for probability estimates
svm_pipeline = Pipeline([
    ("preprocessor", preprocessor_separate),
    ("classifier", CalibratedClassifierCV(
        estimator=LinearSVC(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        cv=5,
    )),
])

# Train
svm_pipeline.fit(X_train, y_train)

# %%
# Evaluate
y_pred_svm = svm_pipeline.predict(X_test)
y_prob_svm = svm_pipeline.predict_proba(X_test)[:, 1]

# AUROC
auroc_svm = roc_auc_score(y_test, y_prob_svm)
print(f"AUROC: {auroc_svm:.3f}\n")

# Classification report (precision, recall, F1 per class)
print(classification_report(y_test, y_pred_svm, target_names=["Alive", "Expired"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_svm, display_labels=["Alive", "Expired"], ax=ax, cmap="Blues"
)
ax.set_title(f"Separate TF-IDF + Calibrated SVM (AUROC={auroc_svm:.3f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color:red">**Calibrated SVM Results:** </span>
# AUROC is the highest at ~82%, indicating the best ranking ability.
# However, the default 0.5 threshold produces very low recall (~12%)
# because the calibrated probabilities cluster below 0.5 for the
# minority class. Threshold tuning could recover recall, but adds
# complexity. The logistic regression model achieves nearly identical
# AUROC (~82%) with a much better precision/recall balance out of
# the box.

# %% [markdown]
# ### Threshold Tuning for Calibrated SVM
#
# The SVM achieves the best AUROC (~82%) but the default 0.5 threshold
# barely predicts anyone as Expired (~12% recall). This is a known
# behavior with calibrated SVMs on imbalanced data: the predicted
# probabilities for the minority class cluster well below 0.5 even
# when the model ranks patients correctly.
#
# Good AUROC means the ranking is right, the threshold is just wrong.
# We find the optimal threshold by sweeping the precision-recall curve
# and selecting the point that maximizes Expired F1.
#
# <span style="color:red">**Caveat:** </span>
# Threshold optimized on the test set for demonstration. In production,
# this should be done via cross-validation on the training set.

# %%
from sklearn.metrics import precision_recall_curve, f1_score

# Get precision, recall, and thresholds for the positive class
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_svm)

# Calculate F1 at each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

# Find optimal threshold
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"At this threshold:")
print(f"  Precision: {precisions[optimal_idx]:.3f}")
print(f"  Recall:    {recalls[optimal_idx]:.3f}")
print(f"  F1:        {f1_scores[optimal_idx]:.3f}")

# Apply optimal threshold
y_pred_svm_tuned = (y_prob_svm >= optimal_threshold).astype(int)

print(f"\n")
print(classification_report(y_test, y_pred_svm_tuned, target_names=["Alive", "Expired"]))

fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_svm_tuned, display_labels=["Alive", "Expired"], ax=ax, cmap="Blues"
)
ax.set_title(f"Calibrated SVM, Tuned Threshold={optimal_threshold:.3f} (AUROC={auroc_svm:.3f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color:red">**Tuned SVM Results:** </span>
# With the threshold lowered from 0.50 to 0.135, the SVM achieves
# the best AUROC (~82%) and Expired F1 (~41%) across all models.
# Recall matches logistic regression (~62%) with slightly fewer
# false positives (~58 vs ~64).
#
# <span style="color:red">**Caveat:** </span>
# The threshold was optimized on the test set for demonstration.
# In production, threshold selection should use cross-validation
# on the training set to avoid optimistic bias.

# %% [markdown]
# ### Ensemble: Soft Voting over Logistic Regression + Calibrated SVM
#
# Both the logistic regression and calibrated SVM achieved similar
# AUROC (~82%) using different decision mechanisms (probabilistic
# vs margin-based). Averaging their predicted probabilities via soft
# voting can reduce variance and improve calibration, particularly
# on small datasets where individual models are more sensitive to
# the specific train/test composition.

# %%
from sklearn.ensemble import VotingClassifier

ensemble_pipeline = Pipeline([
    ("preprocessor", preprocessor_separate),
    ("classifier", VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
            ("svm", CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
                cv=5,
            )),
        ],
        voting="soft",
    )),
])

# Train
ensemble_pipeline.fit(X_train, y_train)

# %%
# Evaluate
y_pred_ens = ensemble_pipeline.predict(X_test)
y_prob_ens = ensemble_pipeline.predict_proba(X_test)[:, 1]

# AUROC
auroc_ens = roc_auc_score(y_test, y_prob_ens)
print(f"AUROC: {auroc_ens:.3f}\n")

# Classification report (precision, recall, F1 per class)
print(classification_report(y_test, y_pred_ens, target_names=["Alive", "Expired"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_ens, display_labels=["Alive", "Expired"], ax=ax, cmap="Blues"
)
ax.set_title(f"Ensemble: LR + Calibrated SVM (AUROC={auroc_ens:.3f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color:red">**Ensemble Results:** </span>
# The ensemble achieves the best Expired precision (~41%) and fewest
# false positives (~19), but recall drops to ~31%. The SVM's
# conservative probability estimates pull the averaged probabilities
# down, causing the default 0.5 threshold to miss most expired
# patients. Same issue as the raw SVM, so we can try tuning the threshold.

# %% [markdown]
# ### Threshold tuning for Ensemble
#
# Same approach as the SVM threshold tuning. The ensemble's averaged
# probabilities are pulled down by the SVM component, so the default
# 0.5 threshold is too conservative. Sweeping the precision-recall
# curve to find the F1-optimal threshold.

# %%
# Tune threshold for ensemble
precisions_ens, recalls_ens, thresholds_ens = precision_recall_curve(y_test, y_prob_ens)
f1_scores_ens = 2 * (precisions_ens * recalls_ens) / (precisions_ens + recalls_ens + 1e-8)

# Find optimal threshold by maximizing F1 score
optimal_idx_ens = np.argmax(f1_scores_ens)
optimal_threshold_ens = thresholds_ens[optimal_idx_ens]

# Print optimal threshold
print(f"Optimal threshold: {optimal_threshold_ens:.3f}")

# Apply optimal threshold
y_pred_ens_tuned = (y_prob_ens >= optimal_threshold_ens).astype(int)

# Print classification report for tuned ensemble
print(f"\n")
print(classification_report(y_test, y_pred_ens_tuned, target_names=["Alive", "Expired"]))

# Confusion matrix for tuned ensemble
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_ens_tuned, display_labels=["Alive", "Expired"], ax=ax, cmap="Blues"
)
ax.set_title(f"Ensemble Tuned Threshold={optimal_threshold_ens:.3f} (AUROC={auroc_ens:.3f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color:red">**Tuned Ensemble Results:** </span>
# Threshold drops from 0.50 to ~0.39, recovering recall to ~55%
# while maintaining the best Expired F1 (~42%) across all models.
# The ensemble smooths out individual model weaknesses but still
# requires threshold tuning due to the SVM component.

# %% [markdown]
# ### Model Comparison Summary

# %%
from sklearn.metrics import f1_score, precision_score, recall_score

# Collect all model results for comparison
models = {
    "Baseline: Single TF-IDF + LR": (baseline_pipeline, None),
    "Separate TF-IDF + LR": (separate_pipeline, None),
    "Separate TF-IDF + XGBoost": (xgb_pipeline, None),
    "Separate TF-IDF + Calibrated SVM": (svm_pipeline, None),
    f"Separate TF-IDF + SVM (tuned {optimal_threshold:.3f})": (svm_pipeline, optimal_threshold),
    "Ensemble: LR + SVM": (ensemble_pipeline, None),
    f"Ensemble: LR + SVM (tuned {optimal_threshold_ens:.3f})": (ensemble_pipeline, optimal_threshold_ens)
}

# Evaluate each model and collect metrics
results = []
for name, (pipeline, threshold) in models.items():
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Apply threshold if specified, otherwise use default 0.5
    if threshold is not None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = pipeline.predict(X_test)
    
    # Collect metrics for the results table
    results.append({
        "Model": name,
        "AUROC": round(roc_auc_score(y_test, y_prob), 3),
        "Expired Precision": round(precision_score(y_test, y_pred), 3),
        "Expired Recall": round(recall_score(y_test, y_pred), 3),
        "Expired F1": round(f1_score(y_test, y_pred), 3),
        "False Positives": int(((y_pred == 1) & (y_test == 0)).sum()),
    })

# Create comparison dataframe and highlight the best AUROC, precision, recall, F1, and lowest false positives
comparison_df = pd.DataFrame(results).set_index("Model")

# Find best and second best for each metric
def highlight_top2(s, higher_is_better=True):
    """Highlight best in green and second best in yellow."""
    if higher_is_better:
        sorted_vals = s.nlargest(2)
    else:
        sorted_vals = s.nsmallest(2)
    
    styles = []
    for v in s:
        if v == sorted_vals.iloc[0]:
            styles.append("background-color: darkgreen; color: white")
        elif v == sorted_vals.iloc[1]:
            styles.append("background-color: darkgoldenrod; color: white")
        else:
            styles.append("")
    return styles

# Display the comparison table with highlights and formatted numbers
comparison_df.style.apply(
    highlight_top2, higher_is_better=True, subset=["AUROC", "Expired Precision", "Expired Recall", "Expired F1"]
).apply(
    highlight_top2, higher_is_better=False, subset=["False Positives"]
).format({
    "AUROC": "{:.3f}",
    "Expired Precision": "{:.3f}",
    "Expired Recall": "{:.3f}",
    "Expired F1": "{:.3f}",
    "False Positives": "{:.0f}",
})

# %% [markdown]
# <span style="color:red">**Final Production Model Selection:** </span>
# **Separate TF-IDF + Logistic Regression** is selected for deployment.
#
# While the baseline LR has marginally higher recall (~64% vs ~62%),
# the separate vectorizer model cuts false positives nearly in half
# (117 → 64) and improves precision (~19% → ~29%) with minimal
# recall loss. Tuned models (SVM, Ensemble) achieve slightly better
# AUROC and F1 but require post-hoc threshold selection. The separate
# LR offers:
#
#  - Strong recall (~62%) without threshold tuning
#  - Native probability calibration (no post-hoc threshold needed)
#  - Interpretability (linear coefficients map directly to TF-IDF terms)
#  - Operational simplicity for the API
#
# What would potentially improve the model in production:
#
#  - More data (the biggest impact)
#  - Numeric features alongside text (vitals, lab values, APACHE scores, etc.)
#  - N-grams to capture multi-word phrases like "acute renal" as features, but would need more data
#
# In a mortality prediction context, missing at-risk patients is
# more costly than flagging healthy ones for additional monitoring.
# The API returns a probability score, allowing the consuming
# application to apply its own threshold based on clinical risk
# tolerance.

# %% [markdown]
# <h2 style="color:cyan">3. Final Model Analysis and Export</h2>
#
#  1. Feature importance (top TF-IDF terms by class)
#  2. ROC curve for final model
#  3. Export final model pipeline

# %% [markdown]
# ### Feature Importance: Top TF-IDF Terms by Class
#
# Logistic regression coefficients map directly to TF-IDF features.
# Positive coefficients push toward Expired, negative toward Alive.
# Inspecting the top terms validates that the model learned clinically
# meaningful patterns rather than artifacts.

# %%
# Extract feature names and coefficients from the final model
preprocessor_fit = separate_pipeline.named_steps["preprocessor"]
classifier = separate_pipeline.named_steps["classifier"]

# Get feature names from each transformer
feature_names = []
for name, transformer, _ in preprocessor_fit.transformers_:
    if hasattr(transformer, "get_feature_names_out"):
        feature_names.extend(transformer.get_feature_names_out())
    elif name == "age":
        feature_names.append("age_numeric")

coefficients = classifier.coef_[0]

# Build dataframe of features and their coefficients
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", ascending=False)

# Top 20 for each class
top_expired = coef_df.head(20).reset_index(drop=True)
top_alive = coef_df.tail(20).sort_values("coefficient").reset_index(drop=True)

# Side by side display
display_df = pd.DataFrame({
    "Expired Feature": top_expired["feature"].values,
    "Expired Coef": top_expired["coefficient"].values,
    "Alive Feature": top_alive["feature"].values,
    "Alive Coef": top_alive["coefficient"].values,
})

display_df.index = range(1, 21)

display_df.style.bar(
    subset=["Expired Coef"], color="firebrick", align="left"
).bar(
    subset=["Alive Coef"], color="steelblue", align="right"
).format({
    "Expired Coef": "{:.4f}",
    "Alive Coef": "{:.4f}",
})

# %% [markdown]
# <span style="color:red">**Feature Importance Interpretation:** </span>
# The model has learned clinically meaningful patterns. Terms associated
# with mortality (hepatic, arrest, vasopressors, shock, resuscitate,
# norepinephrine) reflect critical illness and organ failure. Terms
# associated with survival (po, tabs, surgery, cabg, hypertension)
# reflect stable patients on oral medications or recovering from
# planned procedures.
#
# Some dosing units (kg, min, micrograms) appear as Expired-associated
# features. These are not clinical concepts but might correlate with
# high-dose IV medications typical of critical care. A future iteration
# could add these to the stop words list or apply domain-specific
# text preprocessing to reduce noise.
#
# **Clinical abbreviations in the feature list:**
#
#  - **PO**: "per os" (Latin) means "by mouth". Oral medication administration.
#  - **Tabs**: tablets. Oral dosage form.
#  - **CABG**: coronary artery bypass graft. Planned open-heart surgery
#    with low mortality (~1-2%). ICU admission is protocol, not acuity.
#  - **CSICU**: cardiac surgery ICU. Unittype associated with survival
#    because patients are recovering from planned cardiac procedures.

# %% [markdown]
# ### ROC Curve for Final Model

# %%
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_estimator(
    separate_pipeline, X_test, y_test, ax=ax, name="Separate TF-IDF + LR"
)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUROC=0.500)")
ax.set_title("ROC Curve: Final Model")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color:red">**Interpretation:** </span>
# Clear separation from the random baseline. The curve hugs the upper left, confirming strong discrimination. The staircase pattern is expected with only 42 expired patients in the test set.

# %% [markdown]
# ### Cross-Validation: Stability Assessment
#
# The final model is evaluated with shuffled stratified 5-fold
# cross-validation on the full dataset to assess performance
# stability beyond the single train/test split.

# %%
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Run 5-fold stratified CV on the full dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aurocs = cross_val_score(
    separate_pipeline, X, y,
    cv=cv, scoring="roc_auc", n_jobs=-1
)

print(f"5-Fold Stratified CV AUROC")
print(f"  Mean:  {cv_aurocs.mean():.3f} ± {cv_aurocs.std():.3f}")
print(f"  Range: {cv_aurocs.min():.3f} – {cv_aurocs.max():.3f}")
print(f"  Folds: {' | '.join(f'{x:.3f}' for x in cv_aurocs)}")

# %% [markdown]
# <span style="color:red">**Cross-Validation Results:** </span>
# The CV AUROC (~0.819 ± 0.027) is consistent with the single test
# split (~0.817), confirming the model's discrimination ability is
# stable across different data partitions. The fold range
# (0.789–0.855) indicates reasonable stability given the small
# dataset size and class imbalance.

# %% [markdown]
# ### Export final model pipeline
#
# Save the trained pipeline to disk for use in the API (Part 2).
# The pipeline includes preprocessing (TF-IDF vectorizers, one-hot
# encoder, age passthrough) and the logistic regression classifier
# as a single serialized object.

# %%
import joblib

model_path = PROJECT_ROOT / "submission" / "part2_deployment" / "models" / "final_model.joblib"
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(separate_pipeline, model_path)
print(f"Model saved to: {model_path}")
print(f"File size: {model_path.stat().st_size / 1024:.1f} KB")

# %% [markdown]
# The exported pipeline is self-contained: it accepts a DataFrame with
# the columns `diagnosis_text`, `admission_text`, `history_text`,
# `age_numeric`, `gender`, and `unittype`, and returns mortality
# probability scores. This is loaded directly by the API in Part 2
# at `submission/part2_deployment/models/final_model.joblib`.
