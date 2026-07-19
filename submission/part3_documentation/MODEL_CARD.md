# Model Card: ICU Mortality Prediction from Clinical Text

*Following the Model Card framework (Mitchell et al., 2019)*

---

## Model Details

- **Model Name:** ICU Mortality NLP Classifier
- **Version:** 1.0.0
- **Model Type:** Logistic Regression with TF-IDF text vectorization and one-hot encoded demographics
- **Architecture:** Scikit-learn pipeline with three separate TF-IDF vectorizers (one per text input field), OneHotEncoder for categorical demographics, and passthrough for numeric age. The separate vectorizer design allows the model to learn that the same clinical term carries different predictive weight depending on whether it appears in a diagnosis, admission note, or past medical history.
- **Training Date:** February 2026
- **Framework:** Python 3.12, scikit-learn 1.8.0
- **Developer:** Adam Blake
- **License:** For assessment purposes only. Not licensed for clinical use.
- **Contact:** See resume for contact details

---

## Intended Use

### Primary Intended Uses
- **Clinical decision support:** Flag ICU patients at elevated mortality risk based on text-based clinical documentation from the first 24 hours of admission.
- **Triage prioritization:** Help clinical teams allocate monitoring resources by identifying high-risk patients earlier in their ICU stay.
- **Research and demonstration:** Proof-of-concept that unstructured clinical text contains meaningful mortality signal, even without numeric vitals or lab values.

### Primary Intended Users
- Clinical informaticists evaluating NLP approaches for mortality prediction
- Data science teams at healthcare organizations building clinical risk models
- Hospital administrators assessing feasibility of text-based early warning systems

### Out-of-Scope Uses
- **Not for autonomous clinical decision-making.** This model should not be used as the sole basis for treatment decisions, code status changes, or resource allocation without clinician review.
- **Not for use outside the ICU setting.** The model was trained exclusively on ICU admission data and may not generalize to general ward, emergency department, or outpatient populations.
- **Not for use on non-English clinical text.** Training data is English-language clinical documentation from U.S. hospitals.
- **Not for individual patient prognosis communication.** Model outputs should not be shared directly with patients or families as mortality probabilities.
- **Not for use beyond 24-hour admission window.** The model uses only data from the first 24 hours of ICU admission. Clinical text from later in the stay was excluded during training to prevent temporal data leakage.

---

## Training Data

### Dataset Source
eICU Collaborative Research Database Demo v2.0.1, an open-access dataset from PhysioNet containing de-identified health data from over 2,500 ICU admissions across 20 hospitals in the United States.

- **Source:** https://physionet.org/content/eicu-crd-demo/2.0.1/
- **Citation:** Pollard, T., Johnson, A., Raffa, J., Celi, L. A., Badawi, O., & Mark, R. (2019). eICU Collaborative Research Database Demo (version 2.0.1). PhysioNet.

### Data Description
The dataset contains 30+ tables covering patient demographics, diagnoses, medical history, vital signs, lab measurements, clinical notes, assessments, treatment information, and care plan documentation. Text features were extracted from 14 tables containing 30 text columns across three classification buckets: categorical text (22 columns), hierarchical structured text (7 columns), and mixed text (1 column).

### Preprocessing Steps
1. **Table selection:** Tables with ≥50% patient coverage were retained; low-coverage tables excluded.
2. **Text column classification:** Each text column was classified as categorical, hierarchical structured, mixed, numeric artifact, or removed based on content inspection.
3. **Temporal filtering:** Only clinical events occurring within the first 24 hours of ICU admission (offset ≤ 1440 minutes) were included to prevent data leakage from information that would not be available at prediction time.
4. **Text aggregation:** Categorical and hierarchical columns were deduplicated (DISTINCT); mixed text was ordered by clinical event offset. Hierarchical delimiters (pipes and slashes) were replaced with spaces for tokenization.
5. **Placeholder filtering:** Empty strings and "Unknown" values were excluded during aggregation.
6. **API field mapping:** Aggregated columns were combined into three text fields matching the API specification: `diagnosis_text`, `admission_text`, and `history_text`.
7. **Demographics:** Age string converted to numeric ("> 89" → 90 per HIPAA convention, missing → median of 66). Gender and ICU unit type retained as categorical features.

### Train/Test Split
- **Method:** Stratified 80/20 split (preserving class distribution)
- **Training set:** ~1,993 stays (~170 expired, ~8.5%)
- **Test set:** ~499 stays (~42 expired, ~8.4%)
- **Random state:** 42 (reproducible)

---

## Evaluation Data

### Description
The evaluation dataset is the held-out 20% stratified test split from the same eICU demo dataset. It contains ~499 ICU stays with the same class distribution as the training set (~8.4% mortality rate).

### Motivation
A stratified split was chosen to ensure the minority class (Expired) is represented in both sets given the severe class imbalance (~92% Alive, ~8% Expired). The eICU demo dataset is the only dataset available for this assessment; no external validation set was used.

### Limitations
- The test set contains only ~42 expired patients, which limits the precision of performance estimates for the minority class.
- The evaluation data comes from the same source distribution as training data. Performance on external hospital systems is unknown.
- The demo dataset is a subset of the full eICU database and may not fully represent the diversity of ICU populations.

---

## Metrics

**Note:** Metrics are approximate (~) to account for minor variations across different environments when the notebook is re-run.

### Performance Metrics

| Metric | Value |
|---|---|
| AUROC | ~0.817 |
| Expired Precision | ~0.29 |
| Expired Recall | ~0.62 |
| Expired F1 | ~0.39 |
| Alive Precision | ~0.96 |
| Alive Recall | ~0.86 |
| Alive F1 | ~0.91 |
| Overall Accuracy | ~0.84 |
| False Positives | ~64 |
| False Negatives | ~16 |
| 5-Fold Stratified CV AUROC | ~0.819 ± 0.027 |

### Metric Selection Rationale
- **AUROC** was chosen as the primary ranking metric because it is threshold-independent and appropriate for imbalanced classification. It measures how well the model discriminates between classes regardless of the operating point.
- **Expired Recall** was prioritized over precision because in a mortality prediction context, missing an at-risk patient (false negative) is more costly than flagging a healthy patient for additional monitoring (false positive).
- **Expired F1** provides a balanced view of precision and recall for the minority class.

### Decision Threshold
The default 0.5 probability threshold is used. The API returns raw probability scores, allowing consuming applications to select their own threshold based on clinical risk tolerance. During model development, threshold tuning was explored for alternative models (Calibrated SVM, Ensemble) and demonstrated that recall can be traded against precision depending on the clinical use case.

### Confidence and Variability
- Performance was validated with shuffled stratified 5-fold cross-validation (AUROC ~0.819 ± 0.027), confirming stability across data partitions.
- The staircase pattern in the ROC curve reflects the small number of positive examples (~42 expired patients in the test set).
- Approximate 95% confidence intervals were not computed but would be wide given the small test set size. For example, with 42 expired patients and ~62% recall, the Clopper-Pearson interval would be roughly 46%–76%.

---

## Factors

### Relevant Factors
The following patient characteristics may influence model performance:

- **Age:** Older patients have higher baseline mortality. The model includes age as a numeric feature.
- **Gender:** Included as a one-hot encoded feature. The training data contains Male, Female, and a small number of missing values.
- **ICU Unit Type:** Different ICU types (Med-Surg, SICU, MICU, Cardiac, Neuro, etc.) serve different patient populations with different baseline mortality rates. Included as a one-hot encoded feature.
- **Hospital/EHR System:** All 20 hospitals in the eICU demo dataset use the Philips eICU telehealth system, so documentation structure is consistent
  across the training data. Performance on clinical text from other EHR platforms (Epic, Cerner, etc.) is unknown and likely degraded due to
  differences in documentation templates and terminology.
- **Diagnosis Complexity:** Patients with multiple comorbidities generate more clinical text, which could influence TF-IDF feature density.

### Evaluated Factors
Disaggregated evaluation by gender and ICU unit type was not performed in this assessment due to the small test set size (42 expired patients), which would produce unreliable subgroup estimates. With sufficient data, disaggregated evaluation across gender, age group, unit type, and their intersections would be recommended.

---

## Ethical Considerations

### Potential Biases
- **Documentation bias:** The model relies on clinical text, which reflects the documentation practices of clinicians. Patients who receive more documentation (e.g., sicker patients, patients in teaching hospitals) may be better represented in the feature space. Conversely, patients with language barriers or those receiving less attentive documentation may have sparser text features.
- **Demographic representation:** The eICU demo dataset represents a subset of U.S. hospitals and may not reflect the demographic diversity of all ICU populations. Race and ethnicity were available in the dataset but were intentionally excluded as model features.
- **Label bias:** The target variable (hospitaldischargestatus) captures in-hospital mortality only. Patients who died shortly after discharge or were transferred to hospice are labeled as "Alive," which underestimates true mortality.
- **Historical care patterns:** The model may learn patterns that reflect historical care disparities rather than true clinical risk. For example, if certain patient populations receive less aggressive treatment, the model could learn to associate their documentation patterns with survival rather than the underlying clinical trajectory.

### Fairness Considerations
- Race and ethnicity were excluded from model features to avoid encoding demographic proxies for mortality. However, other features (unit type, hospital) may correlate with demographic composition.
- The model uses `class_weight="balanced"` to upweight the minority class during training, which helps prevent the model from simply predicting all patients as Alive.
- No formal fairness audit (e.g., equalized odds, demographic parity) was performed due to the small dataset size.

### Privacy and Data Protection
- The eICU demo dataset is de-identified and publicly available under PhysioNet's open data use agreement.
- Ages above 89 are grouped as "> 89" per HIPAA Safe Harbor de-identification standards.
- No patient names, dates of birth, or other personally identifiable information are used as model features.
- The model does not store or log patient data at inference time.

---

## Caveats and Recommendations

### Known Limitations
1. **Small dataset:** ~2,500 stays with ~212 expired patients is insufficient for production deployment. Production clinical models typically train on hundreds of thousands of observations.
2. **Text-only features:** The model uses only clinical text, excluding numeric features (vital signs, lab values, APACHE scores) that are strong mortality predictors. This was a deliberate design choice for the NLP-focused assessment but limits overall performance.
3. **No external validation:** All evaluation was performed on data from the same source distribution. Performance on data from different hospital systems, EHR platforms, or clinical documentation practices is unknown.
4. **Single train/test split for model selection:** Model comparison was performed on a single 80/20 split. Cross-validation on the final model 
  confirmed stability (AUROC ~0.819 ± 0.027), but ideally CV would be used during model selection as well.
5. **Noise features:** Some top-weighted features (kg, min, micrograms) are dosing units rather than clinical concepts. These correlate with critical care medications but represent noise that a domain-specific stopword list could address.
6. **No temporal validation:** The train/test split is random, not temporal. A production model should be validated on data from a later time period to assess concept drift.

### Groups with Potentially Different Performance
- **Surgical vs. medical ICU patients:** Surgical ICU patients (especially CABG/cardiac surgery) have lower baseline mortality and more standardized documentation, which may inflate the model's apparent discrimination ability.
- **Short-stay patients:** Patients who expire within the first few hours of ICU admission may have very sparse 24-hour text features, potentially reducing the model's sensitivity for the most acute cases.
- **Patients with atypical documentation:** Hospitals or units with non-standard documentation practices may produce text that diverges from the training distribution.

### Recommendations
1. **Do not deploy as-is for clinical use.** This model is a proof-of-concept demonstrating that clinical text contains meaningful mortality signal.
2. **Augment with numeric features** (vitals, labs, severity scores) for production-grade performance.
3. **Validate on external data** from different hospital systems before any deployment consideration.
4. **Perform disaggregated evaluation** across demographic groups with a larger dataset.
5. **Implement monitoring** for model drift if deployed, tracking prediction distributions and calibration over time.
6. **Use the probability score, not the binary prediction,** to allow clinical teams to set their own risk threshold based on their specific workflow and risk tolerance.

---

## References

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D., & Gebru, T. (2019). Model Cards for Model Reporting. *FAT* '19*. https://doi.org/10.1145/3287560.3287596
- Pollard, T., Johnson, A., Raffa, J., Celi, L. A., Badawi, O., & Mark, R. (2019). eICU Collaborative Research Database Demo (version 2.0.1). PhysioNet. https://doi.org/10.13026/4mxk-na84