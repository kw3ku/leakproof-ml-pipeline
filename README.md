# leakproof-ml-pipeline
Data leakproof Machine Learning Pipeline for production ready analysis. Handling End-to-end data cleaning and model training.
# leakproof-ml-pipeline

A production-grade, end-to-end machine learning pipeline built on the California Housing dataset. This repository demonstrates how to architect a robust predictive system by maintaining strict causal data boundaries, rectifying skewed real-world feature distributions, and applying cross-validated model tuning.

## 🏗️ Project Architecture & Workflow

To maintain absolute data integrity, this pipeline follows a strict, non-negotiable sequential execution flow to ensure that statistical data properties from unseen data never contaminate the training cycle.

[Raw Data Input]│▼[1. Train/Test Split] ───► Isolate Test Partition (20%)│▼[2. Imputation Stage] ───► Compute Median on X_train ONLY ───► Backfill Both Splits│▼[3. Statistical EDA]  ───► Pearson Correlation Matrix via X_train│▼[4. Feature Engineers] ───► Fit Yeo-Johnson on X_train ───────► Project Both Splits│▼[5. Model Benchmarking] ──► Baseline Linear Regression vs. Optimized Random Forest


---

## 🛠️ Key Engineering Features

### 1. Zero-Leakage Data Hygiene
* **The Problem:** Many baseline workflows execute global transformations (imputation, scaling) prior to dataset splitting. This leaks the variance, mean, and range of the test matrix into the model's training process.
* **The Architecture:** This pipeline immediately splits the data into an **80/20 train/test distribution**. Missing entries in `total_bedrooms` (207 records) are resolved by calculating the median strictly from the training dataset and projecting it forward.

### 2. Parametric Skew Remediation via Yeo-Johnson
* **The Limitation of Log (`log1p`):** Standard log transformations overcorrected heavily right-skewed volume features like `population` (raw skew: 4.93) into structural left-skew configurations (~ -1.0).
* **The Solution:** We implemented a parametric **Yeo-Johnson Power Transformation** to dynamically optimize lambda values per variable, drawing extreme distribution curves smoothly back into normal profiles.

### 3. Spatial Isolation
* Geographical attributes (`latitude` and `longitude`) along with `housing_median_age` are intentionally isolated from the power transformer to preserve unwarped spatial coordinate vectors.

---

## 📊 Performance & Skewness Scorecard

### Distribution Skew Correction
The pipeline centers highly asymmetrical metrics safely within the optimal symmetric threshold of **-0.5 to 0.5**:

| Feature Name | Raw Training Skew | Post-Transformation Skew | Engineering Action |
| :--- | :---: | :---: | :---: |
| `population` | **4.9358** | **0.1106** | Yeo-Johnson Transformed |
| `total_rooms` | **4.1473** | **0.1213** | Yeo-Johnson Transformed |
| `total_bedrooms` | **3.4595** | **0.1079** | Imputed + Transformed |
| `households` | **3.4104** | **0.1095** | Yeo-Johnson Transformed |
| `median_income` | **1.6466** | **-0.0025** | Yeo-Johnson Transformed |
| `latitude` | **0.4659** | **0.4659** | Isolated (Preserved Spatial) |

### Baseline vs. Challenger Model Outcome

| Framework Model | Root Mean Squared Error (RMSE) | R² Score | Result Status |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | ~\$69,000.00 | ~0.6400 | Underfits Spatial Data |
| **Random Forest (Optimized)** | **~\$49,000.00** | **~0.8100** | **Production Winner** |

* **Architectural Insight:** Linear models attempt to evaluate latitude and longitude as flat lines, entirely missing regional coordinate groupings. The Random Forest easily establishes non-linear grid splits, creating high-value bounding boxes over coastal and metropolitan tech hubs to reduce total error by roughly \$20,000.

---

## 🚀 How to Run the Pipeline

### Prerequisites
Ensure you have Python 3.10+ and the required packages installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution
Run the complete automated pipeline using the terminal script entry:
```bash
python model_pipeline.py
```
