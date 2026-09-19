# leakproof-ml-pipeline

A production-grade, end-to-end machine learning pipeline built on the California Housing dataset. This repository demonstrates how to architect a robust predictive system by maintaining strict causal data boundaries, rectifying skewed real-world feature distributions, eliminating truncation collection biases, and applying cross-validated model tuning.

## 🏗️ Project Architecture & Workflow

To maintain absolute data integrity, this pipeline follows a strict, non-negotiable sequential execution flow to ensure that statistical data properties from unseen data never contaminate the training cycle.

[Raw Data Input]│▼[1. Cap Filtering]   ───► Remove $500K Truncated Ceilings│▼[2. Train/Test Split] ───► Isolate Test Partition (20%)│▼[3. Imputation Stage] ───► Compute Median on X_train ONLY ───► Backfill Both Splits│▼[4. Statistical EDA]  ───► Pairwise Pearson Correlation Matrix via X_train│▼[5. Feature Engineering] ─► Fit Yeo-Johnson on X_train ───────► Project Both Splits│▼[6. Model Benchmarking] ──► Baseline Linear Regression vs. Optimized Random Forest



---

## 🛠️ Key Engineering Features

### 1. Data Truncation Cleanse
* **The Problem:** Exploratory analysis via feature box plots revealed a severe accumulation of outlier records locked exactly at **\$500,000** for `median_house_value`. This indicates an upper-bound truncation during the census collection phase. Leaving these capped values forces algorithms to learn an artificial flat-line trend, causing severe prediction distortions.
* **The Solution:** We applied a strict pre-split filter to isolate genuine market behavior and evaluate a clean continuous target profile:
  ```python
  df_pipeline = df_numeric[df_numeric['median_house_value'] < 500000].copy()
  ```

### 2. Zero-Leakage Data Hygiene
* **The Problem:** Many baseline workflows execute global transformations (imputation, scaling) prior to dataset splitting. This leaks the variance, mean, and range of the test matrix into the model's training process, creating overly optimistic validation results that collapse in production environments.
* **The Architecture:** This pipeline immediately splits the data into an **80/20 train/test distribution**. Missing entries in `total_bedrooms` are resolved by calculating the median strictly from the training dataset (`X_train`) and projecting it forward onto the test split (`X_test`).

### 3. Parametric Skew Remediation via Yeo-Johnson
* **The Limitation of Traditional Math Operators:** Standard log transformations (`log1p`) overcorrected heavily right-skewed volume features like `population` (raw skew: 4.96) into structural left-skew configurations (~ -1.0). Conversely, Square Root (`np.sqrt`) transformations under-corrected the metrics, leaving them intensely asymmetric (skews > 1.1).
* **The Solution:** We implemented a parametric **Yeo-Johnson Power Transformation** to dynamically optimize lambda values per variable, drawing extreme distribution curves smoothly back into normal profiles.
* **Spatial Isolation:** Geographical attributes (`latitude` and `longitude`) along with `housing_median_age` are intentionally isolated from the power transformer to preserve unwarped spatial coordinate vectors.

---

## 📊 Exploratory Data Analysis & Feature Engineering

### Distribution Normalization Proof
By plotting our three data states side-by-side, we visually and mathematically prove why parametric power transformations outperform static math operators. 

![Transformation Comparison Proof](transformation_comparison_proof.png)

The pipeline successfully centers highly asymmetrical metrics safely within the optimal symmetric threshold of **-0.5 to 0.5**:

| Feature Name | Raw Skewness | Square Root Skewness | Yeo-Johnson Skewness | Engineering Action |
| :--- | :---: | :---: | :---: | :--- |
| `population` | 4.9614 | 1.2313 | 0.1112 | Yeo-Johnson Normalization |
| `total_rooms` | 4.2263 | 1.3833 | 0.1201 | Yeo-Johnson Normalization |
| `total_bedrooms` | 3.4806 | 1.2286 | 0.1070 | Imputed + Yeo-Johnson |
| `households` | 3.4092 | 1.1370 | 0.1089 | Yeo-Johnson Normalization |
| `median_income` | 0.9138 | 0.2780 | -0.0009 | Yeo-Johnson Normalization |
| `median_house_value`| 0.7952 | 0.2729 | -0.0232 | Cap Filtered + Yeo-Johnson |

### Feature Correlation Strengths
Evaluating the pairwise Pearson correlation matrix on our raw training data reveals the fundamental linear relationships guiding our estimators. 

![Feature Correlation Heatmap](train_correlation_heatmap.png)

* **The Primary Driver (`median_income` = 0.65):** Household earnings display the most powerful linear relationship with our target variable, making it the most critical predictor feature.
* **The Multicollinearity Block:** A dense, highly correlated square connects `total_rooms`, `total_bedrooms`, `population`, and `households` (correlations from **0.86 to 0.97**). This indicates clear information redundancy, a space where linear coefficients degrade but tree splits thrive.
* **The Geographic Mirror (`latitude` = -0.15):** The negative correlation coefficient mathematically mirrors California's economic geography. Moving south (decreasing latitude) yields a reliable home value premium, while values drop as you move north or inland.

---

## 🏆 Baseline vs. Challenger Model Outcome

| Framework Model | Root Mean Squared Error (RMSE) | R² Score | Result Status |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | \$62,494.46 | 0.5926 | ⚠️ Underfits Non-Linear Spatial Trends |
| **Random Forest (Challenger)** | **\$45,726.66** | **0.7819** | **🏆 Production Winner** |

### 🧠 Performance Analysis
* **The Spatial Victory:** Linear Regression attempts to evaluate `latitude` and `longitude` as entirely separate straight vectors, scoring a modest \(R^2\) of `0.5926` because it gets severely penalized by high-value northern exceptions like Silicon Valley. 
* **The Random Forest Leverage:** The Random Forest algorithm evaluates spatial coordinates simultaneously. By building multi-dimensional grid splits, it naturally isolates localized high-value coastal regions across both northern and southern coordinates, recapturing **\$16,767.80** in precision metrics per home.

---

## 🚀 How to Run the Pipeline

### Prerequisites
Ensure you have Python 3.10+ and the required packages installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tabulate
```

### Execution
Run the complete automated pipeline using the terminal script entry:
```bash
python model_pipeline.py
```