

---

## 📖 Overview
Predicting house prices involves dealing with highly skewed features, missing values, and a large number of categorical variables. This project focuses on heavy feature engineering, outlier detection using `IsolationForest`, and a robust **blended ensemble model** (Lasso + XGBoost + CatBoost) to achieve strong generalization and minimize the Root Mean Squared Logarithmic Error (RMSLE).

## 📊 Dataset
The dataset is the **Ames Housing dataset**, which was compiled by Dean De Cock for use in data science education. 
- `train.csv`: The training set containing features and the target variable `SalePrice`.
- `test.csv`: The test set containing features (target variable needs to be predicted).
- `data_description.txt`: Full description of each column.

---

## 🛠️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
* Analyzed the target variable (`SalePrice`) using Histograms and Q-Q plots to check for normality.
* Detected and removed manual extreme outliers (e.g., `GrLivArea > 4000` & `SalePrice < 300000`).
* Employed **Isolation Forest** to programmatically detect and analyze multivariate anomalies within numeric data.

### 2. Data Preprocessing & Feature Engineering
* **Handling Missing Values:**
  * Imputed missing categorical variables with meaningful 'None' or the statistical mode.
  * Replaced missing continuous variables with `0` (e.g., Garage features) or the column mean.
* **Feature Construction:**
  * Calculated property age metrics (`YearBuilt`, `YearRemodAdd`, `GarageYrBlt` relative to `YrSold`).
  * Consolidated square footage into comprehensive features (`TotalFlrSF`, `BsmtFinSF`).
  * Aggregated bathroom counts into a single continuous `Total_Bath` feature.
* **Skewness Correction:** Applied `log1p` transformation to heavily skewed numeric variables (skew > 0.5) to normalize distributions.
* **Encoding:**
  * Custom ordinal mapping for quality and condition features (e.g., `Ex` -> 5, `Po` -> 1).
  * One-Hot Encoding (`pd.get_dummies`) for the remaining low-cardinality categorical features.

### 3. Modeling & Ensemble Strategy
To prevent overfitting on the high-dimensional feature space, I used a combination of tree-based and regularized linear models evaluated via **5-Fold Cross-Validation**:
1. **Lasso Regression:** Regularized linear model (uses `StandardScaler` to normalize features).
2. **XGBoost Regressor:** Gradient-boosted decision trees.
3. **CatBoost Regressor:** Gradient boosting with built-in robust handling for mixed feature data.

**Ensemble Blending:**
Predictions were blended using a weighted average optimized for cross-validation performance:
> `Final Prediction = (0.2 * Lasso) + (0.4 * XGBoost) + (0.4 * CatBoost)`

---

## 📈 Results & Evaluation
The model's performance is evaluated using **Root Mean Squared Error (RMSE)** on the log-transformed `SalePrice` (which effectively calculates RMSLE). 

* **Cross-Validation Strategy:** 5-Fold CV (`KFold`)
* Target variable was log-transformed (`np.log1p`) during training and exponentiated (`np.expm1`) for the final submission.
* The final blended model achieves a stable CV RMSE score. *(See notebook output for exact fold metrics).*
### Model Weights & Performance Breakdown
> **Final Blended Score (5-Fold CV Average RMSE):** `0.12651`
* **Lasso Regression:** 20%
* **XGBoost Regressor:** 40%
* **CatBoost Regressor:** 40%

---
* Download the data from the [Kaggle Competition Page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place `train.csv` and `test.csv` in the root/data directory.
