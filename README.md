# Kaggle_Challenge

## Steps for predicting the insolvency of a company using a working model

1. Feature selection: univariate selection (ANOVA, chi-squared test), dropping columns, scaling (min-max scaling, Standardization), regularisation (L1 Lasso, L2 Ridge)
2. Feature engineering: need to create new variables out of existants? Dummification? new ratios? debt-to-equity ratio, liquidity ratio, profit margin...
3. EDA: clean data (outliers, nans), correlated features, feature selection (drop columns).  
3.1. Missing data: dropping columns? Filling rows? (avg, median, predictives models).  
3.2. Unbalaced? Insolvent companies < solvent companies: SMOTE, XGBoost, etc. ?  
4. Getting variable independant: defining X and y.
5. Model selection: Logistic or Linear regression, Gradient Boosting, Support Vector Machines (SVM), Decision Tree, Random Forest.  
5.1. Cross-validation for testing the model.  
5.2. Model evaluation: matrix confusion, precision, recall, F1-score, ROC-AUC.  
6. Ajusting Overfitting (L1 or L2 in linear models), cross-validation for adjusting hyperparameters, pruning for random forest or gradient boosting.
7. Deployment.

# Insolvency Prediction Project – Code Summary

## 1. Data Loading & Exploration
- Loads insolvency data from CSV (`data_insolvency.csv`).
- Defines target variable (`Bankrupt?`) and features.
- Performs exploratory data analysis (EDA): checks data types, missing values, outliers, and feature distributions.
- Documents and explains financial ratios and metrics in the dataset.

## 2. Feature Selection & Engineering
- Uses correlation analysis to identify and drop highly correlated features.
- Applies ANOVA F-test to select statistically significant features (p < 0.05).
- Cleans data by removing outliers using the IQR method.
- Scales features using `StandardScaler`.
- Optionally engineers new features and ratios.

## 3. Handling Class Imbalance
- Notes severe imbalance: ~3% bankrupt vs. ~97% solvent.
- Tests multiple strategies:
  - `class_weight='balanced'` in logistic regression.
  - SMOTE (Synthetic Minority Oversampling Technique).
  - Random undersampling and oversampling.

## 4. Model Training & Evaluation
- **Logistic Regression:**
  - Trains with and without balancing techniques.
  - Uses cross-validation and regularization (L1/L2).
  - Evaluates with confusion matrix, F1-score, precision, recall, ROC-AUC.
  - Finds logistic regression struggles with severe imbalance.
- **Random Forest:**
  - Trains and evaluates using validation/test splits.
  - Shows improved results over logistic regression.
- **XGBoost:**
  - Handles imbalance with `scale_pos_weight`.
  - Performs hyperparameter tuning via GridSearchCV and RandomizedSearchCV.
  - Uses cross-validation for robust F1-score estimation.
  - Achieves high ROC-AUC and recall, but lower precision due to thresholding.

## 5. Results & Insights
- **Best Model:** XGBoost with tuned hyperparameters.
- **Metrics:**
  - ROC-AUC: ~0.99 (excellent discrimination).
  - Recall: 1.0 (no missed bankruptcies).
  - Precision: ~0.47 (many false positives).
  - F1-score: moderate, reflecting trade-off.
- **Confusion Matrix Example:**
  - 6,354 true negatives (solvent correctly identified)
  - 245 false positives (solvent flagged as bankrupt)
  - 0 false negatives (no missed bankruptcies)
  - 220 true positives (bankrupt correctly identified)

## 6. Recommendations
- Adjust classification threshold to reduce false positives.
- Consider business cost of false negatives vs. false positives.
- Provide probability scores for risk assessment.
- Investigate false positives for possible future bankruptcies.

## 7. Conclusions
- Tree-based models outperform logistic regression for this imbalanced problem.
- Model tuning and threshold adjustment are key for practical deployment.
- The project demonstrates a full pipeline: EDA, feature selection, balancing, modeling, and evaluation.

---

# Chronology of Insolvency Prediction Project Code

## 1. Data Import & Initial Setup
- Import libraries: pandas, numpy, sklearn, matplotlib, seaborn, etc.
- Load dataset (`data_insolvency.csv`) into a DataFrame.
- Define target (`Bankrupt?`) and features (`X`).

## 2. Exploratory Data Analysis (EDA)
- Print dataset info and data types.
- Document and explain financial ratios and metrics.
- Group features by type (income, expenses, balance sheet).

## 3. Feature Selection
- Plot correlation heatmap to identify highly correlated features.
- Drop features with correlation > 0.9 to reduce multicollinearity.
- Use ANOVA F-test to select statistically significant features (p < 0.05).

## 4. Data Cleaning
- Check for missing values (NaNs) and confirm none present.
- Visualize outliers with boxplots.
- Remove outliers using IQR method (various multipliers tested).

## 5. Feature Scaling & Regularization
- Scale features using `StandardScaler`.
- Test regularization: L1 (Lasso) and L2 (Ridge) with cross-validation.
- Select best regularization parameter (`C`) using LogisticRegressionCV.

## 6. Model Training: Logistic Regression
- Split data into train/test sets.
- Train logistic regression with L2 regularization.
- Evaluate with confusion matrix and F1-score.
- Address overfitting by adjusting regularization strength (`C`).

## 7. Handling Class Imbalance
- Note severe imbalance in target variable.
- Apply balancing techniques:
  - `class_weight='balanced'` in logistic regression.
  - SMOTE (oversampling minority class).
  - Random undersampling and oversampling.

## 8. Model Training: Tree-Based Models
- Train Random Forest classifier; evaluate on validation/test sets.
- Train XGBoost classifier with `scale_pos_weight` for imbalance.
- Compare results to logistic regression.

## 9. Hyperparameter Tuning
- Use GridSearchCV and RandomizedSearchCV to optimize XGBoost parameters.
- Select best model based on cross-validated F1-score.

## 10. Final Evaluation
- Calculate metrics: ROC-AUC, F1-score, precision, recall, confusion matrix.
- Achieve best results with XGBoost (high recall, moderate precision).

## 11. Conclusions & Recommendations
- Summarize findings and model performance.
- Recommend threshold adjustment and probability-based risk scoring.
- Document full pipeline and results.

---
**See `Project.ipynb` for detailed code and outputs.**


---

### **Model Performance (XGBoost + RandomizedSearchCV)**

- **ROC-AUC:** `0.9997`  
  *Extremely high. The model almost perfectly distinguishes between bankrupt and solvent companies.*

- **F1-score:** `0.7051`  
  *Good balance between precision and recall, especially for an imbalanced dataset.*

- **Precision:** `0.5446`  
  *Of all companies predicted as bankrupt, about 54% are actually bankrupt. There are some false positives.*

- **Recall:** `1.0`  
  *The model identifies **all** actual bankrupt companies (no false negatives).*

- **Confusion Matrix:**  
  ```
  [[6415  184]
   [   0  220]]
  ```
  - **True Negatives (6415):** Correctly predicted solvent companies.
  - **False Positives (184):** Predicted bankrupt, actually solvent.
  - **False Negatives (0):** None missed—excellent recall.
  - **True Positives (220):** Correctly predicted bankrupt companies.

---

### **Interpretation**

- **Strengths:**  
  - The model is highly effective at catching every bankrupt company (recall = 1.0).
  - Overall discrimination is excellent (ROC-AUC near 1).

- **Weaknesses:**  
  - Precision is moderate, meaning some solvent companies are incorrectly flagged as bankrupt (false positives).
  - This could be a concern if false alarms are costly, but is often acceptable in risk-sensitive domains.

- **Imbalanced Data:**  
  - The model handles class imbalance well, likely due to `scale_pos_weight` and careful tuning.

---

### **Business Implications**

- **No bankrupt company goes undetected** (no false negatives), which is critical for insolvency prediction.
- **Some solvent companies may be flagged as bankrupt** (false positives), so further review or secondary screening may be needed for flagged cases.

---

**Summary:**  
Your XGBoost model is highly reliable for identifying bankrupt companies, with a trade-off of some false alarms. This is generally preferred in financial risk scenarios, where missing a bankruptcy is much worse than a false alarm.