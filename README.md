# Multivariate Analysis of Financial Factors Influencing Consumer Credit Scores & Default Patterns

**Authors:** Kyle Atlas · Nicholas Bray · Yaseen Ghani · Daniel Kim · Shria Narapaneni  
**Tech:** R, RMarkdown, `tidyverse`, `glmnet`, `caret`, `randomForest`, `e1071`, `psych`/`FactoMineR`

---

## 📌 Summary

This repository investigates how **income, savings, spending patterns, and debt** shape **credit scores** and **loan default risk**. We use regularized regression for feature selection, linear regression for credit score prediction, PCA & factor analysis for structure discovery, and SVM/Random Forest to classify default risk.

**Key outcomes (from the final report):**
- Linear model explains **~80.7%** of variance in credit scores (Adjusted R² ≈ **0.8067**).
- **Debt** and **debt-to-income** ratios are the most influential drivers of creditworthiness.
- **Recent (last 6 months) spending** relates more strongly to credit scores than 12-month spending.
- **Random Forest** outperforms SVM for default classification (**Accuracy ~72.3%**, F1 ~83.1%; SVM Accuracy ~63.8%).

---

## 🔎 Problem Statement

Credit scoring systems are highly impactful yet opaque. We ask:

1. Which expenditure categories and financial ratios (short-run vs long-run) most influence credit scores?  
2. How do these relationships vary across income ranges?  
3. Can we predict defaults using demographics and spending factors with competitive accuracy?  
4. What latent structures (PCA/factors) organize the financial variables?

---

## Methods Overview

### Data
- **Source:** Kaggle “Credit Score” dataset (n ≈ 1,000 individuals) with 6- and 12-month expenditures, income, savings, debt, interaction terms, and ratios.  
- **Targets:** Continuous **CREDIT_SCORE**; Binary **DEFAULT**.

### Feature Engineering & Selection
- Handle multicollinearity and high dimensionality via:
  - **Lasso** / **Ridge** (`glmnet`) for selection & shrinkage (λ tuned by CV; example λ ≈ **0.0941** in final lasso run).
  - **Ratios**: expenditures relative to **debt**, **income**, **savings**.
  - **Temporal splits**: 6- vs 12-month spend.

### Modeling
- **Credit Score (regression):**  
  - Lasso preselection → **OLS** with parsimony (insignificant predictors removed).  
  - Example retained terms include: `INCOME`, `DEBT`, `R_DEBT_INCOME`, `T_HOUSING_12`, `CAT_GAMBLING`, `T_UTILITIES_12`, `T_EDUCATION_12`, `T_CLOTHING_12`, `CAT_DEPENDENTS` (log-score transform shown in code snippet in report).

- **Default (classification):**  
  - **SVM (RBF)** tuned with `sigma≈0.08`, `cost≈0.25` (class weighting for imbalance).  
  - **Random Forest** tuned via grid search (e.g., **500 trees**, `mtry=2`, `nodesize=1`, unlimited depth).  
  - **Evaluation:** 80/20 split, **5-fold CV**, Accuracy/Precision/Recall/F1 + confusion matrices.

### Structure Discovery
- **PCA** (loadings identify “Ratios to Debt” component most correlated with credit score; **PC2–score r ≈ 0.421**; PC1 explains more variance overall but relates weakly to score).  
- **Factor Analysis** (varimax) → **4 factors** explaining **~62.7%** variance:
  1) Ratios to **Debt**; 2) Necessities vs **Savings**; 3) **12-month** category spends; 4) **Luxury** ratios to income/savings.

---

## Results Snapshot

### Credit Score Regression
- **Adjusted R² ~ 0.8067** with a compact set of predictors.
- **Debt level** and **R_DEBT_INCOME** dominate importance.
- **Short-run (6-mo) spending** matters more than 12-mo spend for score variation.
- Residuals widen at higher income/expenditure levels, hinting at nonlinearities/complex behaviors.

### Default Classification

| Model | Accuracy | Precision | Recall | F1 |
|------:|:--------:|:---------:|:------:|:--:|
|  SVM  |  63.8%   |   74.8%   | 74.8%  |74.8%|
|   RF  |  72.3%   |   73.9%   | 95.1%  |83.1%|

_RF beats SVM across metrics; class weighting improved SVM recall but hurt balance._

---

## Interpretation & Takeaways

- **Debt management is pivotal**. Ratios tied to **debt** (utilities, groceries, entertainment, housing vs debt) load strongly on components most associated with credit score.  
- **Income and savings help**, but positive effects can be offset when expenses outpace them.  
- **Recent behaviors** (6-month expenditures) have outsized influence versus 12-month patterns → practical advice: focus on near-term financial discipline.  
- Model generalizability may drop for **higher-income** individuals due to more complex behavior; consider nonlinear/interaction terms or tree-based/boosted models for those segments.

---

## Limitations

- Minimal demographics (e.g., **no age**): possible hidden biases.
- **12-month window** may miss major long-term credit events (auto/home loans).  
- **Macroeconomics** (rates, inflation) not modeled.  
- PCA components were **not** fed into classifiers due to time constraints.

**Future work:** Elastic Net; Gradient Boosting/XGBoost; KNN; shallow NNs; longer panels; incorporate macro factors & richer demographics; interaction/nonlinearity modeling; partial dependence/SHAP for explainability.

---

### 1) Prerequisites
- **R ≥ 4.2**, **RStudio** (recommended)

### 2) Install Packages

```r
# core
install.packages(c("tidyverse","readr","dplyr","ggplot2","stringr","lubridate"))
# modeling
install.packages(c("glmnet","caret","randomForest","e1071"))
# multivariate
install.packages(c("psych","FactoMineR","factoextra","corrplot"))
# utilities
install.packages(c("car","janitor","scales","knitr","rmarkdown"))

