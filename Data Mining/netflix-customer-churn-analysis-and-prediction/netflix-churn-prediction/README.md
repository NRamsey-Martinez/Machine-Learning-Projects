# Netflix Churn Prediction & Error Analysis

This project explores user churn behavior in a Netflix-style subscription dataset and develops a high-performing machine learning model to identify at-risk users. The focus is not only on predictive accuracy, but also on **interpretability, experimental rigor, and actionable error analysis**.

---

## Project Overview

Customer churn is a critical challenge for subscription-based businesses. This project aims to:

- Understand behavioral and demographic drivers of churn
- Build a robust churn prediction model using structured data
- Compare statistical feature selection strategies
- Optimize model performance for business-relevant metrics
- Analyze model errors to surface practical retention insights

---

## Data & Problem Setup

- **Target:** Binary churn indicator
- **Features:** User demographics, subscription details, engagement behavior, and recency metrics
- **Challenge:** Highly skewed engagement variables and strong behavioral separability

The dataset was split into **training, validation, and test sets** using stratified sampling to preserve class balance.

---

## Exploratory Data Analysis

Key findings from EDA include:

- **User engagement is the strongest predictor of churn**, with clear separation in watch behavior and inactivity metrics.
- Engagement features exhibit **heavy right skew**, motivating log transformations rather than aggressive outlier removal.
- Churn concentration varies meaningfully across **subscription tier, payment method, and recency-of-login bins**.
- Price-related features matter primarily when combined with low engagement.

EDA incorporated univariate analysis, bivariate comparisons, churn-rate heatmaps, effect size calculations, and correlation analysis.

---

## Feature Engineering & Preprocessing

All preprocessing was implemented inside scikit-learn pipelines to prevent data leakage and ensure consistency across splits:

- Log transformations for skewed numeric features
- Binning of recency metrics
- One-hot encoding for nominal categorical features
- Ordinal encoding for subscription tiers
- Scaling of numeric variables

---

## Feature Selection Strategy

Two complementary statistical feature selection methods were evaluated:

- **Chi-squared tests** for categorical features
- **ANOVA F-tests** for numeric features

Feature sets were evaluated independently and then combined using parallel pipelines, allowing the model to leverage both categorical associations and numeric separability.

---

## Modeling & Evaluation

Multiple models were evaluated using cross-validation, including:

- Logistic Regression
- Support Vector Machines
- Random Forest
- Gradient Boosting
- AdaBoost
- Multi-layer Perceptron

**AdaBoost** emerged as the top-performing model. Hyperparameters were tuned using GridSearchCV with **recall** as the primary optimization metric.

### Final Test Performance
- Recall: ~0.996  
- Precision: ~0.992  
- Accuracy: ~0.994  
- ROC AUC: ~0.999  

Given the unusually strong performance, additional care was taken to validate experimental design and rule out data leakage.

---

## Error Analysis & Interpretability

Post-hoc error analysis was conducted to understand where the model fails:

- **False negatives** cluster among moderately engaged users, suggesting churn driven by factors beyond inactivity alone.
- **False positives** often involve recently active users with declining engagement, reflecting recall-oriented model behavior.
- Certain categorical groups exhibit higher misclassification rates, indicating opportunities for segment-specific strategies.

Feature importance analysis using AdaBoost highlights engagement and recency metrics as dominant drivers.

---

## Key Takeaways

- Behavioral disengagement is the primary driver of churn.
- Pipeline-based preprocessing and feature selection are essential for reliable evaluation.
- High model performance must be interpreted cautiously and validated rigorously.
- Error analysis provides actionable insights beyond aggregate metrics.

---

## Tools & Technologies

- **Python** (pandas, numpy)
- **scikit-learn** (pipelines, feature selection, ensemble models)
- **matplotlib / seaborn** (visualization)
- **scipy** (statistical analysis)

---

## Next Steps

Potential extensions include:

- Threshold tuning based on retention cost tradeoffs
- Segment-specific churn models


