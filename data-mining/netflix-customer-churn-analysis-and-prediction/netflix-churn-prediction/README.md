# Netflix Churn Prediction

A machine learning project focused on **predicting customer churn** in a Netflix-style subscription dataset, with emphasis on **experimental rigor, interpretability, and error analysis**, not just accuracy.

---

## Why This Matters

Subscription businesses need to identify churn risk early and act efficiently.  
This project demonstrates how careful feature engineering, statistical validation, and post-model error analysis can translate predictions into **actionable retention insights**.

---

## What I Did

- Explored behavioral and demographic drivers of churn  
- Built leakage-safe preprocessing pipelines  
- Compared statistical feature selection strategies  
- Trained and tuned multiple classification models  
- Performed detailed error analysis to understand model failures  

---

## Modeling Highlights

- Stratified train / validation / test splits  
- Pipeline-based preprocessing (encoding, scaling, transforms)  
- Feature selection via Chi-squared and ANOVA F-tests  
- Model comparison across linear, tree-based, and ensemble methods  

**Best model:** AdaBoost (optimized for recall)

**Final test performance**
- Recall: ~0.996  
- Precision: ~0.992  
- Accuracy: ~0.994  
- ROC AUC: ~0.999  

---

## Key Insights

- Engagement and recency metrics dominate churn risk  
- Skewed behavioral features benefit from log transformation  
- False negatives cluster among moderately engaged users  
- Error analysis reveals segment-specific retention opportunities  

---

## Stack

Python · pandas · numpy · scikit-learn · scipy · matplotlib · seaborn

---

## Next Steps

- Threshold tuning based on retention cost tradeoffs  
- Segment-specific churn models  

