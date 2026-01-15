# Insurance Fraud Detection

A machine learning project focused on **detecting fraudulent insurance claims** in a highly imbalanced dataset, with emphasis on **experimental rigor, interpretability, and decision-aware evaluation**, not just model accuracy.

---

## Why This Matters

Insurance fraud detection requires balancing two competing risks:  
missing fraudulent claims and over-flagging legitimate ones.

This project demonstrates how **careful exploratory analysis, appropriate metric selection, threshold tuning, and error analysis** can translate model outputs into **operationally meaningful fraud signals** rather than raw predictions.

---

## What I Did

- Explored policy, claim, and temporal drivers of fraud  
- Performed association analysis tailored to feature types (nominal, ordinal, cyclic)  
- Built leakage-safe preprocessing pipelines using `ColumnTransformer`  
- Compared multiple classification model families under severe class imbalance  
- Tuned decision thresholds to explicitly control precision–recall tradeoffs  
- Conducted detailed post-model error analysis on the held-out test set  

---

## Modeling Highlights

- Stratified **train / validation / test** split to mirror production workflows  
- Pipeline-based preprocessing (encoding, scaling, cyclic feature handling)  
- Association testing via **Cramér’s V** (nominal) and **Spearman correlation** (ordinal/cyclic)  
- Model comparison across linear, tree-based, and ensemble methods  
- Hyperparameter tuning using **PR-AUC (average precision)** as the primary optimization metric  

**Best model:** XGBoost (class-weighted, PR-AUC optimized)

### Final Test Performance (Threshold = 0.45)

- **Recall:** 0.8261  
- **Precision:** 0.1735  
- **Accuracy:** 0.7549  
- **ROC AUC:** 0.8559  
- **PR AUC:** 0.2315  

These results reflect a deliberate emphasis on fraud capture in a low-prevalence setting, with threshold tuning used to balance recall against investigation volume.

---

## Key Insights

- Fraud risk is driven primarily by **policy structure and claim context**, rather than demographics alone  
- Deductible level, fault attribution, and liability structure consistently emerge as strong risk indicators  
- Temporal features contribute through interactions rather than standalone effects  
- Most remaining errors occur near the decision threshold, indicating **borderline cases rather than confident misclassification**  
- Error analysis highlights segments where fraud signal is weaker, suggesting opportunities beyond model complexity alone  

---

## Stack

Python · pandas · numpy · scikit-learn · xgboost · scipy · matplotlib · seaborn

---

## Next Steps

- Feature enrichment using external or historical claim signals  
- Segment-aware thresholding for different policy types  
- Cost-sensitive evaluation aligned with investigation capacity  
