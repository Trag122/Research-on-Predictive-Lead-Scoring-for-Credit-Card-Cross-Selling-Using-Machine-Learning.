# Predictive Lead Scoring for Credit Card Cross-Selling

## Project Overview

This project introduces a hybrid machine learning framework to optimize predictive lead scoring for credit card cross-selling within the retail banking sector. It aims to transition financial institutions from inefficient, mass telemarketing campaigns to data-driven, customer-centric strategies by accurately predicting a customer's purchasing likelihood.

## Dataset & Preprocessing

* **Data Source:** The "Credit Card Lead Prediction" dataset sourced from Kaggle (`trainnckh.csv`).
* **Scale:** The dataset contains 245,725 historical customer records with 10 independent features and one binary target variable (`Is_Lead`).
* **Class Imbalance:** The data is highly skewed, consisting of 76.3% uninterested customers and only 23.7% potential leads.
* **Missing Values (MNAR):** Missing data in the `Credit_Product` feature was treated as a deliberate behavioral signal (Missing Not At Random) and mapped to a distinct "Unknown" category.
* **Scaling:** Continuous numerical features (`Age`, `Vintage`, `Avg_Account_Balance`) were normalized using `StandardScaler` to prevent magnitude bias.

## Methodology (Hybrid Framework)

The analytical framework operates through a rigorous three-phase pipeline:

1. **Phase 1: Unsupervised Behavioral Segmentation:**
* Applied the K-Prototypes algorithm with 'Cao' initialization to effectively process mixed CRM data (both categorical and numerical).
* Segmented the customer base into four distinct topological clusters, generating a highly predictive engineered feature named `Cluster_Label`.

2. **Phase 2: Supervised Classification & Benchmarking:**
* Benchmarked three Gradient Boosted Decision Trees (CatBoost, XGBoost, LightGBM) and one Tabular Deep Learning model (TabNet).
* Addressed structural class imbalance by applying a dynamic `scale_pos_weight` of 3.2157 to the target functions.
* Utilized Automated Machine Learning (AutoML) via Optuna with Tree-structured Parzen Estimators (TPE) to globally optimize hyperparameters and maximize the ROC-AUC metric.

3. **Phase 3: Explainable AI (XAI):**
* Extracted Global Feature Importance to decipher the algorithmic "black box" and bridge the gap between mathematical accuracy and business actionability.

## Evaluation Results & Insights

* **Champion Model:** **CatBoost** emerged as the absolute best-performing architecture, achieving an optimal ROC-AUC score of 0.8741. It successfully handled discrete categorical features using Ordered Target Statistics without suffering from target leakage or dimensionality explosion.
* **Deep Learning Shortcomings:** The TabNet model systematically underperformed (ROC-AUC 0.8691) and suffered from severe overfitting. This was caused by an inductive bias mismatch, as neural networks struggle to map the sharp, discontinuous logic inherent to tabular financial data.
* **Feature Importance:** XAI validation revealed that the `Credit_Product` status and the engineered `Cluster_Label` were the strongest predictive drivers, significantly outperforming raw metrics like account balance.

## Author
@All Rights Reserved.
* **Author:** Nguyen Pham Quynh Trang
* **Project Reference:** Student Research Report KH.NC.SV.25_15
