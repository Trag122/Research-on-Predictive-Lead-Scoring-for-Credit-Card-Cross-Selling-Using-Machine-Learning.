# Research-on-Predictive-Lead-Scoring-for-Credit-Card-Cross-Selling-Using-Machine-Learning.
This study designed and validated a hybrid machine learning framework to optimize predictive lead scoring in the banking sector. The study integrated the K-Prototypes algorithm, efficiently processing mixed CRM attributes to generate a multidimensional latent variable (Cluster_Label).
Methodology
2.2.1. Overall framework Brief and gap resolution
To address the limitations of conventional lead scoring methods, such as reliance
on excessively aggregated sampling and opaque architectures, this study proposes a
hybrid framework combining K-prototypes and a supervised classification model. This
architecture operates through three stages:
1. Unsupervised extraction of behavioral profiles using K-Prototypes.
2. Four-dimensional supervised classification using CatBoost, XGBoost,
LightGBM, and TabNet, tightly optimized through Bayesian search.
3. Application of Explainable AI (XAI) to ensure transparency in prediction. This
end-to-end process ensures that the final model is not only mathematically
superior but also operational.

Phase 1: Unsupervised Behavioral Segmentation
Traditional clustering algorithms such as K-Means are mathematically incompatible with the heterogeneous nature of bank CRM data, which contains many categorical variables. Therefore, phase 1 implements the K-Prototypes algorithm. To handle mixed CRM data without diluting the categorical signal, this algorithm uses a combined difference measure, calculating the squared Euclidean distance for continuous variables and the Hamming distance for categorical features. A significant weakness of partition-based clustering is the high sensitivity to the selection of initial cluster centroids, where random initialization often causes the model to get stuck in suboptimal local minima. To ensure maximum stability, the cluster centroids are explicitly initialized using a mathematically rigorous 'Cao' initialization method. Unlike random selection, Cao's algorithm determines data point density based on the frequency of categorical values, proactively selecting initial cluster centroids representing the highest-density regions while maximizing the distance between them.
Based on industry standards for banking customer segmentation, the algorithm divides the dataset into exactly four distinct structural clusters. The output is a highly predictive, engineered feature, "Cluster_Label". By integrating this label back into the dataset, the analytics framework synthesizes individual demographic and transactional
attributes into a unified, mathematically optimized behavioral context.

Phase 2: SOTA Benchmarking and Imbalance Correction
Before building the model, the enriched dataset was split into a training set and a test set in an 80/20 ratio using stratified sampling to preserve the inherent target
distribution. To directly address the severe class imbalance, a dynamic "scale_pos_weight" was calculated based on the exact ratio of negative to positive cases in the training set. To determine the optimal classifier, the analytical framework establishes a comprehensive benchmark comparing the Deep Learning model with the
Gradient Boosting trio:
1. XGBoost: This model is selected for its proven exact greedy algorithm and
advanced regularization mechanisms.
2. LightGBM: Included to evaluate the efficacy of depth-first structural
optimization and gradient-based sampling on large-scale datasets.
3. CatBoost: This is a Symmetric oblivious tree. CatBoost was purposefully
chosen for its Ordered Target Statistics, which directly neutralize prediction
shift and target leakage.
4. TabNet: A Tabular Deep Learning model that challenges neural networks,
integrating sparse sequential attention mechanisms (governed by structural
parameters such as "n_steps", "n_d", and "n_a") to simulate explainable
decision tree logic within a deep learning framework.
Manual hyperparameter tuning is entirely discarded. The framework utilizes
Optuna with Tree-structured Parzen Estimators (TPE) to perform automatic Bayesian
optimization on all four algorithms. By explicitly maximizing the ROC-AUC metric
over multiple trials per model, the framework ensures that the comparison results
reflect true architectural superiority.

Phase 3: Model Interpretability (XAI)
High predictive accuracy alone is insufficient for deployment in a tightly regulated financial environment. To bridge the gap between algorithmic performance and business actionability, phase 3 integrates Explainable Artificial Intelligence (XAI) to clarify the decision limitations of the optimal model identified in phase 2. Regardless of whether the empirically superior architecture is tree-based or neural network-based, this framework will extract the Global Characteristic Importance score to clearly quantify the predictive power of all independent variables.
