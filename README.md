# 💳 Credit Card Fraud Detection using SVM

> **A machine learning classification project that identifies potentially fraudulent credit card transactions using Support Vector Machine (SVM), feature scaling, exploratory data analysis, and imbalance-aware classification.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)](https://scikit-learn.org/stable/supervised_learning.html)

---

## 📌 Project Overview

Credit card fraud detection is a binary classification problem where the objective is to distinguish legitimate transactions from potentially fraudulent ones.

This project develops an **SVM-based fraud detection pipeline** using the well-known credit card transaction dataset. The workflow covers data loading, exploratory analysis, data-quality checks, feature scaling, class-imbalance handling through SVM class weighting, model training, hyperparameter-oriented experimentation, and performance evaluation.

The trained model and scaler are also serialized using Python's `pickle` module so they can be reused for future inference.

---

## 🎯 Business Problem

Financial institutions process a large number of transactions every day, while fraudulent transactions represent only a small fraction of total activity.

A fraud detection system therefore needs to identify suspicious transactions while minimizing incorrect classifications of legitimate transactions.

The central objective of this project is:

> **Can machine learning identify fraudulent credit card transactions from transaction-level features while accounting for severe class imbalance?**

---

## 🔎 Project Objectives

* Explore the structure and distribution of credit card transaction data.
* Identify the imbalance between legitimate and fraudulent transactions.
* Perform data-quality checks for missing values and duplicate records.
* Prepare numerical features for machine learning.
* Apply feature scaling using `StandardScaler`.
* Build an SVM classification model.
* Address class imbalance using `class_weight='balanced'`.
* Evaluate the model using multiple classification metrics.
* Analyze fraud-specific precision, recall, and F1-score.
* Calculate ROC-AUC and visualize model discrimination.
* Save the trained SVM model and scaler for future reuse.

---

# 📂 Dataset

The project uses a credit card transaction dataset containing anonymized transaction features.

The original dataset contains transaction information represented by:

| Feature      | Description                                |
| ------------ | ------------------------------------------ |
| `Time`       | Time elapsed between transactions          |
| `V1` – `V28` | Anonymized numerical transaction features  |
| `Amount`     | Transaction amount                         |
| `Class`      | Target variable: `0 = Normal`, `1 = Fraud` |

For this project, a reproducible random sample of **10,000 transactions** was selected using:

```python
df.sample(10000, random_state=42)
```

The resulting working dataset contains:

* **10,000 transactions**
* **31 columns**
* **9,984 normal transactions**
* **16 fraudulent transactions**

This corresponds to a highly imbalanced classification problem.

---

# 🧠 Why Fraud Detection Is Different

A normal classification problem might rely heavily on accuracy.

Fraud detection is different.

If the dataset contains overwhelmingly more legitimate transactions, a model can achieve very high accuracy while still failing to identify a meaningful number of fraudulent transactions.

Therefore, this project evaluates the model using:

### Accuracy

Measures the overall percentage of correctly classified transactions.

### Precision

Measures how many transactions predicted as fraud were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were successfully detected.

### F1-Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between normal and fraudulent transactions across classification thresholds.

The project therefore considers fraud-specific metrics alongside accuracy rather than relying on accuracy alone.

---

# 🔄 Machine Learning Workflow

```text
Raw Transaction Data
        │
        ▼
Random Sampling
        │
        ▼
Data Understanding
        │
        ▼
EDA & Class Distribution
        │
        ▼
Data Quality Checks
        │
        ├── Missing Values
        └── Duplicate Records
        │
        ▼
Feature / Target Separation
        │
        ▼
Train-Test Split
        │
        ▼
StandardScaler
        │
        ▼
SVM Classifier
        │
        ├── RBF Kernel
        ├── Balanced Class Weights
        └── Probability Estimates
        │
        ▼
Model Prediction
        │
        ▼
Performance Evaluation
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1-Score
        └── ROC-AUC
        │
        ▼
Model Serialization
```

---

# 🧹 Data Preparation

The notebook performs several data-quality checks before model training.

### Missing Values

The dataset was checked using:

```python
df.isnull().sum()
```

No missing values were identified in the working sample.

### Duplicate Records

Duplicate rows were checked using:

```python
df.duplicated().sum()
```

No duplicate records were identified in the working sample.

---

# ⚖️ Handling Class Imbalance

The working dataset contains:

```text
Normal Transactions : 9,984
Fraudulent          :    16
```

This creates a severe class imbalance.

Instead of treating both classes equally during SVM training, the model uses:

```python
class_weight='balanced'
```

This automatically assigns greater importance to the minority class during training.

The approach allows the model to account for the significantly smaller number of fraudulent transactions without simply optimizing for the majority class.

---

# 📏 Feature Scaling

SVM models are sensitive to the scale of numerical features.

The project therefore uses:

```python
StandardScaler()
```

to standardize the feature values before training the classifier.

The fitted scaler is saved separately as:

```text
scaler.pkl
```

This allows the same preprocessing transformation to be reused when making future predictions.

---

# 🤖 Model

## Support Vector Machine — SVM

The project uses the Support Vector Classifier from Scikit-learn.

The implemented configuration includes:

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
```

### Important Parameters

| Parameter      | Value      | Purpose                                       |
| -------------- | ---------- | --------------------------------------------- |
| `kernel`       | `rbf`      | Captures non-linear relationships             |
| `C`            | `1.0`      | Controls regularization                       |
| `gamma`        | `scale`    | Controls influence of individual observations |
| `class_weight` | `balanced` | Gives greater weight to the minority class    |
| `probability`  | `True`     | Enables probability estimates                 |
| `random_state` | `42`       | Supports reproducibility                      |

These settings are reflected in the trained estimator saved by the notebook.

---

# 📊 Model Performance

The trained SVM model produced the following results on the test set:

| Metric          |      Score |
| --------------- | ---------: |
| **Accuracy**    | **99.90%** |
| **ROC-AUC**     | **0.9978** |
| **F1-Score**    | **0.5000** |
| Fraud Precision |       1.00 |
| Fraud Recall    |       0.33 |
| Fraud F1-Score  |       0.50 |

The test set contained **1,997 normal transactions and only 3 fraudulent transactions**, so the fraud-specific metrics should be interpreted cautiously because of the very small number of positive examples.

### Classification Report

```text
              precision    recall    f1-score    support

Normal           1.00       1.00       1.00       1997
Fraud            1.00       0.33       0.50          3
```

The results demonstrate a strong ability to distinguish classes according to ROC-AUC, but the fraud recall of **0.33** indicates that the model did not identify all fraudulent transactions in this particular test split.

---

# 📈 Evaluation Strategy

The project evaluates the classifier using multiple perspectives.

### Confusion Matrix

Used to understand:

* True Positives
* True Negatives
* False Positives
* False Negatives

### ROC Curve

Used to examine the model's class-separation capability across thresholds.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

### ROC-AUC

Provides a threshold-independent view of the classifier's ability to distinguish between normal and fraudulent transactions.

---

# 💾 Saved Model Artifacts

The trained components are serialized using Python's `pickle` module:

```text
svm_model.pkl
scaler.pkl
```

### `svm_model.pkl`

Contains the trained SVM classifier.

### `scaler.pkl`

Contains the fitted feature-scaling transformation.

Keeping the model and preprocessing object together is important because future prediction data must undergo the same transformation used during model training.

---

# 📁 Repository Structure

```text
Credit-Card-Fraud-Detection-Project/
│
├── SVM Fraud Detection.ipynb
│
├── svm_model.pkl
│
├── scaler.pkl
│
└── README.md
```

The current repository contains the notebook and the two serialized model artifacts.

---

# 🛠️ Technology Stack

| Technology           | Purpose                   |
| -------------------- | ------------------------- |
| **Python**           | Programming language      |
| **Pandas**           | Data manipulation         |
| **NumPy**            | Numerical computation     |
| **Matplotlib**       | Visualization             |
| **Seaborn**          | Statistical visualization |
| **Scikit-learn**     | Machine learning          |
| **SVM / SVC**        | Classification            |
| **StandardScaler**   | Feature scaling           |
| **Jupyter Notebook** | Development environment   |
| **Pickle**           | Model serialization       |

The notebook imports and uses Pandas, NumPy, Seaborn, Matplotlib, SVC, StandardScaler, train-test splitting, GridSearchCV, and classification metrics.

---

# 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/aryan2026-mishra/Credit-Card-Fraud-Detection-Project.git
```

### 2. Navigate to the project

```bash
cd Credit-Card-Fraud-Detection-Project
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
SVM Fraud Detection.ipynb
```

### 6. Dataset

Place the required `creditcard.csv` dataset in the project directory before executing the notebook.

---

# 🔮 Future Improvements

The current project provides a strong foundation but can be extended significantly.

### 1. Better Class-Imbalance Strategy

Future versions could experiment with:

* SMOTE
* Random undersampling
* Stratified cross-validation
* Hybrid sampling strategies

### 2. Model Comparison

Compare SVM against:

* Logistic Regression
* Random Forest
* XGBoost
* Gradient Boosting
* Isolation Forest

### 3. Hyperparameter Optimization

Systematically tune:

```text
C
gamma
kernel
class_weight
```

using cross-validation and an appropriate scoring metric.

### 4. Threshold Optimization

Instead of relying only on the default classification threshold, optimize the threshold based on the business cost of:

* Missing a fraudulent transaction
* Incorrectly flagging a legitimate transaction

### 5. Explainability

Add model-interpretability techniques to understand which features contribute most strongly to fraud predictions.

### 6. Deployment

The serialized model can be integrated into:

* Streamlit
* FastAPI
* Flask
* REST API
* Real-time transaction monitoring pipeline

---

# ⚠️ Important Limitation

The project uses a **10,000-row random sample**, resulting in only **16 fraudulent transactions in the working dataset** and only **3 fraud cases in the test set**.

Therefore, the reported performance should be considered an **experimental result on this sample**, not evidence that the model would achieve the same performance in a production fraud-detection environment.

The low fraud recall in the test set also shows why fraud detection should not be evaluated using accuracy alone.

This limitation is an important consideration for future iterations of the project.

---

# 🎓 Skills Demonstrated

This project demonstrates practical experience in:

* Exploratory Data Analysis
* Data Quality Checking
* Binary Classification
* Imbalanced Classification
* Feature Scaling
* Support Vector Machines
* Model Evaluation
* ROC-AUC Analysis
* Precision / Recall Analysis
* F1-Score
* Confusion Matrix
* Model Serialization
* Machine Learning Workflow
* Python & Scikit-learn

---

# 👨‍💻 Author

## Aryan Mishra

**B.Tech — Computer Science & Engineering**

Aspiring Data Analyst | Machine Learning | Python | SQL | Power BI

* **LinkedIn:** https://www.linkedin.com/in/aryan-mishra-61561b298/
* **GitHub:** https://github.com/aryan2026-mishra

---

## ⭐ Project Summary

This project demonstrates how machine learning can be applied to a highly imbalanced financial classification problem. By combining exploratory analysis, feature scaling, imbalance-aware SVM classification, and multiple evaluation metrics, the project provides a practical foundation for developing more robust credit-card fraud detection systems.
