# Financial Fraud Detection System

An ongoing machine learning project exploring fraud detection through behavioral analysis, feature engineering, and reusable preprocessing pipelines.

The project focuses on building a reliable data foundation for detecting suspicious financial transactions across datasets with different structures, characteristics, and limitations.

## Project Overview

Financial fraud detection is a challenging machine learning problem due to highly imbalanced data, evolving transaction behavior, and the difficulty of distinguishing legitimate anomalies from fraudulent activity.

This project investigates these challenges through two complementary datasets:

- **Credit Card Fraud:** Real-world transaction data with anonymized numerical features and extreme class imbalance.
- **PaySim:** A large-scale synthetic financial dataset containing transaction sequences, account identifiers, and behavioral information.

The current milestone focuses on **data understanding, preprocessing, feature engineering, and reusable pipeline development** before moving to model training and evaluation.

## 📊 Datasets

| Dataset                                                                             | Transactions | Characteristics                                                     |
| ----------------------------------------------------------------------------------- | -----------: | ------------------------------------------------------------------- |
| [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |      284,807 | Real transactions, 492 fraud cases, 31 columns                      |
| [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)                            |    6,362,620 | Synthetic transactions, 11 columns, sequential transaction behavior |

### Credit Card Fraud

The dataset contains transactions made by European cardholders over two days in September 2013. Most features (`V1`–`V28`) are anonymized PCA-transformed variables.

The fraud rate is approximately **0.17%** which makes class imbalance a central challenge.

Its limited behavioral information requires a different feature engineering strategy from PaySim.

### PaySim

PaySim simulates mobile money transactions using aggregated information from real financial logs.

Its sequential structure enables behavioral analysis through transaction frequency, amount patterns, and counterparty diversity.

A particularly important dataset limitation is that fraudulent transactions are cancelled in the simulation. Therefore, account balance fields are excluded from the fraud detection feature pipeline to avoid relying on misleading information.

## ⚙️ Data Engineering & Feature Engineering

The current implementation includes:

- Exploratory data analysis (EDA)
- Data cleaning and duplicate handling
- Dataset sampling and split preparation
- Custom feature transformers
- Reusable preprocessing pipelines
- Pipeline serialization and validation

### Behavioral Features — PaySim

To capture transaction behavior at different time scales, the project implements:

| Feature category       | Examples                                          |
| ---------------------- | ------------------------------------------------- |
| Transaction frequency  | Counts over 24-hour, 72-hour, and 7-day windows   |
| Transaction amounts    | Rolling sums and average amounts                  |
| Counterparty diversity | Number of unique destinations within time windows |
| Transaction categories | One-hot encoding of transaction types             |

These features aim to capture unusual transaction frequency, spending patterns, and rapid changes in recipient behavior.

The `step` column represents simulated hours rather than real-world timestamps. Therefore, temporal features are interpreted within the simulation's time framework.

### Credit Card Feature Engineering

Due to the anonymized PCA features and lack of customer-level identifiers, behavioral aggregation is not supported by the dataset.

Feature engineering is therefore intentionally limited to a logarithmic transformation of transaction amounts:

`log_amount = ln(1 + Amount)`

This reduces the influence of highly skewed transaction amounts while preserving their relative ordering.

## Reusable Preprocessing Pipelines

The project uses **scikit-learn pipelines and custom transformers** to organize preprocessing into modular, reusable components.

**PaySim pipeline:**

`Cleaning → Transaction History → Behavioral Features → Type Encoding → Scaling`

**Credit Card pipeline:**

`Cleaning → Log Amount Transformation → Scaling`

The pipelines are designed to support consistent preprocessing across future experiments.

Pipeline artifacts are serialized using `joblib`, and their loading and transformation behavior are validated through sample inputs.

## 📁 Project Structure

```
Financial-Fraud-Detection-System/
│
├── data/
│   ├── eda/
│   │   ├── creditcard_balanced_eda.csv
│   │   └── paysim_balanced_eda.csv
│   │
│   └── sample/
│       ├── creditcard_sample.csv
│       └── paysim_sample.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── feature_engineering.ipynb
│   └── preprocessing.ipynb
│
├── scripts/
│   ├── create_eda_balanced.py
│   ├── create_paysim_splits.py
│   └── create_samples.py
│
├── src/
│   └── data/
│       ├── behavioral_feature_engineer.py
│       ├── cleaners.py
│       ├── creditcard_pipeline.py
│       ├── data_splitter.py
│       ├── feature_scaler.py
│       ├── fit_and_save_pipelines.py
│       ├── log_amount_transformer.py
│       ├── paysim_pipeline.py
│       ├── transaction_history_preprocessor.py
│       └── transaction_type_encoder.py
│
├── tests/
│   ├── test_creditcard_pipeline.py
│   ├── test_data_splitter.py
│   └── test_paysim_pipeline.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Large datasets and generated artifacts are excluded from version control. Small samples are provided where appropriate for inspection and development.

## Current Status & Next Sprint

**Completed: Sprint 1 — Data Engineering**

The current milestone establishes the data preparation foundation, including exploratory analysis, feature engineering, custom transformers, and reusable preprocessing pipelines.

### Sprint 2 — Baseline Models & Evaluation

The next milestone will focus on:

- Establishing baseline models, including Isolation Forest, Local Outlier Factor (LOF), and Autoencoders.
- Defining appropriate train, validation, and test strategies.
- Reviewing temporal and user-level data leakage risks.
- Evaluating performance under extreme class imbalance.
- Comparing models using precision, recall, F1-score, and PR-AUC.
- Investigating synthetic transactions and edge cases to assess robustness.

The broader objective is to develop a reliable fraud detection framework that combines behavioral feature engineering with systematic model evaluation.

---

_This project is actively evolving. Current results and implementation details will be updated as subsequent experiments are completed._
