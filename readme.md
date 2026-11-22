# Classification of Terrorist Attack Severity Using Pre-Incident Characteristics

A Machine Learning Research Project | BRAC University | CSE437

## Overview

Terrorist attacks vary in their potential impact and severity. Understanding the characteristics associated with different severity levels can support research into preventive security planning and resource allocation.

This project investigates whether **pre-incident characteristics** from the Global Terrorism Database (GTD) can be used to classify terrorist attack severity into four categories:

- None
- Low
- Medium
- High

The study compares five machine learning algorithms and applies hyperparameter optimization and SMOTE to address class imbalance. It also examines feature importance to understand which characteristics contribute most strongly to model predictions.

> **Research focus:** Multiclass severity classification using pre-incident characteristics, rather than relying on post-incident consequences as predictive input features.

## Research Objectives

The project addresses three research questions:

1. To what extent can pre-incident characteristics from the GTD classify terrorist attacks into multiple severity levels?
2. Which machine learning approaches provide the most reliable classification under class imbalance?
3. Which pre-incident characteristics contribute most strongly to predicted severity levels, and how can these contributions be interpreted?

## Dataset

The study uses the **Global Terrorism Database (GTD)**, a historical database of terrorist incidents.

| Dataset information | Description |
|---|---|
| Original records | 181,691 |
| Original variables | 135 |
| Historical coverage | 1970–2017 |
| Records after preprocessing | 171,378 |
| Final predictive features | 41 |
| Target variable | `nkill` (used only to construct severity labels) |

The dataset contains information about attack types, targets, weapons, geographical characteristics, temporal characteristics, and other incident attributes.

Dataset source: [Global Terrorism Database (GTD)](https://www.start.umd.edu/gtd/)

*Note: Access to the original GTD dataset may require registration or acceptance of its terms of use.*

## Severity Classification

The target variable is constructed from the number of fatalities (`nkill`).

Four severity classes are defined:

| Number of fatalities | Severity class |
|---:|---|
| 0 | None |
| 1–2 | Low |
| 3–10 | Medium |
| More than 10 | High |

The `nkill` variable is used exclusively to generate the target label. It is excluded from the predictive feature set to avoid directly providing the model with the outcome used to define its target.

### Class Distribution

After preprocessing, the dataset contains 171,378 incidents.

| Severity class | Incidents | Percentage |
|---|---:|---:|
| None | 88,149 | 51.43% |
| Low | 50,723 | 29.59% |
| Medium | 24,517 | 14.31% |
| High | 7,989 | 4.66% |
| **Total** | **171,378** | **100%** |

The distribution demonstrates significant class imbalance. In particular, the High severity class represents only 4.66% of the dataset.

This makes accuracy alone insufficient for evaluating model performance.

## Theoretical Background

### 1. Multiclass Classification

Multiclass classification is a supervised machine learning task in which a model assigns an observation to one of more than two possible categories.

In this project, the model predicts one of four severity classes:

`None → Low → Medium → High`

The model learns patterns from historical incident characteristics and their corresponding severity labels.

### 2. Pre-Incident Characteristics

Pre-incident characteristics are attributes intended to represent information that could reasonably be available before the final consequences of an incident are observed.

The research excludes outcome-related variables from the predictive feature set, including casualty and wound outcomes, attack success indicators, hostage outcomes, ransom-related information, property-damage outcomes, and other post-incident consequences.

This separation is important because the objective is to investigate early severity classification rather than classification using information revealed after an attack.

### 3. Class Imbalance

Class imbalance occurs when some categories contain significantly more observations than others.

In this dataset:

- None: 51.43%
- Low: 29.59%
- Medium: 14.31%
- High: 4.66%

A model may perform well on the majority class while performing poorly on minority classes.

### 4. SMOTE

Synthetic Minority Over-sampling Technique (SMOTE) addresses class imbalance by generating synthetic samples for minority classes using neighboring training observations.

In this project, SMOTE is applied **only to the training data after the train-test split**.

The test dataset remains unchanged to preserve a realistic evaluation distribution and prevent information leakage.

### 5. Feature Importance

Feature importance measures how strongly individual features contribute to a trained model's predictive behavior.

The project uses global feature importance from the tuned Random Forest classifier to identify influential geographical, temporal, target-related, and weapon-related characteristics.

Feature importance indicates predictive association, not causation.

## Methodology

The experimental workflow consists of the following stages.

### Step 1: Data Loading

Load the Global Terrorism Database and examine its structure, variables, missing values, and duplicate records.

### Step 2: Data Cleaning

The preprocessing pipeline includes:

- Checking and removing duplicate records where necessary.
- Excluding variables with missing-value proportions greater than or equal to 70%.
- Removing identifiers, free-text fields, redundant variables, and post-incident information.
- Removing observations without valid `nkill` values.

After removing observations with missing fatality information, 171,378 incidents remain.

### Step 3: Target Construction

Create four severity categories using the number of fatalities:

- 0 fatalities → None
- 1–2 fatalities → Low
- 3–10 fatalities → Medium
- More than 10 fatalities → High

The original `nkill` variable is then excluded from the model's input features.

### Step 4: Feature Engineering

Different preprocessing techniques are applied to numerical and categorical features.

| Feature type | Preprocessing technique |
|---|---|
| Numerical features | StandardScaler |
| Low-cardinality categorical features | One-Hot Encoding |
| High-cardinality categorical features | Label Encoding |

The final feature matrix contains 171,378 observations and 41 predictive features.

### Step 5: Train-Test Split

The dataset is divided using an 80:20 stratified train-test split.

| Dataset | Approximate records |
|---|---:|
| Training set | 137,102 |
| Testing set | 34,276 |

A fixed random state of 42 is used.

Stratification preserves the relative distribution of severity classes in both subsets.

### Step 6: Handling Class Imbalance

SMOTE is applied exclusively to the training data.

The test set is not oversampled.

### Step 7: Model Training

Five baseline classification algorithms are evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Support Vector Machine (SVM) with an RBF kernel
5. K-Nearest Neighbors (KNN)

### Step 8: Hyperparameter Optimization

Random Forest and XGBoost are selected for further optimization based on their baseline performance and suitability for structured nonlinear data.

RandomizedSearchCV with three-fold stratified cross-validation is used.

**Best Random Forest configuration**

| Parameter | Value |
|---|---:|
| n_estimators | 150 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| max_depth | None |
| class_weight | balanced_subsample |

**Best XGBoost configuration**

| Parameter | Value |
|---|---:|
| n_estimators | 350 |
| max_depth | 10 |
| learning_rate | 0.15 |
| subsample | 0.8 |
| colsample_bytree | 1.0 |
| min_child_weight | 1 |

### Step 9: Model Evaluation

The trained models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Weighted F1-score

Macro F1 is the primary evaluation metric because it gives equal importance to each severity class.

## Machine Learning Algorithms

### Logistic Regression

A supervised classification algorithm that estimates the relationship between input features and class probabilities.

It provides a useful baseline for comparing more complex models.

### Random Forest

An ensemble learning algorithm that combines multiple decision trees to produce a classification result.

It can capture nonlinear relationships and interactions between structured features.

### XGBoost

An optimized gradient boosting algorithm that builds decision trees sequentially to improve predictive performance.

It is evaluated because of its effectiveness on structured datasets.

### Support Vector Machine (SVM)

A classification algorithm that uses a decision boundary to separate different classes.

An RBF kernel is used to capture nonlinear decision boundaries.

### K-Nearest Neighbors (KNN)

A distance-based classification algorithm that predicts a class using the labels of nearby training observations.

It provides another baseline for evaluating the effectiveness of the ensemble approaches.

## Experimental Results

### Overall Model Performance

The following table presents the reported experimental results.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 55.31% | 0.4350 | 0.5709 |
| Random Forest | 69.90% | 0.5910 | 0.7009 |
| XGBoost | 70.77% | 0.5946 | 0.7036 |
| SVM | 59.14% | 0.4753 | 0.6073 |
| KNN | 58.39% | 0.4845 | 0.6062 |
| Random Forest (Tuned) | 71.07% | **0.6005** | 0.7092 |
| XGBoost (Tuned) | **71.74%** | 0.5999 | **0.7111** |

### Best Model

**Tuned Random Forest is selected as the champion model.**

Although tuned XGBoost achieves the highest accuracy (71.74%), tuned Random Forest obtains the highest Macro F1-score (0.6005).

Since Macro F1 is the primary metric for this imbalanced multiclass classification problem, Random Forest is selected as the final model.

### Per-Class Performance of Tuned Random Forest

| Severity | Precision | Recall | F1-score |
|---|---:|---:|---:|
| None | 0.8223 | 0.8415 | 0.8318 |
| Low | 0.6673 | 0.6499 | 0.6585 |
| Medium | 0.4657 | 0.4467 | 0.4560 |
| High | 0.4485 | 0.4631 | 0.4557 |

The model performs better on the None and Low classes than on the Medium and High classes.

This demonstrates that minority-class severity prediction remains challenging even after applying SMOTE.

## Visual Results

### 1. Confusion Matrix

The confusion matrix illustrates the relationship between actual severity labels and predicted severity labels.

The reported matrix for tuned Random Forest is:

| Actual \ Predicted | None | Low | Medium | High |
|---|---:|---:|---:|---:|
| None | 14,434 | 2,063 | 836 | 297 |
| Low | 1,954 | 6,563 | 1,316 | 312 |
| Medium | 855 | 1,191 | 2,110 | 747 |
| High | 170 | 89 | 486 | 853 |

The matrix shows that classification errors occur across severity categories, with particular difficulty distinguishing the Medium and High classes.

### 2. Feature Importance

The global feature-importance analysis identifies the following top 20 features reported for the tuned Random Forest model.

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | longitude | 0.0852 |
| 2 | iyear | 0.0823 |
| 3 | latitude | 0.0790 |
| 4 | targsubtype1_txt | 0.0660 |
| 5 | targtype1_txt | 0.0657 |
| 6 | iday | 0.0596 |
| 7 | weapsubtype1_txt | 0.0578 |
| 8 | imonth | 0.0564 |
| 9 | country_txt | 0.0501 |
| 10 | property | 0.0489 |
| 11 | region_txt | 0.0381 |
| 12 | weaptype1_txt | 0.0356 |
| 13 | attacktype1_txt_Assassination | 0.0303 |
| 14 | nperps | 0.0297 |
| 15 | suicide | 0.0206 |
| 16 | specificity | 0.0204 |
| 17 | attacktype1_txt_Facility/Infrastructure Attack | 0.0192 |
| 18 | INT_IDEO | 0.0190 |
| 19 | doubtterr | 0.0185 |
| 20 | attacktype1_txt_Bombing/Explosion | 0.0178 |

Geographical variables such as longitude and latitude, and temporal variables such as year, day, and month, have relatively high importance.

Target-related and weapon-related characteristics also contribute to the model's predictions.

**Important:** Feature importance represents predictive contribution, not a causal relationship.

## Project Workflow

```text
Global Terrorism Database (GTD)
              |
              v
       Data Exploration
              |
              v
        Data Cleaning
              |
              v
   Pre-Incident Feature Selection
              |
              v
     Severity Label Creation
              |
              v
      Feature Engineering
              |
              v
    Stratified Train-Test Split
              |
              v
       SMOTE on Training Set
              |
              v
       Model Training
              |
              v
  Hyperparameter Optimization
              |
              v
       Model Evaluation
              |
              v
    Random Forest Selection
              |
              v
      Feature Importance
```

## Technologies and Libraries

The study uses the following technologies and machine learning libraries:

- **Python** — Main programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Scikit-learn** — Machine learning, preprocessing, model evaluation, and hyperparameter optimization
- **XGBoost** — Gradient boosting classification
- **imbalanced-learn** — SMOTE for class imbalance handling
- **Matplotlib / Seaborn** — Data visualization

## How to Reproduce the Study

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/terrorist-attack-severity-classification.git
```

### 2. Navigate to the Project

```bash
cd terrorist-attack-severity-classification
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### 5. Obtain the Dataset

Download or obtain access to the Global Terrorism Database from:

https://www.start.umd.edu/gtd/

Place the dataset in the location expected by the project's data-loading code.

### 6. Run the Project

Run the main notebook or Python script used in the implementation.

The exact execution command and dataset path should be updated according to the final repository structure.

## Limitations

This research has several limitations:

1. **Historical data:** The GTD version used in this study covers incidents only through 2017, so it may not represent current patterns.
2. **Class imbalance:** The High severity class accounts for only 4.66% of the final dataset.
3. **Limited predictive information:** Excluding post-incident variables is necessary for the research objective, but it also reduces the information available to the models.
4. **Severity definition:** The four classes are based on fatality counts. Alternative definitions may produce different classifications.
5. **Geographical and temporal variation:** Historical relationships may not generalize equally across different regions or future periods.
6. **Interpretability:** Global feature importance helps explain overall model behavior but does not explain every individual prediction or establish causation.

## Responsible Research Note

This project is an academic machine learning study using historical data. It explores statistical associations between incident characteristics and recorded severity categories.

It is **not an operational threat-detection system**, does not identify specific individuals or organizations, and should not be used as the sole basis for real-world security decisions.

The results should be interpreted with caution, particularly because the model performs less reliably on the Medium and High severity classes.

## Research Team

**Department of Computer Science and Engineering**  
BRAC University  
Dhaka, Bangladesh

- Md. Abir Hasan Rohan
- Navid Nawaj
- Mostofa Ahamed Sejan
- Restadul Alam

## Conclusion

This study investigates the classification of terrorist attack severity using pre-incident characteristics from the Global Terrorism Database.

A four-class target was constructed, and five machine learning algorithms were evaluated. Hyperparameter optimization improved the performance of both Random Forest and XGBoost.

The tuned Random Forest achieved the highest Macro F1-score of **0.6005**, while tuned XGBoost achieved the highest accuracy of **71.74%**.

The findings indicate that historical pre-incident characteristics contain useful information for multiclass severity classification. However, predicting minority severity categories remains challenging.

The project provides an experimental framework combining pre-incident feature selection, multiclass classification, class imbalance handling, model comparison, and interpretability.

---

**Project:** Classification of Terrorist Attack Severity Using Pre-Incident Characteristics  
**Institution:** BRAC University  
**Course:** CSE437  
**Research Area:** Machine Learning | Data Science | Multiclass Classification | Explainable ML

