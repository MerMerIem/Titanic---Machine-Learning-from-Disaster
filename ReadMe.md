# 🚢 **Titanic — Machine Learning from Disaster**

## 📝 **Project Overview**

This project tackles the classic [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/overview): predicting whether a passenger survived the 1912 sinking based on attributes like class, sex, age, and fare. The project follows a disciplined, notebook-by-notebook data science workflow — every cleaning and feature-engineering decision is backed by an explicit hypothesis, checked against the data, and documented before being applied.

The workflow covers:

- **Problem definition & success criteria** 🎯
- **Data loading, inspection & train/validation split** 🔍
- **Exploratory data analysis (EDA)** 📊
- **Preprocessing & feature engineering** 🛠️
- **Baseline modeling** 🤖

---

## 🎯 **Problem Definition**

- **Objective:** Predict whether a passenger survived (`Survived`: 1) or not (0) 🚢
- **Evaluation metric:** **F1-score** (β = 1) — chosen because both false positives (false hope) and false negatives (missed practical/legal decisions) carry real cost, so precision and recall are weighted equally ⚖️
- **Success criteria:**
  - Beat the trivial majority-class baseline (F1 = 0) as a sanity check ✅
  - Beat/match the simple logistic-regression baseline (F1 = `0.7786`) — the real bar 🏁
- **Constraint:** This is a Kaggle exercise — only predictions on the provided `test.csv` are submitted for scoring 📤

---

## ⚡ **Key Steps & Findings**

### 🔍 **1. Load, Inspect & Split**

- Loaded `train.csv` (891 rows × 12 columns), checked dtypes, missing values, and duplicates 🧐
- Missing data: **Age** ~19.9%, **Cabin** ~77.1%, **Embarked** ~0.2% — no duplicate rows 🕳️
- Split into `X_train`/`X_val` (80/20) with `stratify=y` and `random_state=42` to preserve the ~62%/38% died/survived class balance in both sets 🔀

### 📊 **2. Exploratory Data Analysis**

- **Numeric features:** Age and SibSp look similar across survivors/non-survivors (weak signal alone); Fare shows a clear split (higher fare → higher survival); Parch shows a mild survival advantage for small families 👨‍👩‍👧
- **Fare outliers investigated:** two passengers sharing ticket `PC 17755` both show Fare = 512.33 — evidence that Fare sometimes reflects a *group* total rather than a per-person price, motivating a later `Fare_per_person` feature 🎫
- **Categorical features:** females survived far more often (74.3% vs 18.5% for males); Pclass 1 survived most (64.9%) down to Pclass 3 (24.3%); Cherbourg (`Embarked=C`) passengers had the highest survival rate 🚻
- **Confound check:** Cherbourg's higher survival rate is explained by it having a much higher share of first-class passengers (53.2% Pclass 1) than Queenstown (1.8%) or Southampton (18.2%) — Embarked acts as a proxy for Pclass, not an independent cause 🧭
- **Cabin vs. Pclass:** missingness in Cabin tracks class almost perfectly (Pclass 1: 20.5% missing, Pclass 2: 90.0%, Pclass 3: 97.7%) → **decision: drop `Cabin`**, since it's too sparse to impute reliably and largely redundant with Pclass 🗑️
- **Correlations:** strongest links to `Survived` are Pclass (−0.35) and Fare (+0.28); Fare and Pclass are themselves correlated (−0.56), a moderate multicollinearity noted for later modeling decisions 🔗
- **Target leakage check:** all features are recorded at booking time, before the disaster — no leakage risk ✅
- **Cardinality check:** `Ticket` (571 unique) and `Name` (712 unique, ≈ one per row) are too high-cardinality to use raw → motivated extracting `Title` from `Name` and engineering `Group_Size`/`Fare_per_person` from `Ticket` 🪪

### 🛠️ **3. Preprocessing & Feature Engineering**

- **Missing values:** dropped `Cabin`; filled `Embarked` with the train-set mode (`S`); filled `Age` with the train-set median (28.5) — all statistics computed on `X_train` only to avoid leakage into validation 🧮
- **`Title` extraction:** parsed the honorific from `Name` (e.g. `Mr`, `Mrs`, `Miss`, `Master`), merged `Mlle`/`Ms` into `Miss`, and grouped rare titles (`Dr`, `Rev`, `Col`, `Major`, `Lady`, `Sir`, `Jonkheer`, `Don`) into `Other` 🏷️
- **`Group_Size` & `Fare_per_person`:** counted passengers sharing the same `Ticket` (computed across train+val, since ticket sharing is a fixed historical fact rather than something derived from the target) and divided `Fare` by `Group_Size` to correct for group bookings 🎟️
- **Encoding:**
  - One-hot encoded `Embarked` (`pd.get_dummies`) and `Title` (`sklearn.OneHotEncoder`, fit on train only, to safely handle unseen categories in validation) 🔢
  - Binary-mapped `Sex` (`male`→0, `female`→1) since it's strictly two categories ⚧
  - Left `Pclass` as a raw ordinal number, since survival-rate gaps between classes (1→2, 2→3) were roughly equal (~0.20 each) 🔢
- **Dropped** `Name`, `Ticket`, and `PassengerId` after extracting their useful signal 🗑️
- **Scaling:** `StandardScaler` (fit on train only) applied to continuous/count features (`Age`, `Fare`, `Fare_per_person`, `SibSp`, `Parch`, `Group_Size`); binary/one-hot columns and `Pclass` left unscaled ⚖️
- Saved processed splits to `data/processed/` 💾

### 🤖 **4. Baseline Modeling**

- **Trivial baseline:** `DummyClassifier` (majority-class) → **F1 = 0.0000**, confirming the sanity floor 🧱
- **Real baseline:** plain `LogisticRegression` (default params) → **F1 = 0.7786** on the validation set, with 0.85/0.90 precision/recall for "died" and 0.82/0.74 for "survived" 📈
- This logistic regression score sets the bar (`0.7786`) that subsequent, more advanced models are expected to beat 🎯

---

## 🛠️ **Tools and Technologies**

- **Python** — core language 🐍
- **pandas** — data loading, inspection, cleaning, feature engineering 🧮
- **scikit-learn** — `train_test_split`, `OneHotEncoder`, `StandardScaler`, `DummyClassifier`, `LogisticRegression`, F1/classification-report metrics 🤖
- **seaborn / matplotlib** — boxplots, correlation heatmaps, distribution visualizations 📊
- **Jupyter notebooks** — step-by-step, documented workflow 📓

---

## 🗂️ **Project Structure**

```
Titanic---Machine-Learning-from-Disaster-main/
├── data/
│   ├── raw/          # Kaggle train.csv & test.csv
│   ├── interim/       # Post-split X/y train & val (pre-preprocessing)
│   └── processed/     # Fully encoded & scaled X/y train & val
├── notebooks/
│   ├── 01_load_inspect_split.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_preprocessing_&_feature_engineering.ipynb
│   └── 04_baseline_model.ipynb
└── reports/
    └── figures/        # correlation_matrix.png, output.png
```

---

## 🏗️ **Project Workflow**

### **1. Load, Inspect & Split** (Notebook 01)

- Load raw data, audit quality, stratified 80/20 train/val split 🔍

### **2. Exploratory Data Analysis** (Notebook 02)

- Distribution checks, survival-rate breakdowns, confound analysis, correlation & leakage checks, cardinality review 📊

### **3. Preprocessing & Feature Engineering** (Notebook 03)

- Missing-value handling, `Title`/`Group_Size`/`Fare_per_person` engineering, encoding, scaling 🛠️

### **4. Baseline Modeling** (Notebook 04)

- Dummy and Logistic Regression baselines to set the performance floor and bar 🤖

---

## 🎓 **Learning Outcomes**

This project helped practice:

- [ ] Defining a problem with a justified evaluation metric and success criteria 🎯
- [ ] Careful, hypothesis-driven exploratory data analysis 🔍
- [ ] Distinguishing correlation from confounding (e.g. Embarked vs. Pclass) 🧭
- [ ] Leak-free preprocessing (fitting encoders/scalers/imputers on train only) 🛡️
- [ ] Domain-informed feature engineering (Title, Group_Size, Fare_per_person) 🛠️
- [ ] Establishing trivial and simple-model baselines before building complex models 🤖

---

## 🔮 **Next Steps**

- Try tree-based and ensemble models (Random Forest, Gradient Boosting/XGBoost) against the logistic regression baseline 🌲
- Hyperparameter tuning with cross-validation 🔧
- Generate predictions on Kaggle's `test.csv` for submission 📤
