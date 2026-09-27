# Business Entity Resolution

A machine learning and entity-matching pipeline for identifying whether business records from multiple data sources refer to the same real-world business.

## 📌 Problem Statement

Business information is often collected from different sources, resulting in duplicate or inconsistent records.

For example:

```text
Source 1:
Starbucks Coffee, Tirupati

Source 2:
Starbucks Coffee Shop, Tirupathi

Source 3:
Starbucks, Tirupati
```

Although the records are written differently, they may represent the same business.

This project performs **Business Entity Resolution** by comparing business records using name, address, country, and other engineered features.

---

## 🚀 Features

* Data loading and preprocessing
* Text normalization
* Blocking to reduce unnecessary comparisons
* Candidate pair generation
* Feature engineering
* Business name similarity
* Address similarity
* Token-based similarity
* TF-IDF similarity
* Country matching
* Missing-value indicators
* Pairwise entity matching
* Train/validation evaluation
* Test-set prediction
* Official submission validation

---

## 🏗️ Project Structure

```text
Businness Entity Resolution/
│
├── code/
│   └── business_entity_resolution/
│       ├── README.md
│       ├── requirements.txt
│       │
│       └── src/
│           ├── __init__.py
│           ├── blocking.py
│           ├── data_loader.py
│           ├── evaluate.py
│           ├── features.py
│           ├── main.py
│           ├── predict.py
│           ├── preprocessing.py
│           ├── train.py
│           └── utils.py
│
├── dataset/
│   ├── train/
│   └── test/
│
├── utils/
│   └── validate_submission.py
│
├── ml_challenge.pdf
├── Documentation_template.md
├── README.md
└── .gitignore
```

---

## 🔄 Pipeline

The project follows this workflow:

```text
Raw Business Data
        ↓
Data Loading
        ↓
Preprocessing
        ↓
Blocking
        ↓
Candidate Pair Generation
        ↓
Feature Engineering
        ↓
Pair Matching Model
        ↓
Entity Resolution
        ↓
Evaluation
        ↓
Test Prediction
        ↓
Submission Validation
```

---

## 🧠 Feature Engineering

The matcher uses multiple features to determine whether two business records represent the same entity.

### Business Name Features

* Exact name match
* Fuzzy similarity
* Token similarity
* TF-IDF similarity
* Name length
* First-token comparison

### Address Features

* Exact address comparison
* Fuzzy address similarity
* Token-based similarity
* TF-IDF similarity
* Address length

### Other Features

* Country match
* Country mismatch
* Missing-value indicators
* Combined similarity features

These features are combined to make the final matching decision.

---

## ⚡ Blocking

Comparing every record with every other record can be computationally expensive.

Therefore, the project uses **blocking** to generate a smaller set of candidate pairs before applying the matching model.

For the validation data:

```text
Brute-force candidate pairs : 76,560
Blocked candidate pairs     : 5,100
Reduction ratio             : ~93.34%
```

This significantly reduces the number of comparisons while maintaining high candidate recall.

---

## 📊 Validation Results

The validation pipeline produced:

| Metric          | Result |
| --------------- | -----: |
| Precision       |   1.00 |
| Recall          |   1.00 |
| F0.5 Score      |   1.00 |
| Blocking Recall |   1.00 |

The blocking stage achieved approximately **93.34% candidate reduction** compared with brute-force comparison.

---

## 🧪 Test Prediction

The test pipeline generated:

```text
Predicted links      : 117
Predicted singletons : 33
Candidate pairs      : 1261
```

The generated submission was also checked using the provided validation script.

```text
Official Validator: PASS
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Text similarity techniques
* TF-IDF
* Fuzzy matching
* Machine Learning
* Entity Resolution

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Nikhitha-Appecherla/Businness-Entity-Resolution.git
cd Businness-Entity-Resolution
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 5. Run the pipeline

```bash
python code/business_entity_resolution/src/main.py
```

---

## 📁 Output

The pipeline generates matching and candidate-pair results in the output directory.

Typical outputs include:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

---

## 🎯 Objective

The main objective of this project is to build an efficient and reliable system capable of identifying duplicate business entities across multiple heterogeneous data sources.

The combination of **blocking, preprocessing, similarity features, and machine learning** helps reduce computational cost while maintaining accurate entity matching.

---


GitHub: [Nikhitha-Appecherla](https://github.com/Nikhitha-Appecherla)
