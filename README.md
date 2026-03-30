# 🩺 Diabetes Risk Predictor

A beginner Machine Learning project that predicts whether a person is at **high or low risk of diabetes** based on key health indicators, using the well-known Pima Indians Diabetes Dataset.

> ⚠️ This is an educational ML project — not a medical diagnosis tool.

---

## 📌 Problem Statement

Diabetes is a widespread health condition that often goes undetected until it causes serious complications. Early identification of at-risk individuals can enable timely lifestyle changes and medical intervention. This project uses a machine learning model to estimate diabetes risk based on simple health metrics that most people know about themselves.

---

## 🤖 How It Works

The program uses a **Random Forest Classifier** trained on a structured health dataset. The user enters 5 health values, and the model predicts whether they are at high or low risk — along with a confidence percentage.

**Input features used:**
- Number of pregnancies
- Glucose level (mg/dL)
- Blood pressure (mm Hg)
- BMI (Body Mass Index)
- Age

---

## ⚙️ Setup Instructions

### 1. Make sure Python is installed
Download from https://www.python.org if you don't have it.

### 2. Install required libraries
```bash
pip install pandas scikit-learn
```

### 3. Clone or download this repository
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-risk-predictor.git
cd diabetes-risk-predictor
```

### 4. Run the program
```bash
python diabetes_predictor.py
```

---

## 💻 Example Usage

```
✅ Model trained successfully! Accuracy: 83.3%

        🩺 Diabetes Risk Predictor            

Answer the following questions about yourself.

Number of pregnancies (enter 0 if male or none): 2
Glucose level in mg/dL (e.g. 120): 160
Blood pressure in mm Hg (e.g. 70): 80
BMI - Body Mass Index (e.g. 28.5): 34.5
Age in years (e.g. 35): 45

  Prediction : HIGH RISK of Diabetes
  Confidence   : 78.0%

  ⚕️  Please consult a doctor for proper testing.
```

---

## 📊 Dataset

The dataset (`diabetes_data.csv`) is based on the **Pima Indians Diabetes Dataset**, a widely used benchmark dataset in ML research. It contains patient health records with the following columns:

| Column | Description |
|---|---|
| `pregnancies` | Number of times pregnant |
| `glucose` | Plasma glucose concentration (mg/dL) |
| `blood_pressure` | Diastolic blood pressure (mm Hg) |
| `bmi` | Body Mass Index |
| `age` | Age in years |
| `diabetes` | 1 = Diabetic, 0 = Not Diabetic |

---

## 🛠️ Libraries Used

- `pandas` — loading and processing the dataset
- `scikit-learn` — Random Forest model, train/test split, accuracy evaluation

---

## 🔍 Algorithm

**Random Forest Classifier** — an ensemble method that builds multiple decision trees and combines their outputs for a more accurate and stable prediction. It is well-suited for health datasets with mixed numeric features.

---

