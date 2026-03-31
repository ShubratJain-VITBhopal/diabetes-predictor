# Diabetes Risk Predictor

A Machine Learning project that predicts whether a person is at high or low risk of diabetes based on key health indicators, using the Pima Indians Diabetes Dataset.

---

## Problem Statement

Diabetes often goes undetected until it causes serious complications. This tool takes basic health inputs that most people already know and gives them an early risk indication using a trained ML model.

---

## How It Works

The program uses a Random Forest Classifier trained on a structured health dataset. The user enters 5 values and the model predicts High Risk or Low Risk along with a confidence percentage.

---

## Requirements

- Python 3.7 or above
- pandas
- scikit-learn

---

## Setup and Installation

### Step 1 — Install Python

Download and install Python from https://www.python.org/downloads/
Make sure to check "Add Python to PATH" during installation.

### Step 2 — Install dependencies

Open your terminal or command prompt and run:

```bash
pip install pandas scikit-learn
```

### Step 3 — Clone the repository

```bash
git clone https://github.com/ShubratJain-VITBhopal/diabetes-risk-predictor.git
cd diabetes-risk-predictor
```

Or download the ZIP from GitHub and extract it.

### Step 4 — Run the program

Make sure you are inside the project folder, then run:

```bash
python diabetes_predictor.py
```

---

## Usage

When you run the program it will ask you 5 questions:

```
How many pregnancies? (0 if not applicable): 2
Glucose level (mg/dL): 160
Blood pressure (mm Hg): 80
BMI value: 34.5
Age: 45
```

Example output:

```
Model Accuracy: 75.0 %

------------------------------------------
         Diabetes Risk Predictor
------------------------------------------
Note: This is only for educational use.

------------------------------------------
  Result : HIGH RISK
  Model confidence : 78.0 %

  Please visit a doctor for proper checkup.
------------------------------------------

Disclaimer - this is a college project, not real medical advice.
```

---

## Dataset

The file `diabetes_data.csv` contains 60 patient records based on the Pima Indians Diabetes Dataset.

| Column         | Description                       |
| -------------- | --------------------------------- |
| pregnancies    | Number of times pregnant          |
| glucose        | Plasma glucose level in mg/dL     |
| blood_pressure | Diastolic blood pressure in mm Hg |
| bmi            | Body Mass Index                   |
| age            | Age in years                      |
| diabetes       | 1 = Diabetic, 0 = Not Diabetic    |

---

## Libraries Used

- pandas — for loading and processing the CSV dataset
- scikit-learn — for the Random Forest model, train/test split, and accuracy evaluation

---

## Algorithm

Random Forest Classifier — builds multiple decision trees and combines their results for a more stable and accurate prediction. Chosen over Logistic Regression because it gave higher accuracy on this dataset.

---
