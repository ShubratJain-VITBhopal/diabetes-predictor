import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# loading dataset
data = pd.read_csv("diabetes_data.csv")
 
x = data[["pregnancies", "glucose", "blood_pressure", "bmi", "age"]]
y = data["diabetes"]
 
# splitting the data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
 
# using random forest method
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)
 
# checking that how accurate the model is..
pred = clf.predict(x_test)
acc = accuracy_score(y_test, pred)
print("Model Accuracy:", round(acc * 100, 2), "%")
 
print()
print("         Diabetes Risk Predictor          ")
print("------------------------------------------")
print()
 
try:
    preg  = float(input("How many pregnancies? (0 if not applicable): "))
    gluc  = float(input("Glucose level (mg/dL): "))
    bp = float(input("Blood pressure (mm Hg): "))
    bmi  = float(input("BMI value: "))
    age  = float(input("Age: "))
 
    # putting input in a dataframe
    user = pd.DataFrame([[preg, gluc, bp, bmi, age]],
                        columns=["pregnancies", "glucose", "blood_pressure", "bmi", "age"])
 
    result = clf.predict(user)[0]
    prob   = clf.predict_proba(user)[0]
 
    print()
    print("------------------------------------------")
    if result == 1:
        print("  Result : HIGH RISK")
        print("  Model confidence :", round(prob[1] * 100, 1), "%")
        print()
        print("  Please visit a doctor for proper checkup.")
    else:
        print("  Result : LOW RISK")
        print("  Model confidence :", round(prob[0] * 100, 1), "%")
        print()
        print("  Keep up the healthy habits!")
    print("------------------------------------------")
    print()
 
except ValueError:
    print("Please enter numbers only.")