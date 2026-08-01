# Iris ML Streamlit App

## 📌 Project Overview
This project is part of AIML Session 24 assignment.  
It demonstrates:
- Algorithm comparison on Iris dataset
- Saving the best performing model as `.pkl`
- Building an interactive Streamlit app for prediction

---

## 🔹 Step : Algorithm Comparison
Three models were compared on the Iris dataset:
- Logistic Regression
- KNN
- Naive Bayes

**Result:**  
All models achieved perfect performance (Accuracy, Precision, Recall, F1 Score = 1).  
Hence, all algorithms are equally best for this dataset.

---

## 🔹 Step : Best Model Save
The best model was saved using `joblib`:

```python
import joblib
from sklearn.linear_model import LogisticRegression

best_model = LogisticRegression().fit(X_train, y_train)
joblib.dump(best_model, "best_model.pkl")
