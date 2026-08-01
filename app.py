# Step 1: Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_california_housing, load_iris






# Step 2: App Title
st.title(" ML Models Frontend")
st.write("Choose a model and dataset to train & evaluate")







# Step 3: Dropdown for Model Selection
model_choice = st.selectbox(
    "Select Model",
    [
        "Linear Regression",
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "Algorithm Comparison"
    ]
)







# Step 4: Linear Regression 
if model_choice == "Linear Regression":
    st.header("Linear Regression on California Housing Dataset ")

    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    # Load dataset
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Add polynomial features (degree=2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_scaled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show results
    st.write("R2 Score (Polynomial Features):", r2_score(y_test, y_pred))
    st.write("Sample Predictions:", y_pred[:10])






# Question 5: Algorithm Comparison
st.header("Algorithm Comparison on Iris Dataset")

from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

# Collect results
comparison = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    comparison.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1 Score": f1_score(y_test, y_pred, average="macro")
    })

# Show comparison table
st.write(pd.DataFrame(comparison))











# Step 6: KNN
if model_choice == "KNN":
    st.header("K-Nearest Neighbors (KNN) on Iris Dataset")

    # Load Iris dataset
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Try different k values
    results = {}
    for k in [3, 5, 7]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[k] = accuracy_score(y_test, y_pred)

    # Show results in table format
    results_df = pd.DataFrame(list(results.items()), columns=["k Value", "Accuracy"])
    st.write(results_df)

    # Show best k
    best_k = max(results, key=results.get)
    st.write("Best k:", best_k, "with Accuracy:", results[best_k])








# Step 7: Naive Bayes
if model_choice == "Naive Bayes":
    st.header("Naive Bayes on Iris Dataset")

    # Load Iris dataset
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show results
    st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, average="macro"))
    st.write("Recall:", recall_score(y_test, y_pred, average="macro"))
    st.write("F1 Score:", f1_score(y_test, y_pred, average="macro"))




import joblib
from sklearn.linear_model import LogisticRegression

# Train and Save Best Model
best_model = LogisticRegression().fit(X_train, y_train)
joblib.dump(best_model, "best_model.pkl")






import streamlit as st
import joblib

# Load Saved Model
model = joblib.load("best_model.pkl")

# Class mapping (numeric → names)
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.title("Iris Flower Prediction App")

# User Input Form
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(user_input)
    st.write("Predicted Species:", species_map[prediction[0]])
