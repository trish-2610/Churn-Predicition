# Churn Prediction Model using ANN

A deep learning project to predict customer churn using **Artificial Neural Networks (ANN)**, implemented with **TensorFlow** and **Keras**, and deployed as a web app using **Streamlit**.

---

## 🧠 Libraries Used

1. **TensorFlow**  
   - Open-source library for end-to-end machine learning projects.  
   - Used to implement ANN, RNN, LSTM, Transformers, etc.

2. **Keras**  
   - High-level API integrated with TensorFlow.  
   - Simplifies model building by providing prewritten APIs and less boilerplate code.

---

## 📊 Problem Statement

- **Dataset**: `Churn_Modelling.csv`
- **Task**: Binary classification (0 / 1)
- **Objective**:  
  Predict whether a customer will leave the bank (`Exited = 1`) or stay (`Exited = 0`) based on their profile data like:
  - Gender
  - Age
  - Tenure
  - Balance
  - etc.

---

## 🛠 Project Pipeline

```mermaid
graph LR
A[Churn Dataset] --> B[Basic Feature Engineering]
B --> C[Convert Categorical to Numerical + Standardization]
C --> D[Create ANN using Keras & TensorFlow]
D --> E[Save Model as Pickle/h5]
E --> F[Streamlit Web App]
F --> G[Deploy on Streamlit Cloud]
