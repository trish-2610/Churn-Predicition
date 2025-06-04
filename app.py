## Importing libraries 
import streamlit as st 
import numpy as np 
import pandas as pd
import tensorflow 
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
import pickle 

## Loading out trained ANN model 
model = tensorflow.keras.models.load_model("model.h5")

## Loading encoder and scaler 
with open("label_encoder_gender.pkl","rb") as file :
    label_encoder_gender = pickle.load(file)

with open("ohe_encoder_geography.pkl","rb") as file :
    ohe_encoder_geography = pickle.load(file)

with open("scaler.pkl","rb") as file :
    scaler = pickle.load(file)

## streamlit web app title 
st.title(":red[Customer Churn Prediction]")
st.write(":grey[using ANN]")

## Input Fields (user input)
geography = st.selectbox("Geography",options=ohe_encoder_geography.categories_[0])
gender = st.selectbox("Gender",options=label_encoder_gender.classes_)
age = st.slider("Age",min_value=18,max_value=60)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",min_value=0,max_value=10)
num_of_products = st.slider("Number of Products",min_value=1,max_value=5)
has_cr_card = st.selectbox("Has credit card",options=[0,1])
is_active_member = st.selectbox("Is Active member",options=[0,1])

## Input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## Encoding "geography" using OHE
geo_encoder = ohe_encoder_geography.transform([[geography]])
geo_encoder_df = pd.DataFrame(geo_encoder.toarray(),columns=ohe_encoder_geography.get_feature_names_out())

## Merging it with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)

## Scaling the data 
input_data_scaled = scaler.transform(input_data)

## Prediction
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

if prediction:
   st.write("Prediction Probability = ",prediction_probability)
   if prediction_probability > 0.5 :
    st.info("The customer is likely to churn (Exit the bank)")
   else :
    st.info("The customer is unlikely to churn ")
