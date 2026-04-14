import streamlit as st 
import pandas as pd
import numpy as np
import joblib

scaler=joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Insurance claim predictor", layout="centered")
st.title("Health insurance payment prediction App")
st.write("Enter the details below to estimate your insurance amount!!")

with st.form("input_form"):
    col1,col2=st.columns(2)
    with col1:
        age=st.number_input("Age",min_value=0,max_value=100,value=30)
        bmi=st.number_input("BMI",min_value=10.0, max_value=60.0,value=25.0)
        children=st.number_input("No of children", min_value=0, max_value=10, value=4)
    with col2:
        bloodpressure=st.number_input("Blood Pressure", min_value=60, max_value=160, value=120)
        gender_text = st.selectbox("Gender", ["female", "male"])
        diabetic_text = st.selectbox("Diabetic", ["no", "yes"])
        smoker_text = st.selectbox("Smoker", ["no", "yes"])
        

    submitted=st.form_submit_button("Predict amount")

if submitted:
    gender = 0 if gender_text == "female" else 1
    diabetic = 0 if diabetic_text == "no" else 1
    smoker = 0 if smoker_text == "no" else 1

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetic": [diabetic],
        "children": [children],
        "smoker": [smoker]
    })

    num_cols = ["age", "bmi", "bloodpressure", "children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    prediction = model.predict(input_data)[0]
    st.success(f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}")
