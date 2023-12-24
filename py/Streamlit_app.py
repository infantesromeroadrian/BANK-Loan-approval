import streamlit as st
import pandas as pd
import pickle
import os


# Función para preprocesar los datos de entrada
def preprocess_input(input_data, scaler, encoder):
    # Rellenar los valores NaN
    input_data.fillna({
        'Gender': 'Unknown',
        'Married': 'Unknown',
        'Dependents': 0,
        'Education': 'Unknown',
        'Self_Employed': 'Unknown',
        'ApplicantIncome': 0,
        'CoapplicantIncome': 0,
        'LoanAmount': 0,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,  # Asegurarse de que Credit_History está presente
        'Property_Area': 'Unknown'
    }, inplace=True)

    # Conversión a string para las columnas categóricas
    for column in ['Education', 'Property_Area']:
        input_data[column] = input_data[column].astype(str)

    # Aplicar One-Hot Encoding
    encoded = encoder.transform(input_data[['Education', 'Property_Area']])
    # Usar get_feature_names_out para versiones más recientes de scikit-learn
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Education', 'Property_Area']))
    input_data = pd.concat([input_data, encoded_df], axis=1).drop(['Education', 'Property_Area'], axis=1)

    # Escalado de columnas numéricas
    numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    return input_data



# Carga del scaler, encoder y modelo
scaler = pickle.load(open('/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/BankProjects/HomeLoanApproval/model/scaler.pkl', 'rb'))
encoder = pickle.load(open('/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/BankProjects/HomeLoanApproval/model/encoder.pkl', 'rb'))
model = pickle.load(open('/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/BankProjects/HomeLoanApproval/model/ensemble_model.pkl', 'rb'))

# Streamlit UI
st.title("Home Loan Approval Prediction")

# Recolección de entradas del usuario
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.number_input("Dependents", min_value=0, max_value=10, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Botón de predicción
if st.button('Predict Loan Approval'):
    # Preparar datos para la predicción
    input_data = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Property_Area': property_area
    }])

    processed_data = preprocess_input(input_data, scaler, encoder)

    # Realizar predicción
    prediction = model.predict(processed_data)[0]

    # Mostrar resultado
    if prediction == 1:
        st.success('Loan is likely to be approved.')
    else:
        st.error('Loan is unlikely to be approved.')