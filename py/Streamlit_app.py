import streamlit as st
import pandas as pd
import pickle
import os


# Función para preprocesar los datos de entrada
def preprocess_input(input_data, scaler, encoder):
    # Asignar valores a categorías conocidas
    input_data['Gender'] = input_data['Gender'].map({'Male': 1, 'Female': 0, 'Other': -1}).fillna(-1)
    input_data['Married'] = input_data['Married'].map({'Yes': 1, 'No': 0}).fillna(-1)
    input_data['Dependents'] = input_data['Dependents'].replace({'3+': 3}).astype(int).fillna(0)
    input_data['Self_Employed'] = input_data['Self_Employed'].map({'Yes': 1, 'No': 0}).fillna(-1)

    # Conservar 'Education' y 'Property_Area' como texto para One-Hot Encoding
    input_data['Education'] = input_data['Education'].fillna('Unknown')
    input_data['Property_Area'] = input_data['Property_Area'].fillna('Unknown')

    # Escalar columnas numéricas
    numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Aplicar One-Hot Encoding para 'Education' y 'Property_Area'
    categorical_cols = ['Education', 'Property_Area']
    encoded = encoder.transform(input_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    input_data = pd.concat([input_data, encoded_df], axis=1).drop(categorical_cols, axis=1)

    # Asegurar presencia y orden de columnas necesarias
    expected_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'ApplicantIncome',
                     'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'] + list(encoded_df.columns)
    input_data = input_data.reindex(columns=expected_cols, fill_value=0)

    return input_data


# Carga del scaler, encoder y modelo
scaler = pickle.load(open('/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/BankProjects/HomeLoanApproval/model/scaler.pkl', 'rb'))
encoder = pickle.load(open('/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/BankProjects/HomeLoanApproval/model/encoder.pkl', 'rb'))
model = pickle.load(open('/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/BankProjects/HomeLoanApproval/model/ensemble_model.pkl', 'rb'))


# Streamlit UI setup
st.title("Home Loan Approval Prediction")

# Define UI elements to capture input data
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.number_input("Dependents", 0, 3, 0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", 0, 100000, 0)
coapplicant_income = st.number_input("Coapplicant Income", 0, 50000, 0)
loan_amount = st.number_input("Loan Amount", 0, 1000, 0)
loan_amount_term = st.number_input("Loan Amount Term", 0, 360, 0)
credit_history = st.number_input("Credit History", 0, 1, 1)
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# When the user clicks the 'Predict' button
if st.button('Predict Loan Approval'):
    # Prepare the input data for prediction
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
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    # Preprocess the input data
    processed_data = preprocess_input(input_data, scaler, encoder)

    # Ensure the order of columns matches the training dataset
    correct_feature_order = ['Gender', 'Married', 'Dependents', 'Education_Graduate', 'Education_Not Graduate',
                             'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                             'Credit_History', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']

    # Reorder 'processed_data' to match the expected feature order
    processed_data = processed_data[correct_feature_order]

    # Ensure no additional columns
    processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]

    # Make a prediction
    try:
        prediction = model.predict(processed_data)[0]
        # Display the result
        if prediction == 1:
            st.success('Loan is likely to be approved.')
        else:
            st.error('Loan is unlikely to be approved.')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")