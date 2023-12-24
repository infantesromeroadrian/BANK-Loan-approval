import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

class DataPreprocessor:
    def __init__(self, dataframe, scaler=None, encoder=None):
        self.dataframe = dataframe
        self.scaler = scaler if scaler else StandardScaler()
        self.encoder = encoder if encoder else OneHotEncoder(sparse=False)
        self.columns_to_encode = ['Education', 'Property_Area']
        self.columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

    def handle_missing_values(self):
        numeric_cols = self.dataframe.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            self.dataframe[col].fillna(self.dataframe[col].median(), inplace=True)

        categorical_cols = self.dataframe.select_dtypes(include='object').columns
        for col in categorical_cols:
            self.dataframe[col].fillna(self.dataframe[col].mode()[0], inplace=True)

    def encode_binary_columns(self):
        mappings = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}
        columns = ['Gender', 'Married', 'Self_Employed']
        if 'Loan_Status' in self.dataframe.columns:
            columns.append('Loan_Status')
        for col in columns:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].map(mappings)

    def encode_dependents(self):
        if 'Dependents' in self.dataframe.columns:
            self.dataframe['Dependents'] = self.dataframe['Dependents'].replace({'3+': 3}).astype(int)

    def apply_one_hot_encoding(self):
        encoded = self.encoder.fit_transform(self.dataframe[self.columns_to_encode])
        cols = [f"{col}_{category}" for col in self.columns_to_encode for category in self.encoder.categories_[self.columns_to_encode.index(col)]]
        self.dataframe[cols] = encoded
        self.dataframe.drop(self.columns_to_encode, axis=1, inplace=True)

    def scale_numeric_columns(self):
        self.dataframe[self.columns_to_scale] = self.scaler.fit_transform(self.dataframe[self.columns_to_scale])

    def preprocess(self, new_data):
        new_data = new_data.copy()
        self.handle_missing_values()
        self.encode_binary_columns()
        self.encode_dependents()
        self.apply_one_hot_encoding()
        self.scale_numeric_columns()
        return new_data

    def save_transformers(self, scaler_path, encoder_path):
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.encoder, f)