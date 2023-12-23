import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def handle_missing_values(self):
        """Manejo de valores nulos en el DataFrame."""
        # Para columnas numéricas
        numeric_cols = self.dataframe.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            self.dataframe[col].fillna(self.dataframe[col].median(), inplace=True)

        # Para columnas categóricas
        categorical_cols = self.dataframe.select_dtypes(include='object').columns
        for col in categorical_cols:
            self.dataframe[col].fillna(self.dataframe[col].mode()[0], inplace=True)

    def encode_binary_columns(self):
        """Codifica columnas binarias según las reglas especificadas."""
        mappings = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 'Y': 1, 'N': 0}
        columns = ['Gender', 'Married', 'Self_Employed']
        if 'Loan_Status' in self.dataframe.columns:
            columns.append('Loan_Status')
        for col in columns:
            self.dataframe[col] = self.dataframe[col].map(mappings)

    def encode_dependents(self):
        """Convierte la columna 'Dependents' a valores numéricos."""
        self.dataframe['Dependents'] = self.dataframe['Dependents'].replace({'3+': 3}).astype(int)

    def apply_one_hot_encoding(self):
        """Aplica One-Hot Encoding a las columnas seleccionadas."""
        encoder = OneHotEncoder(sparse=False)
        columns = ['Education', 'Property_Area']
        for col in columns:
            encoded = encoder.fit_transform(self.dataframe[[col]])
            # Crear nombres de columnas para las nuevas características
            cols = [f"{col}_{category}" for category in encoder.categories_[0]]
            # Añadir las nuevas columnas al DataFrame
            self.dataframe[cols] = encoded
            # Eliminar la columna original
            self.dataframe.drop(col, axis=1, inplace=True)

    def scale_numeric_columns(self):
        """Escala las columnas numéricas especificadas."""
        scaler = StandardScaler()
        columns_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        self.dataframe[columns_to_scale] = scaler.fit_transform(self.dataframe[columns_to_scale])

    def encode_dependents(self):
        """Convierte la columna 'Dependents' a valores numéricos."""
        # Primero reemplazar '3+' con 3
        self.dataframe['Dependents'] = self.dataframe['Dependents'].replace({'3+': 3}).astype(int)

        # Luego convertir cualquier valor distinto de 0 en 1
        self.dataframe['Dependents'] = self.dataframe['Dependents'].apply(lambda x: 1 if x != 0 else 0)