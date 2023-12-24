# Importaciones necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Importar tus clases personalizadas aquí
from data_loader import DataLoader
from data_inspector import DataInspector
from data_preprocessor import DataPreprocessor
from data_visualizer import DataVisualizer
from data_splitter import DataSplitter
from model_tester import ModelTester


def main():
    # Cargar y explorar datos
    data_loader = DataLoader("/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/BankProjects/HomeLoanApproval/data/loan_sanction_train.csv")
    data_loader.load_data()
    print(data_loader.show_head())

    data_inspector = DataInspector(data_loader.data)
    print("Información General del DataFrame:")
    print(data_inspector.get_info())
    print("\nEstadísticas Descriptivas:")
    print(data_inspector.describe_data())
    print("\nValores Nulos en el DataFrame:")
    print(data_inspector.check_null())

    # Preprocesamiento de datos
    data_preprocessor = DataPreprocessor(data_loader.data)
    data_preprocessor.handle_missing_values()
    data_preprocessor.encode_binary_columns()
    data_preprocessor.encode_dependents()
    data_preprocessor.apply_one_hot_encoding()
    data_preprocessor.scale_numeric_columns()
    print(data_preprocessor.dataframe.head())

    # Visualización de datos
    data_visualizer = DataVisualizer(data_preprocessor.dataframe)
    columns_to_visualize = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Status']
    for col in columns_to_visualize:
        data_visualizer.plot_distribution(col)

    # División de los datos
    data_splitter = DataSplitter(data_preprocessor.dataframe)
    X_train, X_test, y_train, y_test = data_splitter.split_data("Loan_Status")

    # Entrenamiento y evaluación de modelos
    model_tester = ModelTester(X_train, X_test, y_train, y_test)
    ensemble_model = model_tester.ensemble_models()

    # Guardar el modelo
    with open('ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    print("Modelo guardado con éxito.")

    # Cargar y preprocesar el conjunto de datos de prueba
    test_data_loader = DataLoader("/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/BankProjects/HomeLoanApproval/data/loan_sanction_test.csv")
    test_data_loader.load_data()
    test_data_preprocessor = DataPreprocessor(test_data_loader.data)
    test_data_preprocessor.handle_missing_values()
    test_data_preprocessor.encode_binary_columns()
    test_data_preprocessor.encode_dependents()
    test_data_preprocessor.apply_one_hot_encoding()
    test_data_preprocessor.scale_numeric_columns()

    # Cargar el modelo y hacer predicciones
    with open('ensemble_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    predictions = loaded_model.predict(test_data_preprocessor.dataframe.drop('Loan_ID', axis=1))
    print("Predicciones realizadas con éxito.")
    print(predictions)


if __name__ == "__main__":
    main()
