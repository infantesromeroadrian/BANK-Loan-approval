from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """Divide los datos en conjuntos de entrenamiento y prueba, excluyendo la columna 'ID'."""
        X = self.dataframe.drop([target_column, 'Loan_ID'], axis=1)  # Excluir 'ID' junto con la columna objetivo
        y = self.dataframe[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)