import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def plot_distribution(self, column):
        """Genera un gráfico de barras para la columna especificada."""
        if column not in self.dataframe.columns:
            print(f"La columna {column} no existe en el DataFrame.")
            return

        # Conteo de valores
        value_counts = self.dataframe[column].value_counts()

        plt.figure(figsize=(8, 4))
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Distribución de {column}')
        plt.ylabel('Cantidad')
        plt.xlabel(column)
        plt.show()