class DataInspector:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_info(self):
        """Muestra información general del DataFrame."""
        return self.dataframe.info()

    def describe_data(self):
        """Proporciona estadísticas descriptivas para las columnas numéricas."""
        return self.dataframe.describe()

    def check_null(self):
        """Identifica y cuenta los valores nulos en cada columna."""
        return self.dataframe.isnull().sum()