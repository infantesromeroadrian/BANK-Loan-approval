import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc

# Clase para visualización de datos
class DataVisualizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def plot_histogram(self, column, title):
        return px.histogram(self.dataframe, x=column, title=title)

    def plot_bar(self, column, title):
        return px.bar(self.dataframe, x=column, title=title)

    def plot_box(self, x_column, y_column, title):
        return px.box(self.dataframe, x=x_column, y=y_column, title=title)

    def plot_heatmap(self, title, color_scale='RdBu'):
        numeric_df = self.dataframe.select_dtypes(include=np.number)
        corr_matrix = numeric_df.corr()
        return px.imshow(corr_matrix, title=title, color_continuous_scale=color_scale)

    def plot_scatter(self, x_column, y_column, title):
        return px.scatter(self.dataframe, x=x_column, y=y_column, title=title)

    def plot_distribution(self, column, title):
        return self.plot_histogram(column, title)

# Cargar los datos
data_loader = pd.read_csv("/Users/adrianinfantes/Desktop/AIR/COLLEGE AND STUDIES/Data_Scientist_formation/BankProjects/HomeLoanApproval/data/loan_sanction_train.csv")
visualizer = DataVisualizer(data_loader)

# Crear figuras para la aplicación Dash
fig_histogram = visualizer.plot_histogram('ApplicantIncome', 'Distribución de Ingresos del Solicitante')
fig_bar = visualizer.plot_bar('Gender', 'Distribución por Género')
fig_box = visualizer.plot_box('Education', 'ApplicantIncome', 'Ingresos por Nivel Educativo')
fig_heatmap = visualizer.plot_heatmap('Mapa de Calor de Correlaciones')
fig_scatter = visualizer.plot_scatter('ApplicantIncome', 'LoanAmount', 'Relación entre Ingresos y Monto del Préstamo')

# Inicializar la aplicación Dash
app = Dash(__name__)

# Definir la disposición de la aplicación
app.layout = html.Div(children=[
    html.H1(children='Dashboard de Análisis de Datos'),

    dcc.Graph(
        id='graph-1',
        figure=fig_histogram
    ),

    dcc.Graph(
        id='graph-2',
        figure=fig_bar
    ),

    dcc.Graph(
        id='graph-3',
        figure=fig_box
    ),

    dcc.Graph(
        id='graph-4',
        figure=fig_heatmap
    ),

    dcc.Graph(
        id='graph-5',
        figure=fig_scatter
    )
])

# Correr la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
