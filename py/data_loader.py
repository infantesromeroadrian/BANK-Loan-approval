import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        print("Datos cargados con éxito.")

    def show_head(self, n=5):
        if self.data is not None:
            return self.data.head(n)
        else:
            print("Datos no cargados. Por favor, primero ejecute la función load_data.")