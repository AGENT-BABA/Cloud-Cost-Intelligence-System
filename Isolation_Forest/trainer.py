import pandas as pd
import os
from model import AnomalyModel


class ModelTrainer:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(base_dir, "Data_Collector", "CSV", "final_data.csv")

        self.model = AnomalyModel()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def preprocess(self, df):

        # Select features
        features = [
            "cpu_utilization",
            "network_in",
            "network_out",
            "memory_usage",
            "requests",
            "cost_per_hour"
        ]

        X = df[features]

        return X

    def train(self):
        df = self.load_data()
        X = self.preprocess(df)

        self.model.train(X)

        return self.model, df