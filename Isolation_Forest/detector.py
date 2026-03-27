class AnomalyDetector:

    def __init__(self, model):
        self.model = model

    def detect(self, df, X):

        predictions = self.model.predict(X)
        scores = self.model.score(X)

        df["anomaly"] = predictions   # -1 = anomaly, 1 = normal
        df["score"] = scores

        return df

    def get_anomalies(self, df):
        return df[df["anomaly"] == -1]