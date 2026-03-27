from trainer import ModelTrainer
from detector import AnomalyDetector
import os


def save_results(df):

    base_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(base_dir, "CSV")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "anomaly_output.csv")
    df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":

    trainer = ModelTrainer()

    model, df = trainer.train()

    X = trainer.preprocess(df)

    detector = AnomalyDetector(model)

    result_df = detector.detect(df, X)

    anomalies = detector.get_anomalies(result_df)

    print("\n Anomalies Found:")
    print(anomalies.head())

    save_results(result_df)