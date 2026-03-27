from pipeline import DataPipeline
import csv
import os

def save_to_csv(data):

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 👉 Target: Data_Collector/CSV/
    csv_dir = os.path.join(base_dir, "CSV")

    # Create folder if not exists
    os.makedirs(csv_dir, exist_ok=True)

    file_path = os.path.join(csv_dir, "final_data.csv")

    keys = data[0].keys()

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data saved to {file_path}")


if __name__ == "__main__":

    pipeline = DataPipeline()

    instance_id = "i-123"
    start_time = "2026-03-25"
    end_time = "2026-03-28"

    final_data = pipeline.run(instance_id, start_time, end_time)

    # Print sample
    for row in final_data[:5]:
        print(row)

    # Save to CSV
    save_to_csv(final_data)