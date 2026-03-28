from pipeline import DataPipeline
import csv
import json
import os
import sqlite3
from datetime import datetime


def save_to_csv(data, filename="final_data_pipeline.csv"):
    # __file__ → DataStorage/main.py
    # Save output into DataStorage/Processed/
    base_dir      = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "Processed")
    os.makedirs(processed_dir, exist_ok=True)
    file_path = os.path.join(processed_dir, filename)

    if not data:
        print("No data to save.")
        return

    keys = list(data[0].keys())
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved → {file_path}  ({len(data)} rows)")
    return file_path


def save_smoke_data(data, n=5, filename_csv="smoke_latest.csv", filename_json="smoke_latest.json"):
    if not data:
        print("No data for smoke output.")
        return None

    smoke = data[-n:]
    # Save as CSV
    save_to_csv(smoke, filename=filename_csv)

    # Save as JSON
    base_dir      = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "Processed")
    os.makedirs(processed_dir, exist_ok=True)
    json_path = os.path.join(processed_dir, filename_json)
    with open(json_path, "w") as f:
        json.dump(smoke, f, indent=5, default=str)

    print(f"Smoke latest {n} rows saved → {json_path}")
    return smoke


def save_to_analysis_db(data, db_name="analysis.db", table_name="anomaly_history"):
    if not data:
        print("No data to store in AnalysisDB.")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, db_name)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Use first row to determine columns
    columns = list(data[0].keys())
    col_defs = ", ".join([f"{c} REAL" if c != "timestamp" else f"{c} TEXT" for c in columns])
    col_defs += ", ingested_at TEXT"

    create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})"
    cursor.execute(create_stmt)

    placeholders = ",".join(["?" for _ in columns] + ["?"])
    insert_stmt = f"INSERT INTO {table_name} ({', '.join(columns)}, ingested_at) VALUES ({placeholders})"

    now = datetime.utcnow().isoformat()
    for row in data:
        vals = [row[col] for col in columns] + [now]
        cursor.execute(insert_stmt, vals)

    conn.commit()
    conn.close()

    print(f"Inserted {len(data)} rows into AnalysisDB ({db_path}) table: {table_name}")
    return db_path


if __name__ == "__main__":
    pipeline   = DataPipeline()
    final_data = pipeline.run("i-123", "2026-03-21", "2026-03-27")

    print(f"\nSample row: {final_data[0]}")
    print(f"Columns: {list(final_data[0].keys())}")

    costs = set(row["cost_per_hour"] for row in final_data)
    print(f"Unique cost_per_hour values: {sorted(costs)}")
    print("(Should have 7 different values — one per day)")

    save_to_csv(final_data)
    save_smoke_data(final_data, n=5)
    save_to_analysis_db(final_data)
