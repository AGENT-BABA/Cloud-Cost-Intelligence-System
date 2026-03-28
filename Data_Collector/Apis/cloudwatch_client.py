import json
import os


class CloudWatchClient:

    def __init__(self):
        # __file__ → DataStorage/Apis/cloudwatch_client.py
        # go up one level → DataStorage/  then into Raw/
        base_dir       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_path = os.path.join(base_dir, "Raw",
                                      "cloudwatch_mock_data_with_anomalies.json")

    def get_metrics(self, instance_id, start_time, end_time):
        with open(self.file_path, "r") as f:
            return json.load(f)

    # Alias so existing callers don't break
    def get_cpu_utilization(self, instance_id, start_time, end_time):
        return self.get_metrics(instance_id, start_time, end_time)