import json
import os

class CloudWatchClient:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_path = os.path.join(base_dir, "JSON_dummy", "cloudwatch_mock_data.json")

    def get_cpu_utilization(self, instance_id, start_time, end_time):
        with open(self.file_path, "r") as f:
            data = json.load(f)
        return data