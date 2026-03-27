import json
import os

class CostExplorerClient:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_path = os.path.join(base_dir, "JSON_dummy", "cost_explorer_mock.json")

    def get_cost_data(self, instance_id, start_date, end_date):
        with open(self.file_path, "r") as f:
            data = json.load(f)
        return data