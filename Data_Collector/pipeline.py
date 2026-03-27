from Apis.cloudwatch_client import CloudWatchClient
from Apis.cost_explorer_client import CostExplorerClient
from processor import DataProcessor


class DataPipeline:

    def __init__(self):
        self.cloudwatch = CloudWatchClient()
        self.cost = CostExplorerClient()
        self.processor = DataProcessor()

    def run(self, instance_id, start_time, end_time):

        # Step 1: Fetch CPU data
        cpu_data = self.cloudwatch.get_cpu_utilization(instance_id, start_time, end_time)

        # Step 2: Transform
        cpu_rows = self.processor.transform_metric(cpu_data, "cpu_utilization")

        # Step 3: Enrich (add missing metrics)
        enriched = self.processor.enrich(cpu_rows)

        # Step 4: Get cost
        cost_data = self.cost.get_cost_data(instance_id, start_time, end_time)

        # Step 5: Attach cost
        final_data = self.processor.attach_cost(enriched, cost_data)

        return final_data