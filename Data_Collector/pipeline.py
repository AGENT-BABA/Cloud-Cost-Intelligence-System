from Apis.cloudwatch_client import CloudWatchClient
from Apis.cost_explorer_client import CostExplorerClient
from processor import DataProcessor


class DataPipeline:

    def __init__(self):
        self.cloudwatch = CloudWatchClient()
        self.cost       = CostExplorerClient()
        self.processor  = DataProcessor()

    def run(self, instance_id, start_time, end_time):
        # Step 1: Fetch all CloudWatch metrics
        cw_data = self.cloudwatch.get_cpu_utilization(instance_id, start_time, end_time)

        # Step 2: Merge all metrics into one aligned DataFrame
        print("Merging CloudWatch metrics...")
        merged = self.processor.merge_metrics(cw_data)
        print(f"  Merged {len(merged)} rows")

        # Step 3: Fetch cost data
        cost_data = self.cost.get_cost_data(instance_id, start_time, end_time)

        # Step 4: Attach real per-day cost to each row
        final_data = self.processor.attach_cost(merged, cost_data)

        return final_data