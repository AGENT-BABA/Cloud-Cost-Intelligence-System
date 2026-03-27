import random

class DataProcessor:

    def transform_metric(self, metric_response, field_name):
        results = metric_response["MetricDataResults"][0]

        rows = []

        for ts, val in zip(results["Timestamps"], results["Values"]):
            rows.append({
                "timestamp": ts,
                field_name: val
            })

        # Sort ascending (VERY IMPORTANT)
        rows.sort(key=lambda x: x["timestamp"])

        return rows

    def enrich(self, data):
        for row in data:
            row["network_in"] = round(random.uniform(100, 1000), 2)
            row["network_out"] = round(random.uniform(100, 1000), 2)
            row["memory_usage"] = round(random.uniform(30, 80), 2)
            row["requests"] = random.randint(50, 500)
        return data

    def attach_cost(self, data, cost_response):

        results = cost_response["ResultsByTime"]

        total_cost = 0

        for day in results:
            amount = float(day["Total"]["BlendedCost"]["Amount"])
            total_cost += amount

        avg_cost_per_hour = total_cost / (24 * len(results))

        for row in data:
            row["cost_per_hour"] = round(avg_cost_per_hour, 4)

        return data