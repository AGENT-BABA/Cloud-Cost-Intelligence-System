import pandas as pd
import numpy as np


# Maps CloudWatch metric ID → DataFrame column name
METRIC_FIELD_MAP = {
    "ec2_cpu":         "cpu_utilization",
    "ebs_read_ops":    "requests",
    "rds_cpu":         "memory_usage",
    "lambda_errors":   "error_rate",
    "rds_storage":     "storage_free",
    "billing_charges": "billing_rate",
}


class DataProcessor:

    def transform_metric(self, metric_response, metric_id):
        target = None
        for m in metric_response["MetricDataResults"]:
            if m["Id"] == metric_id:
                target = m
                break
        if target is None:
            raise ValueError(f"Metric ID '{metric_id}' not found.")

        field_name = METRIC_FIELD_MAP.get(metric_id, metric_id)
        rows = [{"timestamp": ts, field_name: float(val)}
                for ts, val in zip(target["Timestamps"], target["Values"])]
        rows.sort(key=lambda x: x["timestamp"])
        return rows

    def merge_metrics(self, metric_response):
        """
        Merge ALL CloudWatch metrics into one aligned DataFrame.

        Feature mapping:
          ec2_cpu            → cpu_utilization   (direct %)
          rds_cpu            → memory_usage      (proxy %, clamped 0-100)
          ebs_read_ops       → requests          (I/O ops proxy)
          lambda_invocations → network_in  MB    (scaled *0.006)
          lambda_duration    → network_out MB    (scaled *0.380)
          lambda_errors      → error_rate        (count per interval)
          rds_storage        → storage_free GB   (bytes → GB)
          billing_charges    → billing_rate      (estimated $ charge)
        """
        dfs = []

        # Direct metrics
        for metric_id in ["ec2_cpu", "ebs_read_ops", "rds_cpu",
                          "lambda_errors", "billing_charges"]:
            try:
                rows = self.transform_metric(metric_response, metric_id)
                df   = pd.DataFrame(rows).set_index("timestamp")
                dfs.append(df)
            except ValueError:
                pass

        # rds_storage → storage_free_gb
        try:
            rows = self.transform_metric(metric_response, "rds_storage")
            df   = pd.DataFrame(rows).set_index("timestamp")
            df["storage_free"] = (df["storage_free"] / 1e9).round(2)
            dfs.append(df)
        except ValueError:
            pass

        # Network proxies from Lambda metrics
        for metric_id, col_name, scale, floor in [
            ("lambda_invocations", "network_in",  0.006, 10.0),
            ("lambda_duration",    "network_out", 0.380, 10.0),
        ]:
            try:
                rows  = self.transform_metric(metric_response, metric_id)
                field = METRIC_FIELD_MAP.get(metric_id, metric_id)
                df    = pd.DataFrame(rows)
                if field in df.columns:
                    df = df.rename(columns={field: col_name})
                else:
                    df.columns = ["timestamp", col_name]
                df = df.set_index("timestamp")
                df[col_name] = (df[col_name] * scale).clip(lower=floor).round(2)
                dfs.append(df)
            except ValueError:
                pass

        # Outer merge → forward-fill gaps
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.join(df, how="outer")

        merged = merged.sort_index().ffill().bfill()

        if "memory_usage" in merged.columns:
            merged["memory_usage"] = merged["memory_usage"].clip(upper=100.0).round(2)

        merged = merged.reset_index()
        merged.columns.name = None

        col_order = ["timestamp", "cpu_utilization", "network_in", "network_out",
                     "memory_usage", "requests", "error_rate",
                     "storage_free", "billing_rate"]
        merged = merged[[c for c in col_order if c in merged.columns]]
        return merged.to_dict(orient="records")

    def attach_cost(self, data, cost_response):
        """Attach per-hour cost from Cost Explorer per-day totals."""
        daily_cost = {}
        for day in cost_response["ResultsByTime"]:
            date_str  = day["TimePeriod"]["Start"]
            total_usd = float(day["Total"]["BlendedCost"]["Amount"])
            daily_cost[date_str] = round(total_usd / 24, 6)

        for row in data:
            date_key         = row["timestamp"][:10]
            row["cost_per_hour"] = daily_cost.get(date_key, 0.0)
        return data