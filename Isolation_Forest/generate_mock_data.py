"""
generate_mock_data.py  (lives in: CLOUD COST INTEL/Isolation_Forest/)
Generates realistic CloudWatch + Cost Explorer mock JSON files
and saves them into DataStorage/Raw/
"""
import json, os
import numpy as np
from datetime import datetime, timedelta, timezone

# ── Paths ──────────────────────────────────────────────────────────────────────
# __file__ → Isolation_Forest/generate_mock_data.py
# go up one level → CLOUD COST INTEL/  then into DataStorage/Raw/
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_STORE  = os.path.join(THIS_DIR, "..", "Data_Collector")
OUT_DIR     = os.path.join(DATA_STORE, "Raw")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

# ── Time grid: 7 days at 5-min intervals ──────────────────────────────────────
START = datetime(2026, 3, 21, tzinfo=timezone.utc)
END   = datetime(2026, 3, 28, tzinfo=timezone.utc)
STEP  = timedelta(minutes=5)

timestamps = []
t = START
while t < END:
    timestamps.append(t.isoformat())
    t += STEP

N = len(timestamps)
print(f"Total timestamps: {N}")

# ── Diurnal signal ────────────────────────────────────────────────────────────
hours     = np.array([(START + i * STEP).hour for i in range(N)], dtype=float)
day_phase = np.sin(2 * np.pi * hours / 24)

# ── Anomaly mask: 8 clustered bursts (~11%) ───────────────────────────────────
is_anomaly    = np.zeros(N, dtype=bool)
burst_starts  = [100, 350, 620, 900, 1100, 1350, 1600, 1800]
burst_lengths = [ 30,  25,  40,  20,   35,   30,   25,   20]
for s, l in zip(burst_starts, burst_lengths):
    is_anomaly[s:s+l] = True

anomaly_mult = rng.uniform(2.0, 4.5, size=N)

def normal_signal(base, noise_std, diurnal_amp=0.0):
    sig = base + diurnal_amp * day_phase * base + rng.normal(0, noise_std, N)
    return np.clip(sig, 0.1, None)

def inject_anomaly(signal, mult, cap=None):
    spiked = signal.copy()
    spiked[is_anomaly] *= mult[is_anomaly]
    if cap:
        spiked = np.clip(spiked, None, cap)
    return spiked

# ── Generate each metric ──────────────────────────────────────────────────────
ec2_cpu     = inject_anomaly(normal_signal(18, 6,    diurnal_amp=0.4), anomaly_mult * 1.2, cap=350)
lambda_inv  = inject_anomaly(normal_signal(2000, 600, diurnal_amp=0.5), anomaly_mult * 8,  cap=160000)
lambda_dur  = inject_anomaly(normal_signal(400, 120, diurnal_amp=0.2), anomaly_mult * 1.8, cap=3000)
lambda_err  = np.abs(rng.normal(2, 2, N))
lambda_err[is_anomaly] += rng.uniform(30, 163, size=is_anomaly.sum())
rds_cpu     = inject_anomaly(normal_signal(22, 7,    diurnal_amp=0.3), anomaly_mult * 1.5, cap=350)
rds_storage = 15e9 - np.arange(N) * 1e5 + rng.normal(0, 2e8, N)
rds_storage[is_anomaly] -= rng.uniform(1e9, 3e9, size=is_anomaly.sum())
rds_storage = np.clip(rds_storage, 1e9, 20e9)
ebs_ops     = inject_anomaly(normal_signal(800, 250, diurnal_amp=0.35), anomaly_mult * 2.5, cap=120000)
billing     = inject_anomaly(normal_signal(1.2, 0.3, diurnal_amp=0.1), anomaly_mult * 5,   cap=180)

def make_metric(metric_id, label, values):
    return {
        "Id": metric_id,
        "Label": label,
        "Timestamps": timestamps,
        "Values": [round(float(v), 4) for v in values],
        "StatusCode": "Complete"
    }

cloudwatch_data = {
    "MetricDataResults": [
        make_metric("ec2_cpu",            "EC2 CPUUtilization",   ec2_cpu),
        make_metric("lambda_invocations", "Lambda Invocations",   lambda_inv),
        make_metric("lambda_duration",    "Lambda Duration",      lambda_dur),
        make_metric("lambda_errors",      "Lambda Errors",        lambda_err),
        make_metric("rds_cpu",            "RDS CPUUtilization",   rds_cpu),
        make_metric("rds_storage",        "RDS FreeStorageSpace", rds_storage),
        make_metric("ebs_read_ops",       "EBS VolumeReadOps",    ebs_ops),
        make_metric("billing_charges",    "EstimatedCharges",     billing),
    ],
    "Messages": [],
    "ResponseMetadata": {"RequestId": "mock-001", "HTTPStatusCode": 200}
}

# ── Cost Explorer ─────────────────────────────────────────────────────────────
days       = [(START + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(7)]
base_costs = [2.97, 2.06, 7.83, 1.59, 3.80, 9.57, 1.47]

cost_data = {
    "ResultsByTime": [
        {
            "TimePeriod": {"Start": days[i], "End": days[i]},
            "Total": {
                "BlendedCost": {"Amount": str(base_costs[i]), "Unit": "USD"},
                "UsageQuantity": {"Amount": "24.0", "Unit": "Hrs"}
            },
            "Groups": [],
            "Estimated": False
        }
        for i in range(7)
    ],
    "ResponseMetadata": {"RequestId": "mock-cost-001", "HTTPStatusCode": 200}
}

# ── Ground-truth anomaly labels ───────────────────────────────────────────────
anomaly_labels = {ts: -1 if a else 1 for ts, a in zip(timestamps, is_anomaly)}

# ── Save all to DataStorage/Raw/ ─────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "cloudwatch_mock_data_with_anomalies.json"), "w") as f:
    json.dump(cloudwatch_data, f)

with open(os.path.join(OUT_DIR, "cost_explorer_mock_with_anomalies.json"), "w") as f:
    json.dump(cost_data, f)

with open(os.path.join(OUT_DIR, "anomaly_ground_truth.json"), "w") as f:
    json.dump({"anomaly_labels": anomaly_labels,
               "anomaly_count": int(is_anomaly.sum()),
               "total": N}, f, indent=2)

print(f"Anomalies injected: {is_anomaly.sum()} / {N} = {is_anomaly.mean()*100:.1f}%")
print(f"Files saved to → {os.path.abspath(OUT_DIR)}")