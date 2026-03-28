"""
verdict1.py
-----------
Stage 1 RAG Pipeline — Technical Anomaly Verdict

Inputs:
  - anomaly_result (dict):  The anomalous row detected by the Isolation Forest
                            pipeline (single timestamp dict with all metrics).
  - smoke_data_path (str):  Path to smoke_latest.json — the last 5 timestamps
                            of AWS metrics are extracted and included as context.
  - Vector store:           vs_technical (general_context.txt + actions.txt)

Output:
  A JSON-structured dict with keys:
    verdict, confidence, action, parameters, reasoning, risk_level

Usage (standalone):
    python verdict1.py

Usage (as a module):
    from rag_pipeline.verdict1 import run_verdict1
    result = run_verdict1(anomaly_result, smoke_data_path)
"""

import os
import json
import sys

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# Add parent directory to path if running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import get_llm


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VS_TECHNICAL_PATH = os.path.join(BASE_DIR, "vector-store", "vs_technical")

# Default path to smoke_latest.json (relative to this file's location)
DEFAULT_SMOKE_PATH = os.path.join(
    BASE_DIR, "..", "Data_Collector", "Processed", "smoke_latest.json"
)


# ── Prompt ────────────────────────────────────────────────────────────────────

VERDICT1_PROMPT_TEMPLATE = """
You are a cloud cost intelligence agent. Your task is to analyse an AWS anomaly
and decide which remediation action to take from the allowed action list.

=== CONTEXT FROM KNOWLEDGE BASE ===
{context}

=== ANOMALY DETECTED BY THE ISOLATION FOREST PIPELINE ===
{question}

=== INSTRUCTIONS ===
1. Read the anomaly details and the last 5 timestamps of AWS metrics above carefully.
2. Use the context to understand what type of anomaly this is and the trend over time.
3. Select the single most appropriate action from the actions list in the context.
4. Return ONLY a valid JSON object — no extra text, no markdown, no explanation
   outside the JSON.

The JSON must follow this exact schema:
{{
  "verdict": "<one sentence — what the anomaly is>",
  "confidence": <float 0.0-1.0>,
  "action": "<function_name from actions list>",
  "parameters": {{ "<param>": "<value>" }},
  "reasoning": "<one sentence — why this action was chosen>",
  "risk_level": "<LOW | MEDIUM | HIGH>"
}}

If no automatic action is appropriate, set "action" to "send_alert" with a
meaningful message in parameters.

Return only the JSON object and nothing else.
"""

VERDICT1_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=VERDICT1_PROMPT_TEMPLATE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_last_n_timestamps(smoke_path: str, n: int = 5) -> list:
    """
    Load the last N timestamp records from smoke_latest.json.

    Args:
        smoke_path: Absolute or relative path to smoke_latest.json.
        n:          Number of most-recent records to return.

    Returns:
        List of dicts, each representing one timestamp of AWS metrics.
    """
    smoke_path = os.path.abspath(smoke_path)
    if not os.path.exists(smoke_path):
        raise FileNotFoundError(f"smoke_latest.json not found at: {smoke_path}")

    with open(smoke_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data is a list sorted oldest → newest; take the last N
    return data[-n:]


def _format_verdict1_query(anomaly: dict, recent_timestamps: list) -> str:
    """
    Build the LLM query string combining:
      - A KEY SIGNALS summary derived from the anomaly (guides the FAISS retriever)
      - The anomalous metric row from Isolation Forest
      - The last 5 timestamps of AWS data for trend context
    """
    # --- Derive the most anomalous signals to help FAISS find the right pattern ---
    signals = []
    cpu = anomaly.get("cpu_utilization", 0)
    mem = anomaly.get("memory_usage", 0)
    err = anomaly.get("error_rate", 0)
    storage = anomaly.get("storage_free", 100)
    net_in = anomaly.get("network_in", 0)
    requests = anomaly.get("requests", 0)

    if cpu > 85:
        signals.append(f"CPU utilization critically high at {cpu}% — possible CPU runaway or infinite loop")
    elif cpu < 3:
        signals.append(f"CPU utilization extremely low at {cpu}% — possible idle instance")

    if mem > 80:
        signals.append(f"Memory usage critically high at {mem}% — possible memory leak or runaway process")

    if err > 20:
        signals.append(f"Error rate critically high at {err}% — possible error storm or bad deployment")
    elif err > 10:
        signals.append(f"Error rate elevated at {err}%")

    if storage < 10:
        signals.append(f"Free storage critically low at {storage} GB — possible storage drain")

    if net_in > 400:
        signals.append(f"Network-in extremely high at {net_in} MB — possible traffic spike or data exfiltration")

    if not signals:
        signals.append("Unusual combination of metrics — reviewing full data for pattern matching")

    lines = [
        "=== KEY ANOMALY SIGNALS (use these to identify the anomaly type) ===",
    ]
    for s in signals:
        lines.append(f"  - {s}")

    lines.append("")
    lines.append("=== ANOMALOUS DATA POINT (flagged by Isolation Forest) ===")
    for key, value in anomaly.items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append(f"=== LAST {len(recent_timestamps)} TIMESTAMPS OF AWS METRICS (trend context) ===")
    for i, record in enumerate(recent_timestamps, 1):
        lines.append(f"\n  [T-{len(recent_timestamps) - i}] Timestamp: {record.get('timestamp', 'N/A')}")
        for key, value in record.items():
            if key != "timestamp":
                lines.append(f"    {key}: {value}")

    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    """
    Try to extract and parse a JSON object from the LLM's raw response.
    Falls back to a safe send_alert action if parsing fails.
    """
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(clean[start:end])
        except json.JSONDecodeError:
            pass

    print("[verdict1] WARNING: Could not parse JSON from LLM response. Raw output:")
    print(raw)
    return {
        "verdict": "LLM returned unparseable output — sending alert for manual review.",
        "confidence": 0.0,
        "action": "send_alert",
        "parameters": {
            "message": "Anomaly detected but automated verdict could not be determined.",
            "severity": "warning",
            "resource_id": "unknown",
        },
        "reasoning": "JSON parsing failed on LLM output.",
        "risk_level": "MEDIUM",
    }


# ── Core function ─────────────────────────────────────────────────────────────

def run_verdict1(anomaly_result: dict, smoke_data_path: str = DEFAULT_SMOKE_PATH) -> dict:
    """
    Run the Stage-1 RAG pipeline.

    Args:
        anomaly_result:   Dict of the anomalous row produced by the Isolation
                          Forest pipeline. Must contain all metric fields plus
                          'timestamp', 'anomaly', and 'score'.
        smoke_data_path:  Path to smoke_latest.json. Defaults to the standard
                          location relative to this file.

    Returns:
        Parsed dict matching the actions.txt RETURN FORMAT schema.
    """
    # -- Load vector store --
    if not os.path.exists(VS_TECHNICAL_PATH):
        raise FileNotFoundError(
            f"Technical vector store not found at {VS_TECHNICAL_PATH}. "
            "Please run ingestion.py first."
        )

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set.")
        
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    vectorstore = FAISS.load_local(
        VS_TECHNICAL_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # -- Build retriever --
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # -- Build RAG chain --
    llm = get_llm()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": VERDICT1_PROMPT},
        return_source_documents=False,
    )

    # -- Load last 5 timestamps from smoke_latest.json --
    print(f"[verdict1] Loading last 5 timestamps from: {smoke_data_path}")
    recent_timestamps = load_last_n_timestamps(smoke_data_path, n=5)
    print(f"[verdict1] Loaded {len(recent_timestamps)} recent timestamp records.")

    # -- Format combined query --
    query = _format_verdict1_query(anomaly_result, recent_timestamps)

    print("[verdict1] Querying LLM …")
    response = chain.invoke({"query": query})
    raw_text: str = response.get("result", "")

    verdict_dict = _parse_json_response(raw_text)

    # Attach the recent timestamps to the result so verdict2 can reuse them
    verdict_dict["_recent_timestamps"] = recent_timestamps

    return verdict_dict


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example anomaly result (as would come from the Isolation Forest pipeline)
    sample_anomaly = {
        "timestamp": "2026-03-27T23:55:00+00:00",
        "cpu_utilization": 1.2,
        "network_in": 450,
        "network_out": 280,
        "memory_usage": 4.8,
        "requests": 0,
        "error_rate": 15.5,
        "storage_free": 5.0,
        "billing_rate": 3.2,
        "cost_per_hour": 0.096,
        "anomaly": -1,
        "score": -0.18,
    }

    result = run_verdict1(sample_anomaly)
    print("\n[verdict1] Result:")
    # Don't print the embedded timestamps in the summary output
    display = {k: v for k, v in result.items() if k != "_recent_timestamps"}
    print(json.dumps(display, indent=2))
