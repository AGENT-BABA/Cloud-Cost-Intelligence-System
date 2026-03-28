"""
verdict2.py
-----------
Stage 2 RAG Pipeline — Business-Aware Final Judgment

Inputs:
  - verdict1_result (dict):  The JSON output from verdict1.py (includes
                              _recent_timestamps injected by run_verdict1).
  - anomaly_result  (dict):  The same anomalous row passed to verdict1.
  - Vector store:            vs_business (business_context_template.txt + actions.txt)

Output:
  A dict with two keys:
    {
      "decision":  "<function_name to execute OR 'NO_ACTION'>",
      "reason":    "<one paragraph explaining the judgment>"
    }

  "decision" is the final function from actions.txt that should be executed
  on AWS (or "NO_ACTION" if the business context says to hold).

Usage (standalone):
    python verdict2.py

Usage (as a module):
    from rag_pipeline.verdict2 import run_verdict2
    result = run_verdict2(verdict1_result, anomaly_result)
"""

import os
import json
import sys
import re

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import get_llm
from verdict1 import run_verdict1, DEFAULT_SMOKE_PATH


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VS_BUSINESS_PATH = os.path.join(BASE_DIR, "vector-store", "vs_business")


# ── Prompt ────────────────────────────────────────────────────────────────────

VERDICT2_PROMPT_TEMPLATE = """
You are a senior cloud operations advisor. Your role is to review an automated
anomaly-remediation recommendation and decide whether it is safe and appropriate
given the business context of this company.

=== TECHNICAL KNOWLEDGE BASE, BUSINESS CONTEXT & ALLOWED ACTIONS ===
{context}

=== FULL SITUATION FOR REVIEW ===
{question}

=== INSTRUCTIONS ===
You will receive:
  1. The last 5 timestamps of AWS metrics showing the trend leading up to the anomaly.
  2. The original anomaly row flagged by the Isolation Forest model.
  3. The Stage-1 technical recommendation (Verdict 1).
  4. The full Technical Knowledge Base specifying the anomaly rules.

You must judge whether the Stage-1 recommendation aligns with the business
context. CRITICAL RULES:
  - The Stage-1 verdict is based on established, documented technical patterns (e.g., storage drains, runaway loops). You MUST treat Stage-1's technical assessment as accurate.
  - If Stage-1 identifies a HIGH risk technical failure (like "Storage drain" or "Runaway invocation loop"), do NOT reject it or say it's unrelated just because of business hours or vague traffic rules. High risk technical failures must be remediated or alerted.
  - Only override Stage-1 if the business context EXPLICITLY lists the resource or situation as a known exception (e.g., "tagged critical", or "specifically expected to spike storage at night").
  - Do not conflate different metrics (e.g., don't dismiss a Storage warning because CPU is normal).
  - Consider: Is this the right time of day to take the *specific action*? Is the risk level acceptable given the business context? Does the 5-timestamp trend support the recommended action?

Return ONLY a valid JSON object — no extra text, no markdown fence, nothing else.

Schema:
{{
  "decision": "<exact function_name from the actions list, e.g. stop_instance — OR the string NO_ACTION>",
  "reason": "<one detailed paragraph explaining why you approve, modify, or reject the Stage-1 recommendation>"
}}

If you approve the Stage-1 action, set "decision" to the same function name.
If you reject it (e.g. wrong time, critical resource, high risk), set "decision"
to "NO_ACTION" and explain why in "reason".
If you think a safer alternative action is warranted, set "decision" to that
alternative function name and explain the substitution in "reason".

Return only the JSON object and nothing else.
"""

VERDICT2_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=VERDICT2_PROMPT_TEMPLATE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_verdict2_query(anomaly: dict, v1: dict, recent_timestamps: list) -> str:
    """
    Build the combined query string for Stage 2:
      - Last 5 timestamps (trend context)
      - Original anomaly row
      - Stage-1 verdict
    """
    lines = [
        f"=== LAST {len(recent_timestamps)} TIMESTAMPS OF AWS METRICS (trend leading up to anomaly) ===",
    ]
    for i, record in enumerate(recent_timestamps, 1):
        lines.append(f"\n  [T-{len(recent_timestamps) - i}] Timestamp: {record.get('timestamp', 'N/A')}")
        for key, value in record.items():
            if key != "timestamp":
                lines.append(f"    {key}: {value}")

    lines.append("")
    lines.append("=== ORIGINAL ANOMALOUS DATA POINT (flagged by Isolation Forest) ===")
    for key, value in anomaly.items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("")
    lines.append("=== STAGE-1 VERDICT (technical recommendation) ===")
    lines.append(f"  Verdict     : {v1.get('verdict', 'N/A')}")
    lines.append(f"  Confidence  : {v1.get('confidence', 'N/A')}")
    lines.append(f"  Action      : {v1.get('action', 'N/A')}")
    lines.append(f"  Parameters  : {json.dumps(v1.get('parameters', {}))}")
    lines.append(f"  Reasoning   : {v1.get('reasoning', 'N/A')}")
    lines.append(f"  Risk Level  : {v1.get('risk_level', 'N/A')}")

    # Read the full technical context and inject it to guarantee the LLM sees it
    general_context_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "context", "general_context.txt")
    if os.path.exists(general_context_path):
        try:
            with open(general_context_path, "r", encoding="utf-8") as f:
                tech_rules = f.read()
            lines.append("")
            lines.append("=== TECHNICAL KNOWLEDGE BASE (Rules governing the anomaly) ===")
            lines.append(tech_rules)
        except Exception as e:
            print(f"[verdict2] Failed to inject general_context.txt: {e}")

    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    """
    Extract and parse the JSON object from the LLM's raw response.
    Falls back to NO_ACTION with an explanation if parsing fails.
    """
    clean = raw.strip()

    # Strip markdown fences
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    # Find first { … } block
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        try:
            parsed = json.loads(clean[start:end])
            if "decision" in parsed and "reason" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    print("[verdict2] WARNING: Could not parse JSON from LLM response. Raw output:")
    print(raw)
    return {
        "decision": "NO_ACTION",
        "reason": (
            "The Stage-2 LLM returned an unparseable response. "
            "Defaulting to NO_ACTION to avoid unintended automated changes. "
            "Please review the anomaly manually."
        ),
    }


# ── Core function ─────────────────────────────────────────────────────────────

def run_verdict2(verdict1_result: dict, anomaly_result: dict) -> dict:
    """
    Run the Stage-2 RAG pipeline.

    Args:
        verdict1_result: Dict returned by run_verdict1(). May contain a
                         '_recent_timestamps' key injected by verdict1 —
                         if present those are reused, otherwise defaults to
                         an empty trend list.
        anomaly_result:  The original anomalous row dict from the Isolation
                         Forest pipeline.

    Returns:
        Dict with keys "decision" (str) and "reason" (str).
    """
    # Reuse the timestamps already loaded by verdict1 (no double file-read)
    recent_timestamps = verdict1_result.pop("_recent_timestamps", [])

    # -- Load vector store --
    if not os.path.exists(VS_BUSINESS_PATH):
        raise FileNotFoundError(
            f"Business vector store not found at {VS_BUSINESS_PATH}. "
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
        VS_BUSINESS_PATH,
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
        chain_type_kwargs={"prompt": VERDICT2_PROMPT},
        return_source_documents=False,
    )

    # -- Build combined query --
    query = _build_verdict2_query(anomaly_result, verdict1_result, recent_timestamps)

    print("[verdict2] Querying LLM …")
    response = chain.invoke({"query": query})
    raw_text: str = response.get("result", "")

    return _parse_json_response(raw_text)


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
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

    print("=" * 60)
    print("STAGE 1 — TECHNICAL VERDICT")
    print("=" * 60)
    v1 = run_verdict1(sample_anomaly, DEFAULT_SMOKE_PATH)
    display_v1 = {k: v for k, v in v1.items() if k != "_recent_timestamps"}
    print("\n[verdict1] Result:")
    print(json.dumps(display_v1, indent=2))

    print("\n" + "=" * 60)
    print("STAGE 2 — BUSINESS-AWARE FINAL JUDGMENT")
    print("=" * 60)
    v2 = run_verdict2(v1, sample_anomaly)
    print("\n[verdict2] Result:")
    print(json.dumps(v2, indent=2))

    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)
    print(f"  Action  : {v2['decision']}")
    print(f"  Reason  : {v2['reason']}")
