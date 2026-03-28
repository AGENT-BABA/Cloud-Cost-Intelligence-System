"""
verdict2.py
-----------
Stage 2 RAG Pipeline — Business-Aware Judgment

Inputs:
  - verdict1_result (dict):  The JSON output from verdict1.py
  - anomaly_result  (dict):  The same anomaly dict passed to verdict1
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

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import get_llm
from verdict1 import run_verdict1


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VS_BUSINESS_PATH = os.path.join(BASE_DIR, "vector-store", "vs_business")
EMBED_MODEL = "nomic-embed-text"


# ── Prompt ────────────────────────────────────────────────────────────────────

VERDICT2_PROMPT_TEMPLATE = """
You are a senior cloud operations advisor. Your role is to review an automated
anomaly-remediation recommendation and decide whether it is safe and appropriate
given the business context of this company.

=== BUSINESS CONTEXT & ALLOWED ACTIONS (from knowledge base) ===
{context}

=== ORIGINAL ANOMALY ===
{question}

=== INSTRUCTIONS ===
You will receive:
  1. The original anomaly metrics detected by the Isolation Forest model.
  2. The Stage-1 technical recommendation (Verdict 1).

You must judge whether the Stage-1 recommendation aligns with the business
context. Consider:
  - Is this the right time of day / week to take this action?
  - Is the affected resource tagged as critical or owned by a sensitive team?
  - Does the proposed action match the company's auto-action preference?
  - Is the risk level acceptable given the business context?

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


# ── Core function ─────────────────────────────────────────────────────────────

def run_verdict2(verdict1_result: dict, anomaly_result: dict) -> dict:
    """
    Run the Stage-2 RAG pipeline.

    Args:
        verdict1_result: Dict returned by run_verdict1().
        anomaly_result:  The original anomaly dict from the Isolation Forest pipeline.

    Returns:
        Dict with keys "decision" (str) and "reason" (str).
    """
    # -- Load vector store --
    if not os.path.exists(VS_BUSINESS_PATH):
        raise FileNotFoundError(
            f"Business vector store not found at {VS_BUSINESS_PATH}. "
            "Please run ingestion.py first."
        )

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://localhost:11434",
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

    # -- Build combined query for the LLM --
    query = _build_verdict2_query(anomaly_result, verdict1_result)

    print("[verdict2] Querying LLM …")
    response = chain.invoke({"query": query})

    raw_text: str = response.get("result", "")

    # -- Parse JSON --
    verdict2_dict = _parse_json_response(raw_text)

    return verdict2_dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_verdict2_query(anomaly: dict, v1: dict) -> str:
    """
    Construct a combined query string that includes the original anomaly
    details and the Stage-1 verdict for the Stage-2 LLM to reason over.
    """
    lines = [
        "=== ORIGINAL ANOMALY METRICS ===",
    ]
    for key, value in anomaly.items():
        lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("=== STAGE-1 VERDICT (technical recommendation) ===")
    lines.append(f"  Verdict     : {v1.get('verdict', 'N/A')}")
    lines.append(f"  Confidence  : {v1.get('confidence', 'N/A')}")
    lines.append(f"  Action      : {v1.get('action', 'N/A')}")
    lines.append(f"  Parameters  : {json.dumps(v1.get('parameters', {}))}")
    lines.append(f"  Reasoning   : {v1.get('reasoning', 'N/A')}")
    lines.append(f"  Risk Level  : {v1.get('risk_level', 'N/A')}")

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
            # Validate required keys
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


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example anomaly (same as in verdict1.py standalone test)
    sample_anomaly = {
        "resource_type": "EC2",
        "resource_id": "i-0a1b2c3d4e5f67890",
        "cpu_utilization": 1.2,
        "network_in": 450,
        "network_out": 280,
        "memory_usage": 4.8,
        "requests": 0,
        "cost_per_hour": 0.096,
        "anomaly": -1,
        "score": -0.18,
    }

    # Step 1: Get technical verdict
    print("=== Running Verdict 1 ===")
    v1 = run_verdict1(sample_anomaly)
    print("\n[verdict1] Result:")
    print(json.dumps(v1, indent=2))

    # Step 2: Get business-aware judgment
    print("\n=== Running Verdict 2 ===")
    v2 = run_verdict2(v1, sample_anomaly)
    print("\n[verdict2] Result:")
    print(json.dumps(v2, indent=2))

    print("\n=== Final Decision ===")
    print(f"Action to execute : {v2['decision']}")
    print(f"Reason            : {v2['reason']}")
