"""
verdict1.py
-----------
Stage 1 RAG Pipeline — Technical Anomaly Verdict

Inputs:
  - anomaly_result (dict):  Output from the Isolation Forest pipeline describing
                            the detected anomaly (passed in programmatically).
  - Vector store:           vs_technical (general_context.txt + actions.txt)

Output:
  A JSON-structured dict (also returned as a Python dict) with keys:
    verdict, confidence, action, parameters, reasoning, risk_level

Usage (standalone):
    python verdict1.py

Usage (as a module):
    from rag_pipeline.verdict1 import run_verdict1
    result = run_verdict1(anomaly_result)
"""

import os
import json
import sys

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Add parent directory to path if running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import get_llm


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VS_TECHNICAL_PATH = os.path.join(BASE_DIR, "vector-store", "vs_technical")
EMBED_MODEL = "nomic-embed-text"

# ── Prompt ────────────────────────────────────────────────────────────────────

VERDICT1_PROMPT_TEMPLATE = """
You are a cloud cost intelligence agent. Your task is to analyse an AWS anomaly
and decide which remediation action to take from the allowed action list.

=== CONTEXT FROM KNOWLEDGE BASE ===
{context}

=== ANOMALY DETECTED BY THE ISOLATION FOREST PIPELINE ===
{question}

=== INSTRUCTIONS ===
1. Read the anomaly details above carefully.
2. Use the context to understand what type of anomaly this is.
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


# ── Core function ─────────────────────────────────────────────────────────────

def run_verdict1(anomaly_result: dict) -> dict:
    """
    Run the Stage-1 RAG pipeline.

    Args:
        anomaly_result: Dictionary produced by the Isolation Forest pipeline.
                        Example structure:
                        {
                          "resource_type": "EC2",
                          "resource_id": "i-0a1b2c3d4e5f67890",
                          "cpu_utilization": 1.2,
                          "network_in": 500,
                          "network_out": 300,
                          "memory_usage": 5.0,
                          "requests": 0,
                          "cost_per_hour": 0.096,
                          "anomaly": -1,
                          "score": -0.15
                        }

    Returns:
        Parsed dict matching the actions.txt RETURN FORMAT schema.
    """
    # -- Load vector store --
    if not os.path.exists(VS_TECHNICAL_PATH):
        raise FileNotFoundError(
            f"Technical vector store not found at {VS_TECHNICAL_PATH}. "
            "Please run ingestion.py first."
        )

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://localhost:11434",
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

    # -- Format anomaly as a descriptive query --
    query = _format_anomaly_query(anomaly_result)

    print("[verdict1] Querying LLM …")
    response = chain.invoke({"query": query})

    raw_text: str = response.get("result", "")

    # -- Parse JSON from LLM output --
    verdict_dict = _parse_json_response(raw_text)

    return verdict_dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_anomaly_query(anomaly: dict) -> str:
    """Convert the anomaly dict into a human-readable description for the LLM."""
    lines = ["An anomaly has been detected with the following metrics:"]
    for key, value in anomaly.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    """
    Try to extract and parse a JSON object from the LLM's raw response.
    Falls back to a safe send_alert action if parsing fails.
    """
    # Strip any accidental markdown fences
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    # Find the first { … } block
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(clean[start:end])
        except json.JSONDecodeError:
            pass

    # Fallback
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


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example anomaly result (as would come from the Isolation Forest pipeline)
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

    result = run_verdict1(sample_anomaly)
    print("\n[verdict1] Result:")
    print(json.dumps(result, indent=2))
