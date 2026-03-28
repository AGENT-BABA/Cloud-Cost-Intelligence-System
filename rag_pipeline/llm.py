"""
llm.py
------
Configures the LLM used by verdict1.py and verdict2.py.

Automatically switches between:
  - LOCAL:      Ollama (phi3 at localhost:11434) — for local development
  - PRODUCTION: OpenAI (gpt-4o-mini) — for Render / cloud deployment

The switch is controlled by the environment variable LLM_PROVIDER:
  LLM_PROVIDER=ollama   → use local Ollama (default if not set)
  LLM_PROVIDER=openai   → use OpenAI API (requires OPENAI_API_KEY env var)

On Render, set these two environment variables in the dashboard:
  LLM_PROVIDER = openai
  OPENAI_API_KEY = sk-...
"""

import os


def get_llm(temperature: float = 0.1):
    """
    Returns the correct LLM instance based on the LLM_PROVIDER env variable.
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        return _get_openai_llm(temperature)
    else:
        return _get_ollama_llm(temperature)


def _get_ollama_llm(temperature: float):
    """Local development — Ollama running at localhost."""
    from langchain_ollama import OllamaLLM

    model = os.environ.get("OLLAMA_MODEL", "phi3")
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    print(f"[llm] Using Ollama: model={model}, base_url={base_url}")
    return OllamaLLM(model=model, temperature=temperature, base_url=base_url)


def _get_openai_llm(temperature: float):
    """Production — OpenAI API."""
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Add it as an environment variable on Render."
        )

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    print(f"[llm] Using OpenAI: model={model}")
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
