"""
llm.py
------
Configures the local Phi-3 model via Ollama using LangChain's OllamaLLM wrapper.
This single module is imported by both verdict1 and verdict2 so the model is
instantiated in one place.
"""

from langchain_ollama import OllamaLLM


def get_llm(model: str = "phi3", temperature: float = 0.1) -> OllamaLLM:
    """
    Returns a LangChain OllamaLLM instance pointed at a locally-running
    Ollama server (default: http://localhost:11434).

    Args:
        model:       Ollama model tag to use (default "phi3").
        temperature: Sampling temperature — keep low for deterministic outputs.

    Returns:
        OllamaLLM instance.
    """
    llm = OllamaLLM(
        model=model,
        temperature=temperature,
        # Ollama's default base URL; override via OLLAMA_HOST env var if needed.
        base_url="http://localhost:11434",
    )
    return llm
