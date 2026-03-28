"""
llm.py
------
Configures the LLM used by verdict1.py and verdict2.py.

Uses Google Gemini (gemini-1.5-flash) by default via the Gemini API.
It requires the GOOGLE_API_KEY environment variable to be set.

On Render, set this environment variable in the dashboard:
  GOOGLE_API_KEY = AIza...
"""

import os
from dotenv import load_dotenv

# Try to load from .env file (if it exists)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

def get_llm(temperature: float = 0.1):
    """
    Returns a LangChain ChatGoogleGenerativeAI instance pointed at Google Gemini.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Please add it as an environment variable (e.g. locally or on Render)."
        )

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    print(f"[llm] Using Google Gemini: model={model}")
    
    # We use Google Generative AI chat model
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        # Setting a generic block threshold may be needed if safety settings interfere 
        # heavily with code/server patterns, but default is generally fine.
    )
