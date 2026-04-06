"""
Multi-provider LLM factory.

Supported providers (set via LLM_PROVIDER env var):
  gemini  — Google Gemini (default; free tier available)
  openai  — OpenAI GPT
  claude  — Anthropic Claude

Usage:
    from src.llm import get_llm, get_eval_llm

    llm = get_llm()           # uses LLM_PROVIDER from env
    eval_llm = get_eval_llm() # uses EVAL_LLM_PROVIDER (cheaper model for evals)
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


def get_llm(provider: Optional[str] = None, **kwargs) -> BaseChatModel:
    """
    Return a LangChain chat model for the given provider.

    Args:
        provider: One of "gemini", "openai", "claude".
                  Defaults to LLM_PROVIDER env var (fallback: "gemini").
        **kwargs: Override model-specific params (e.g. temperature, model).
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()

    if provider == "gemini":
        return _gemini_llm(**kwargs)
    elif provider == "openai":
        return _openai_llm(**kwargs)
    elif provider == "claude":
        return _claude_llm(**kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Choose one of: gemini, openai, claude"
        )


def get_eval_llm(**kwargs) -> BaseChatModel:
    """
    Return a cheaper LLM intended for RAGAS / DeepEval evaluation calls.
    Configured via EVAL_LLM_PROVIDER (defaults to same as LLM_PROVIDER).
    """
    provider = os.getenv(
        "EVAL_LLM_PROVIDER",
        os.getenv("LLM_PROVIDER", "gemini")
    ).lower()
    return get_llm(provider=provider, **kwargs)


# ── Provider implementations ──────────────────────────────────────────────────

def _gemini_llm(**kwargs) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError("Run: pip install langchain-google-genai")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in environment.")

    params = {
        "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        "google_api_key": api_key,
        "temperature": 0.1,
    }
    params.update(kwargs)
    return ChatGoogleGenerativeAI(**params)


def _openai_llm(**kwargs) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Run: pip install langchain-openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")

    params = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "api_key": api_key,
        "temperature": 0.1,
    }
    params.update(kwargs)
    return ChatOpenAI(**params)


def _claude_llm(**kwargs) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError("Run: pip install langchain-anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")

    params = {
        "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        "api_key": api_key,
        "temperature": 0.1,
    }
    params.update(kwargs)
    return ChatAnthropic(**params)
