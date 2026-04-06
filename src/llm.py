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
from typing import Optional, Any, Dict
from langchain_core.language_models import BaseChatModel

# Provider registry — add new providers here only, no code changes elsewhere needed
_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "gemini": {
        "module": "langchain_google_genai",
        "class": "ChatGoogleGenerativeAI",
        "package": "langchain-google-genai",
        "api_key_env": "GEMINI_API_KEY",
        "api_key_param": "google_api_key",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.0-flash",
    },
    "openai": {
        "module": "langchain_openai",
        "class": "ChatOpenAI",
        "package": "langchain-openai",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_param": "api_key",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o",
    },
    "claude": {
        "module": "langchain_anthropic",
        "class": "ChatAnthropic",
        "package": "langchain-anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key_param": "api_key",
        "model_env": "CLAUDE_MODEL",
        "default_model": "claude-sonnet-4-6",
    },
}


def get_llm(provider: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Return a LangChain chat model. Provider defaults to LLM_PROVIDER env var."""
    _load_env()
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Choose from: {', '.join(_PROVIDERS)}"
        )
    return _build_llm(_PROVIDERS[provider], **kwargs)


def get_eval_llm(**kwargs) -> BaseChatModel:
    """Return a cheaper LLM for RAGAS/DeepEval. Configured via EVAL_LLM_PROVIDER."""
    _load_env()
    provider = os.getenv(
        "EVAL_LLM_PROVIDER",
        os.getenv("LLM_PROVIDER", "gemini")
    ).lower()
    return get_llm(provider=provider, **kwargs)


# ── Internal helpers ──────────────────────────────────────────────────────────

_env_loaded = False


def _load_env() -> None:
    """Load .env once per process; no-op on subsequent calls."""
    global _env_loaded
    if not _env_loaded:
        from dotenv import load_dotenv
        load_dotenv()
        _env_loaded = True


def _build_llm(config: Dict[str, Any], **kwargs) -> BaseChatModel:
    """Instantiate a LangChain chat model from a provider config dict."""
    try:
        module = __import__(config["module"], fromlist=[config["class"]])
        llm_class = getattr(module, config["class"])
    except ImportError:
        raise ImportError(f"Run: pip install {config['package']}")

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise EnvironmentError(f"{config['api_key_env']} not set in environment.")

    params: Dict[str, Any] = {
        "model": os.getenv(config["model_env"], config["default_model"]),
        config["api_key_param"]: api_key,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    }
    params.update(kwargs)
    return llm_class(**params)
