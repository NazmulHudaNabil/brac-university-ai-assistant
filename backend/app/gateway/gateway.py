"""gateway.py — Portkey LLM gateway config for LangChain nodes."""

import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Production gateway config:
#   - Fallback: primary @{GROQ_SLUG}/{PRIMARY_MODEL} -> @{GROQ_SLUG_2}/{FALLBACK_MODEL} on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to
#     simple caching on free/starter plans, so this still works either way)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503],
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/{settings.PRIMARY_MODEL}"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/{settings.FALLBACK_MODEL}"}},
    ],
}

# Native Portkey client — use this if you ever call the LLM directly
# (outside LangChain) and want Portkey's fallback/cache/retry behavior.
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=GATEWAY_CONFIG,
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    logfire.info("Creating Portkey LLM client", feature=feature, model=settings.PRIMARY_MODEL)

    return ChatOpenAI(
        openai_api_key=settings.PORTKEY_API_KEY or "dummy_key",
        api_key=settings.PORTKEY_API_KEY or "dummy_key",
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{settings.PRIMARY_MODEL}",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=GATEWAY_CONFIG,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production",
            },
        ),
    )


def extract_cache_status(raw_response) -> str:
    """
    Pull x-portkey-cache-status from a Portkey raw response.

    Important: this only works if you called the client via `.with_raw_response`,
    e.g.:
        raw = portkey_client.with_raw_response.chat.completions.create(...)
        status = extract_cache_status(raw)
        response = raw.parse()   # the actual completion object

    A normal (non-raw) response object has no `.headers`, so this always
    returns "MISS" for those — that's expected, not a bug in this function.
    """
    headers = getattr(raw_response, "headers", None) or {}
    status = headers.get("x-portkey-cache-status", "MISS").upper()
    logfire.info("Portkey cache status", status=status)
    return status


class LLMGateway:
    """Helper wrapper supporting chat.completions.create with automatic fallback."""
    class Chat:
        class Completions:
            async def create(self, **kwargs):
                from groq import AsyncGroq
                api_key = settings.GROQ_API_KEY or settings.GROQ_FALLBACK_API_KEY
                client = AsyncGroq(api_key=api_key)
                model_name = kwargs.get("model", settings.PRIMARY_MODEL)
                if model_name.startswith("@"):
                    model_name = settings.PRIMARY_MODEL
                kwargs["model"] = model_name
                try:
                    return await client.chat.completions.create(**kwargs)
                except Exception as err:
                    logfire.warn("Primary key failed, routing to fallback", error=str(err))
                    fallback_key = settings.GROQ_FALLBACK_API_KEY or settings.GROQ_API_KEY
                    client_fb = AsyncGroq(api_key=fallback_key)
                    kwargs["model"] = settings.FALLBACK_MODEL
                    return await client_fb.chat.completions.create(**kwargs)

        def __init__(self):
            self.completions = self.Completions()

    def __init__(self):
        self.chat = self.Chat()


llm_gateway = LLMGateway()