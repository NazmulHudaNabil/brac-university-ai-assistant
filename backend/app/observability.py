"""
app/observability.py — Simple, unified Logfire and LangSmith setup.
"""

import os
import logfire
from app.config import settings


def setup_observability(app=None):
    """
    Initializes Logfire and LangSmith tracing with configuration from settings.
    """
    # 1. Configure LangSmith & LangChain Tracing
    if settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING or "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT or "https://api.smith.langchain.com"
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT or "brac_university_ai_assistant"
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT or "brac_university_ai_assistant"

    # 2. Configure Pydantic Logfire
    if settings.LOGFIRE_TOKEN:
        logfire.configure(token=settings.LOGFIRE_TOKEN, inspect_arguments=False)


# Initialize at module load so any standalone run is traced
setup_observability()
