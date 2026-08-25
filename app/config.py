import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --------------------------------------------------------------------
    # 1. Groq LLM API Keys (Inference Engine)
    # --------------------------------------------------------------------
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.environ.get("GROQ_FALLBACK_API_KEY")

    # --------------------------------------------------------------------
    # 2. Portkey LLM Gateway
    # --------------------------------------------------------------------
    PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")
    GROQ_SLUG = os.environ.get("GROQ_SLUG", "rag")            # Portkey virtual slug: primary
    GROQ_SLUG_2 = os.environ.get("GROQ_SLUG_2", "brag")       # Portkey virtual slug: fallback
    PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "openai/gpt-oss-120b")
    FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "openai/gpt-oss-20b")

    # --------------------------------------------------------------------
    # 3. DeepEval Evaluation LLMs (Judge Keys)
    # --------------------------------------------------------------------
    JUDGE_GROQ = os.environ.get("JUDGE_GROQ", "openai/gpt-oss-120b")
    JUDGE_GROQ_FALLBACK = os.environ.get("JUDGE_GROQ_FALLBACK", "openai/gpt-oss-20b")

    # --------------------------------------------------------------------
    # 4. Qdrant Vector Database (Cloud)
    # --------------------------------------------------------------------
    QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.environ.get("QDRANT_CLUSTER_ENDPOINT")
    COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "bracu_docs")

    # --------------------------------------------------------------------
    # 5. Reranker & Embeddings
    # --------------------------------------------------------------------
    JINA_API_KEY = os.environ.get("JINA_API_KEY")
    JINA_EMBEDDING_API_KEY = os.environ.get("JINA_EMBEDDING_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2-preview")
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

    # --------------------------------------------------------------------
    # 6. Observability & Tracing
    # --------------------------------------------------------------------
    LOGFIRE_TOKEN = os.environ.get("LOGFIRE_TOKEN")
    LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "true")
    LANGSMITH_ENDPOINT = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "brac_university_ai_assistant")


config = Config()
settings = config
