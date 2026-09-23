from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    State for the LangGraph agent across all nodes.
    """
    original_query: str
    current_query: str      # Changes if rewritten
    documents: List[Document]
    citations: List[dict]
    answer: str
    needs_rewrite: bool
    rewrite_count: int
    route: str
    chat_history: List[dict]
