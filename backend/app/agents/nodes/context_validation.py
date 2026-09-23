from app.agents.state import AgentState

async def context_validation_node(state: AgentState):
    """
    Validates if the retrieved context is sufficient.
    If documents were retrieved and reranked, proceeds directly to LLM.
    """
    docs = state.get("documents", [])
    if not docs:
        return {"needs_rewrite": True}
    return {"needs_rewrite": False}
