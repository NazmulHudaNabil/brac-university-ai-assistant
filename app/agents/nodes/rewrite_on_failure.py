import logfire
from app.agents.state import AgentState
from app.gateway import llm_gateway

async def rewrite_query_node(state: AgentState):
    """
    Rewrites the query if the original query did not yield sufficient context.
    """
    query = state["current_query"]
    rewrite_count = state.get("rewrite_count", 0)
    
    with logfire.span("rewrite_query_node", query=query, rewrite_count=rewrite_count):
        # If we've rewritten too many times, stop trying
        if rewrite_count >= 2:
            return {"current_query": query, "rewrite_count": rewrite_count, "needs_rewrite": False}
            
        prompt = f"""You are a query rewriting assistant for BRAC University.
The user's original query did not retrieve good results. Rewrite the query to be more specific, clearer, or use different keywords related to BRAC University.
Do not answer the query, just output the new optimized search query as plain text.

Original Query: {query}"""

        try:
            response = await llm_gateway.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            new_query = response.choices[0].message.content.strip().strip('"')
        except Exception:
            new_query = query
            
        logfire.info("Rewrote query", original=query, rewritten=new_query)
        return {"current_query": new_query, "rewrite_count": rewrite_count + 1}
