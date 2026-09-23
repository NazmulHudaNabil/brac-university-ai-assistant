import json
import logfire
from langsmith import traceable
from app.agents.state import AgentState
from app.gateway import llm_gateway

@traceable(name="Query rewrite", run_type="chain")
async def query_analysis_node(state: AgentState):
    """
    Analyzes the query to determine if it requires retrieval (informational)
    or if it's a simple greeting/conversational query.
    """
    query = state["original_query"]
    chat_history = state.get("chat_history", [])

    with logfire.span("Query rewrite"):
        # Format chat history
        history_text = "None"
        if chat_history:
            history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]]) # Last 4 turns
            
        prompt = f"""You are an intelligent router for BRAC University AI assistant.
Given the chat history and the latest user query, perform two tasks:
1. Rewrite the query into a standalone informational query if it relies on context from the history (e.g., "what about CSE?", "how much is it?"). If no history is needed or it's a casual greeting, output it as is.
2. Determine if the standalone query requires searching the university's knowledge base (informational) or if it can be answered directly (conversational).

Chat History:
{history_text}

Latest Query: {query}

Output JSON with two keys:
"standalone_query": <string>
"route": <"retrieval" or "conversational">"""
        
        try:
            response = await llm_gateway.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            route = result.get("route", "retrieval")
            standalone_query = result.get("standalone_query", query)
        except Exception:
            route = "retrieval" # default to retrieval on failure
            standalone_query = query
            
        return {"route": route, "current_query": standalone_query, "rewrite_count": state.get("rewrite_count", 0)}

