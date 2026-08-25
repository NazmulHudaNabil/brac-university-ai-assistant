import logfire
from langsmith import traceable
from app.agents.state import AgentState
from app.gateway import llm_gateway

RAG_SYSTEM_PROMPT = """You are a helpful assistant for BRAC University.
Use the following pieces of retrieved context to answer the user's question.
If the answer cannot be found in the context, explicitly state that you do not have sufficient context to answer the question.
Do not invent or hallucinate information.

CONTEXT:
{context}
"""

CONVERSATIONAL_PROMPT = """You are a helpful and friendly assistant for BRAC University.
The user is just chatting or asking a basic non-informational question (like a greeting).
Be polite, brief, and helpful. Do not hallucinate facts about the university.
"""

@traceable(name="LLM", run_type="llm")
async def generation_node(state: AgentState):
    """
    Generates the final response based on the context and route.
    """
    route = state.get("route", "retrieval")
    query = state["original_query"]
    
    with logfire.span("LLM"):
        chat_history = state.get("chat_history", [])
        
        if route == "conversational":
            messages = [{"role": "system", "content": CONVERSATIONAL_PROMPT}]
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": query})
            citations = []
        else:
            docs = state.get("documents", [])
            context_texts = [d.page_content for d in docs]
            combined_context = "\n\n---\n\n".join(context_texts)
            
            messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=combined_context)}]
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": query})
            citations = state.get("citations", [])
        
        try:
            # Use openai/gpt-oss-120b as requested previously
            response = await llm_gateway.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                temperature=0.0
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
            
        if "insufficient context" in answer.lower() or "do not have sufficient context" in answer.lower():
            citations = []
            
        logfire.info("Generated answer", answer_length=len(answer), citations_count=len(citations))
        return {"answer": answer, "citations": citations}
