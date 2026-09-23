import os
import json
import uuid
import logfire
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from langsmith import traceable
from app.agents.graph import agent
from app.guardrails.guard import check_guardrails
from app.observability import setup_observability

app = FastAPI(title="BRAC University AI Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_observability(app)

class ChatRequest(BaseModel):
    query: str
    conversation_id: str = "default"
    chat_history: Optional[List[Dict[str, str]]] = None

class Citation(BaseModel):
    document_id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: str

class ChatResponse(BaseModel):
    request_id: str
    answer: str
    sources: List[Citation]

# Simple in-memory & file-backed store for conversation history
conversation_memory = {}
SESSION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

def load_session_history(conv_id: str) -> List[Dict[str, str]]:
    if conv_id in conversation_memory:
        return list(conversation_memory[conv_id])
    filepath = os.path.join(SESSION_DIR, f"{conv_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_session_history(conv_id: str, history: List[Dict[str, str]]):
    conversation_memory[conv_id] = history
    filepath = os.path.join(SESSION_DIR, f"{conv_id}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

@app.post("/chat", response_model=ChatResponse)
@traceable(name="Request", run_type="chain")
async def chat_endpoint(request: ChatRequest):
    request_id = str(uuid.uuid4())
    
    with logfire.span("Request", request_id=request_id, query=request.query, conversation_id=request.conversation_id):
        try:
            # Step 1: NeMo Guardrails Safety Check
            guard_result = await check_guardrails(request.query)
                
            if not guard_result["allowed"]:
                refusal_msg = guard_result.get("response", "I cannot fulfill this request as it violates safety policies.")
                logfire.info("Query blocked by guardrails", category=guard_result.get("category"))
                return ChatResponse(
                    request_id=request_id,
                    answer=refusal_msg,
                    sources=[]
                )

            # Step 2: Load conversation memory
            conv_id = request.conversation_id
            if request.chat_history is not None:
                chat_history = list(request.chat_history)
            else:
                chat_history = load_session_history(conv_id)

            initial_state = {
                "original_query": request.query,
                "current_query": request.query,
                "documents": [],
                "citations": [],
                "answer": "",
                "needs_rewrite": False,
                "rewrite_count": 0,
                "route": "",
                "chat_history": chat_history
            }
            
            # Step 3: Run the LangGraph agent with LangSmith request_id metadata
            langsmith_config = {
                "metadata": {
                    "request_id": request_id,
                    "conversation_id": conv_id,
                    "environment": "production"
                }
            }
            final_state = await agent.ainvoke(initial_state, config=langsmith_config)
            
            # Parse citations
            citations = []
            for src in final_state.get("citations", []):
                citations.append(Citation(**src))
                
            answer = final_state.get("answer", "")
            
            # Update memory
            chat_history.append({"role": "user", "content": request.query})
            chat_history.append({"role": "assistant", "content": answer})
            save_session_history(conv_id, chat_history)
            
            return ChatResponse(request_id=request_id, answer=answer, sources=citations)
            
        except Exception as e:
            logfire.error("Error processing chat request", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
