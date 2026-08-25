import asyncio
from app.agents.state import AgentState
from app.services.retrieval.retriever_rerank import retrieve_and_rerank_docs

async def retrieval_node(state: AgentState):
    """
    Retrieves documents through 3 tracked steps:
    1. Embedding -> 2. Qdrant -> 3. Jina reranker
    """
    query = state.get("current_query", state["original_query"])
    
    # Run the retrieval pipeline (Embedding -> Qdrant -> Jina reranker)
    docs = await asyncio.to_thread(retrieve_and_rerank_docs, query, 8)
    
    citations = []
    for doc in docs:
        citations.append({
            "document_id": doc.metadata.get("document_id"),
            "url": doc.metadata.get("source_url"),
            "title": doc.metadata.get("title"),
            "snippet": doc.page_content[:150] + "..."
        })
        
    return {"documents": docs, "citations": citations}
