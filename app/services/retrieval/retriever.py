from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from app.config import config
from app.services.retrieval.embeddings import get_embeddings

def get_retriever(k: int = 5):
    """
    Returns a baseline retriever using Qdrant and Jina Embeddings.
    """
    client = QdrantClient(
        url=config.QDRANT_CLUSTER_ENDPOINT,
        api_key=config.QDRANT_API_KEY,
        timeout=60.0
    )
    
    embeddings = get_embeddings()
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )
    
    return vector_store.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    import sys
    retriever = get_retriever(k=3)
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the residential semester?"
    print(f"Retrieving for query: {query}")
    results = retriever.invoke(query)
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Title: {res.metadata.get('title')}")
        print(f"URL: {res.metadata.get('source_url')}")
        print(f"Content: {res.page_content[:200]}...")
