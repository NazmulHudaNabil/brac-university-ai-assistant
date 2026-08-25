import os
import glob
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from app.config import config
from app.services.retrieval.embeddings import get_embeddings

def index_data(input_dir: str = "processed_data"):
    embeddings = get_embeddings()
    
    print(f"Connecting to Qdrant at {config.QDRANT_CLUSTER_ENDPOINT}...")
    client = QdrantClient(
        url=config.QDRANT_CLUSTER_ENDPOINT,
        api_key=config.QDRANT_API_KEY,
        timeout=60.0
    )
    
    if not client.collection_exists(config.COLLECTION_NAME):
        print(f"Creating collection {config.COLLECTION_NAME} in Qdrant with dimension 1024...")
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        
    print(f"Loading chunks from {input_dir}...")
    documents = []
    for file_path in glob.glob(f"{input_dir}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk["page_content"],
                        metadata=chunk["metadata"]
                    )
                )
                
    print(f"Total chunks loaded: {len(documents)}")
    if not documents:
        print("No documents found. Exiting.")
        return

    print("Uploading to Qdrant. This might take a while...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )
    
    batch_size = 80
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Uploading batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
        vector_store.add_documents(batch)
        
    print("Indexing complete!")

def test_query(query: str):
    embeddings = get_embeddings()
    client = QdrantClient(
        url=config.QDRANT_CLUSTER_ENDPOINT,
        api_key=config.QDRANT_API_KEY,
        timeout=60.0
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
    )
    
    print(f"\n--- Testing Query: '{query}' ---")
    results = vector_store.similarity_search(query, k=2)
    for res in results:
        print(f"\n[Title]: {res.metadata.get('title')}")
        print(f"[URL]: {res.metadata.get('source_url')}")
        print(f"[Chunk ID]: {res.metadata.get('chunk_id')}")
        print(f"[Text]: {res.page_content[:200]}...\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What are the admission requirements?"
        test_query(query)
    else:
        index_data()
