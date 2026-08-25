import asyncio
import copy
import time
from typing import Optional, Sequence

import httpx
import requests
from langchain_core.callbacks.manager import Callbacks
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from app.config import config
from app.services.retrieval.embeddings import get_embeddings
import logfire
from langsmith import traceable

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


class JinaRerank(BaseDocumentCompressor):
    jina_api_key: str
    model: str = "jina-reranker-v2-base-multilingual"
    top_n: int = 5
    min_relevance_score: float = 0.0  # set >0 to drop low-relevance docs even if it means fewer than top_n
    timeout: float = 15.0
    max_retries: int = 3

    def _build_request(self, documents: Sequence[Document], query: str) -> dict:
        return {
            "model": self.model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": min(self.top_n, len(documents)),
        }

    def _to_docs(self, documents: Sequence[Document], results: list) -> list[Document]:
        final_docs = []
        for res in results:
            if res["relevance_score"] < self.min_relevance_score:
                continue
            # copy so we never mutate the caller's original Document/metadata
            doc = copy.deepcopy(documents[res["index"]])
            doc.metadata["relevance_score"] = res["relevance_score"]
            final_docs.append(doc)
        return final_docs

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.jina_api_key}"}

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        payload = self._build_request(documents, query)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    JINA_RERANK_URL, headers=self._headers(), json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return self._to_docs(documents, response.json().get("results", []))
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                time.sleep(2 ** attempt)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status in (429, 500, 502, 503, 504):
                    last_err = e
                    time.sleep(2 ** attempt)
                else:
                    raise  # 4xx other than 429 is a real bug (bad payload, bad key) — don't retry blindly

        # Reranker failed after retries: fall back to unranked top_n from the vector store
        # rather than crashing the whole retrieval call.
        return list(documents[: self.top_n])

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        payload = self._build_request(documents, query)
        last_err = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(JINA_RERANK_URL, headers=self._headers(), json=payload)
                    response.raise_for_status()
                    return self._to_docs(documents, response.json().get("results", []))
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    last_err = e
                    await asyncio.sleep(2 ** attempt)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (429, 500, 502, 503, 504):
                        last_err = e
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise

        return list(documents[: self.top_n])


# Reuse the Qdrant client + embeddings across calls instead of rebuilding per request.
_client: Optional[QdrantClient] = None
_embeddings = None


def _get_shared_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=config.QDRANT_CLUSTER_ENDPOINT,
            api_key=config.QDRANT_API_KEY,
            timeout=60.0,
        )
    return _client


def _get_shared_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


def get_retriever(k: int = 8, use_reranker: bool = False):
    """
    Returns a retriever using Qdrant and Jina Embeddings.
    Optionally wraps it in a Jina Reranker to compress top-30 down to top-k.
    """
    vector_store = QdrantVectorStore(
        client=_get_shared_client(),
        collection_name=config.COLLECTION_NAME,
        embedding=_get_shared_embeddings(),
    )

    fetch_k = 30 if use_reranker else k
    base_retriever = vector_store.as_retriever(search_kwargs={"k": fetch_k})

    if not use_reranker:
        return base_retriever

    compressor = JinaRerank(
        jina_api_key=config.JINA_API_KEY,
        model="jina-reranker-v2-base-multilingual",
        top_n=k,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


@traceable(name="Embedding", run_type="embedding")
def embed_query_step(query: str) -> list:
    """Generates dense vector embeddings for the user query."""
    with logfire.span("Embedding"):
        embeddings = _get_shared_embeddings()
        return embeddings.embed_query(query)


@traceable(name="Qdrant", run_type="retriever")
def search_qdrant_step(query_vector: list, limit: int = 30) -> list[Document]:
    """Searches Qdrant vector database for matching candidate documents."""
    with logfire.span("Qdrant"):
        vector_store = QdrantVectorStore(
            client=_get_shared_client(),
            collection_name=config.COLLECTION_NAME,
            embedding=_get_shared_embeddings(),
        )
        return vector_store.similarity_search_by_vector(query_vector, k=limit)


@traceable(name="Jina reranker", run_type="tool")
def jina_rerank_step(documents: list[Document], query: str, top_k: int = 8) -> list[Document]:
    """Reranks candidate documents using Jina AI reranker API."""
    with logfire.span("Jina reranker"):
        compressor = JinaRerank(
            jina_api_key=config.JINA_API_KEY,
            model="jina-reranker-v2-base-multilingual",
            top_n=top_k,
        )
        return compressor.compress_documents(documents, query)


def retrieve_and_rerank_docs(query: str, top_k: int = 8) -> list[Document]:
    """
    Executes the 3 granular retrieval steps sequentially:
    1. Embedding -> 2. Qdrant -> 3. Jina reranker
    """
    query_vector = embed_query_step(query)
    candidate_docs = search_qdrant_step(query_vector, limit=30)
    final_docs = jina_rerank_step(candidate_docs, query, top_k=top_k)
    return final_docs



if __name__ == "__main__":
    import sys

    retriever = get_retriever(k=8, use_reranker=True)
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How many schools and departments does BRAC University currently have?"
    print(f"Retrieving for query: {query}")
    results = retriever.invoke(query)
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Title: {res.metadata.get('title')}")
        print(f"URL: {res.metadata.get('source_url')}")
        print(f"Content: {res.page_content[:200]}...")
