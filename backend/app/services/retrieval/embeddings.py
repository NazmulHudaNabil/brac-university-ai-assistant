import time
from langchain_community.embeddings import JinaEmbeddings
from app.config import config

class RateLimitedJinaEmbeddings(JinaEmbeddings):
    def embed_documents(self, texts):
        while True:
            try:
                return super().embed_documents(texts)
            except RuntimeError as e:
                if "rate limit exceeded" in str(e).lower():
                    print("Jina API rate limit hit. Sleeping for 30 seconds...")
                    time.sleep(30)
                else:
                    raise e
                    
    def embed_query(self, text):
        while True:
            try:
                return super().embed_query(text)
            except RuntimeError as e:
                if "rate limit exceeded" in str(e).lower():
                    print("Jina API rate limit hit. Sleeping for 30 seconds...")
                    time.sleep(30)
                else:
                    raise e

def get_embeddings():
    return RateLimitedJinaEmbeddings(
        jina_api_key=config.JINA_API_KEY,
        model_name="jina-embeddings-v5-text-small"
    )
