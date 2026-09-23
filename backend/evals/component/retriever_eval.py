import os
# Increase timeout so DeepEval doesn't crash while we wait for Groq TPM resets
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"
import json
import time
import asyncio
from dotenv import load_dotenv
from groq import Groq, AsyncGroq, RateLimitError

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from app.services.retrieval.retriever import get_retriever

load_dotenv()

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = "openai/gpt-oss-20b"
        api_key = os.environ.get("JUDGE_GROQ")
        self.client = Groq(api_key=api_key, max_retries=2)
        self.async_client = AsyncGroq(api_key=api_key, max_retries=2)
    
    def load_model(self): return self.client
    def get_model_name(self): return self.model_name
    
    def generate(self, prompt: str) -> str:
        for _ in range(10):
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0,
                    response_format={"type": "json_object"}
                )
                return res.choices[0].message.content
            except RateLimitError:
                time.sleep(15)

    async def a_generate(self, prompt: str) -> str:
        for _ in range(10):
            try:
                res = await self.async_client.chat.completions.create(
                    model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0,
                    response_format={"type": "json_object"}
                )
                return res.choices[0].message.content
            except RateLimitError:
                await asyncio.sleep(15)

if __name__ == "__main__":
    with open("data/golden/retrieval/dataset.json") as f:
        goldens = json.load(f)

    retriever = get_retriever(k=5)
    judge = GroqJudge()
    
    test_cases = [
        LLMTestCase(
            input=g["query"],
            expected_output=g["expected_output"],
            retrieval_context=[d.page_content for d in retriever.invoke(g["query"])],
        ) for g in goldens
    ]

    evaluate(
        test_cases=test_cases,
        metrics=[
            ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True),
            ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True)
        ],
        hyperparameters={
            "retriever": "base_k5",
            "embedding_model": "jina-embeddings-v5-text-small",
            "top_k": 5,
            "judge_model": "openai/gpt-oss-20b",
        },
    )
