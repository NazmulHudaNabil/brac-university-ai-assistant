import os
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS"] = "600"  # per-batch now, not per-run

import re
import json
import time
import asyncio
from dotenv import load_dotenv
from groq import Groq, AsyncGroq, RateLimitError, APIStatusError, APIConnectionError, BadRequestError

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from app.services.retrieval.retriever_rerank import get_retriever

load_dotenv()

RETRYABLE = (RateLimitError, APIStatusError, APIConnectionError)
RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s")


def _extract_retry_after(err) -> float | None:
    match = RETRY_AFTER_RE.search(str(err))
    return float(match.group(1)) if match else None


class TPMRateLimiter:
    def __init__(self, tpm_limit: int = 8000, safety_margin: float = 0.85):
        self.limit = int(tpm_limit * safety_margin)
        self._window: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    def _prune(self, now: float):
        self._window = [(t, tok) for t, tok in self._window if now - t < 60]

    async def acquire(self, estimated_tokens: int):
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(now)
                used = sum(tok for _, tok in self._window)
                if used + estimated_tokens <= self.limit:
                    self._window.append((now, estimated_tokens))
                    return
                oldest_t = self._window[0][0] if self._window else now
                sleep_for = max(0.5, 60 - (now - oldest_t))
            await asyncio.sleep(min(sleep_for, 5))


def _estimate_tokens(prompt: str) -> int:
    return len(prompt) // 4 + 600


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-120b", max_retries=10, tpm_limit=8000):
        self.model_name = model_name
        self.max_retries = max_retries
        api_key = os.environ.get("JUDGE_GROQ")
        self.client = Groq(api_key=api_key, max_retries=2)
        self.async_client = AsyncGroq(api_key=api_key, max_retries=2)
        self._limiter = TPMRateLimiter(tpm_limit=tpm_limit)

    def load_model(self): return self.client
    def get_model_name(self): return self.model_name

    def _backoff(self, attempt: int, err) -> float:
        retry_after = _extract_retry_after(err)
        if retry_after is not None:
            return retry_after + 1.0
        return min(30, 3 * (2 ** attempt))

    def generate(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nYou MUST return ONLY a valid JSON object. Do not include any conversational text."
        last_err = None
        for attempt in range(self.max_retries):
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return res.choices[0].message.content
            except RETRYABLE as e:
                last_err = e
                time.sleep(self._backoff(attempt, e))
            except BadRequestError as e:
                if 'json_validate_failed' in str(e):
                    last_err = e
                    continue
                raise e
        raise RuntimeError(f"Groq judge failed after {self.max_retries} retries: {last_err}")

    async def a_generate(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nYou MUST return ONLY a valid JSON object. Do not include any conversational text."
        await self._limiter.acquire(_estimate_tokens(prompt))
        last_err = None
        for attempt in range(self.max_retries):
            try:
                res = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return res.choices[0].message.content
            except RETRYABLE as e:
                last_err = e
                await asyncio.sleep(self._backoff(attempt, e))
            except BadRequestError as e:
                if 'json_validate_failed' in str(e):
                    last_err = e
                    continue
                raise e
        raise RuntimeError(f"Groq judge failed after {self.max_retries} retries: {last_err}")


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


if __name__ == "__main__":
    with open("data/golden/retrieval/dataset.json") as f:
        goldens = json.load(f)

    retriever = get_retriever(k=5, use_reranker=True)
    judge = GroqJudge()

    all_test_cases = [
        LLMTestCase(
            input=g["query"],
            expected_output=g["expected_output"],
            retrieval_context=[d.page_content for d in retriever.invoke(g["query"])],
        ) for g in goldens
    ]

    BATCH_SIZE = 5           # tune down further (e.g. 3) if a batch still times out
    COOLDOWN_SECONDS = 45    # lets the TPM window fully clear between batches

    all_results = []
    batches = list(batched(all_test_cases, BATCH_SIZE))

    for i, batch in enumerate(batches, start=1):
        print(f"\n=== Batch {i}/{len(batches)} ({len(batch)} test cases) ===")
        result = evaluate(
            test_cases=batch,
            metrics=[
                ContextualRecallMetric(threshold=0.5, model=judge, include_reason=True),
                ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True),
            ],
            hyperparameters={
                "retriever": "reranked_k5",
                "embedding_model": "jina-embeddings-v5-text-small",
                "top_k": 5,
                "judge_model": judge.model_name,
            },
            async_config=AsyncConfig(run_async=True, max_concurrent=2),
        )
        all_results.append(result)

        if i < len(batches):
            print(f"Cooling down {COOLDOWN_SECONDS}s before next batch...")
            time.sleep(COOLDOWN_SECONDS)

    # simple aggregate pass rate across all batches
    all_test_results = [tr for r in all_results for tr in r.test_results]
    total = len(all_test_results)
    passed = sum(1 for tr in all_test_results if tr.success)
    print(f"\n=== Overall: {passed}/{total} test cases passed ===")