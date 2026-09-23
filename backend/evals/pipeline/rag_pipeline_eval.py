"""
Evaluate the LangGraph RAG pipeline with DeepEval.

Flow:
1. Load golden Q&A pairs.
2. Run each question through the agent to get real answers + retrieved docs.
3. Score everything with three DeepEval metrics, judged by a Groq model
   (with a smaller fallback model if the main one is rate-limited).
"""

import os
import re
import json
import time
import asyncio
from dotenv import load_dotenv
from groq import Groq, AsyncGroq, RateLimitError

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM

from app.config import config
from app.agents.graph import agent

load_dotenv()
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"

MAX_RETRIES = 5
JSON_INSTRUCTION = "\n\nReturn ONLY a valid JSON object. No extra text."


def extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply and fix trailing commas."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in the model's response.")
    snippet = text[start:end + 1]
    return re.sub(r",\s*([\]}])", r"\1", snippet)


class GroqJudge(DeepEvalBaseLLM):
    """A DeepEval-compatible judge backed by Groq, with an automatic fallback model."""

    def __init__(self):
        self.model_name = "openai/gpt-oss-120b"
        self.fallback_model = "openai/gpt-oss-20b"

        self.client = Groq(api_key=config.JUDGE_GROQ)
        self.async_client = AsyncGroq(api_key=config.JUDGE_GROQ)
        self.fallback_client = Groq(api_key=config.JUDGE_GROQ_FALLBACK)
        self.async_fallback_client = AsyncGroq(api_key=config.JUDGE_GROQ_FALLBACK)

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    def _prepare_prompt(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += JSON_INSTRUCTION
        return prompt

    # ---------- synchronous ----------

    def generate(self, prompt: str) -> str:
        prompt = self._prepare_prompt(prompt)

        for label, client, model in [
            ("main", self.client, self.model_name),
            ("fallback", self.fallback_client, self.fallback_model),
        ]:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    res = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=4096,
                    )
                    text = res.choices[0].message.content
                    extract_json(text)  # raises if the JSON is invalid
                    return text
                except RateLimitError as e:
                    if "tokens per day" in str(e):
                        print(f"{label} model: daily token limit hit, moving on.")
                        break
                    print(f"{label} model: rate limited, waiting 15s (attempt {attempt}).")
                    time.sleep(15)
                except Exception as e:
                    print(f"{label} model: attempt {attempt} failed ({e}).")
                    time.sleep(2)

        raise ValueError("Both main and fallback models failed to return valid JSON.")

    # ---------- async ----------

    async def a_generate(self, prompt: str) -> str:
        prompt = self._prepare_prompt(prompt)

        for label, client, model in [
            ("main", self.async_client, self.model_name),
            ("fallback", self.async_fallback_client, self.fallback_model),
        ]:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    res = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=4096,
                    )
                    text = res.choices[0].message.content
                    extract_json(text)
                    return text
                except RateLimitError as e:
                    if "tokens per day" in str(e):
                        print(f"{label} model: daily token limit hit, moving on.")
                        break
                    print(f"{label} model: rate limited, waiting 15s (attempt {attempt}).")
                    await asyncio.sleep(15)
                except Exception as e:
                    print(f"{label} model: attempt {attempt} failed ({e}).")
                    await asyncio.sleep(2)

        raise ValueError("Both main and fallback models failed to return valid JSON.")


async def build_pipeline_test_cases():
    """Run every golden question through the agent and turn results into DeepEval test cases."""
    with open("data/golden/generation/dataset.json") as f:
        goldens = json.load(f)

    # A deliberately vague question, to make sure the query-rewrite path gets tested.
    goldens.append({
        "question": "What is the fee?",
        "reference_answer": (
            "BRAC University has various fees including tuition fees, "
            "admission fees, and semester fees depending on the program."
        ),
    })

    test_cases = []
    print(f"Running the pipeline on {len(goldens)} questions...")

    for g in goldens:
        state = {
            "original_query": g["question"],
            "current_query": g["question"],
            "documents": [],
            "citations": [],
            "answer": "",
            "needs_rewrite": False,
            "rewrite_count": 0,
            "route": "",
        }

        final_state = await agent.ainvoke(state)
        print(f"  - '{g['question']}' -> {final_state['rewrite_count']} rewrite(s)")

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                expected_output=g["reference_answer"],
                actual_output=final_state["answer"],
                retrieval_context=[d.page_content for d in final_state.get("documents", [])],
            )
        )

    return test_cases


def main():
    test_cases = asyncio.run(build_pipeline_test_cases())
    judge = GroqJudge()

    metrics = [
        ContextualRelevancyMetric(threshold=0.5, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=0.5, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=True),
    ]

    print("\nRunning DeepEval pipeline metrics...")
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "pipeline": "langgraph_rag",
            "judge_model": "openai/gpt-oss-120b",
        },
    )


if __name__ == "__main__":
    main()