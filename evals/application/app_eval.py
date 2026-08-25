"""
Phase 12 — End-to-End Application-Level Evaluation with DeepEval G-Eval.

This script evaluates the complete user-facing application:
1. Loads 5 golden end-to-end questions from data/golden/end_to_end/dataset.json.
2. Invokes app.main.chat_endpoint to test the full pipeline
   (Guardrails -> Memory -> Agent -> Portkey Gateway -> Observability -> Citations).
3. Evaluates the answers using DeepEval G-Eval with explicit rubrics:
   - Metric 1: Answer Correctness (factual accuracy against golden reference)
   - Metric 2: Completeness (comprehensiveness without omitting key details)
4. Verifies expected behavior (e.g., citations returned).
5. Outputs real numerical scores and saves results to evals/application/eval_results.json.
"""

import os
import re
import json
import time
import asyncio

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval.g_eval import Rubric
from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq, AsyncGroq, RateLimitError

from app.config import config
from app.main import chat_endpoint, ChatRequest


# ---------------------------------------------------------------------------
# 1. Simple Groq-based Judge for DeepEval
# ---------------------------------------------------------------------------
class GroqJudge(DeepEvalBaseLLM):
    """
    Evaluation judge backed by Groq API.
    Uses primary model (openai/gpt-oss-120b) with fallback to openai/gpt-oss-20b.
    """

    def __init__(self):
        self.model_name = "openai/gpt-oss-120b"
        self.fallback_model = "openai/gpt-oss-20b"

        judge_key = config.JUDGE_GROQ or config.GROQ_API_KEY
        judge_fallback_key = config.JUDGE_GROQ_FALLBACK or config.GROQ_FALLBACK_API_KEY or judge_key

        self.client = Groq(api_key=judge_key)
        self.async_client = AsyncGroq(api_key=judge_key)
        self.fallback_client = Groq(api_key=judge_fallback_key)
        self.async_fallback_client = AsyncGroq(api_key=judge_fallback_key)

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    def _extract_json(self, text: str) -> str:
        """Extract valid JSON from model response."""
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response.")
        snippet = text[start:end + 1]
        return re.sub(r",\s*([\]}])", r"\1", snippet)

    def generate(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nReturn ONLY a valid JSON object."

        for label, client, model in [
            ("primary", self.client, self.model_name),
            ("fallback", self.fallback_client, self.fallback_model)
        ]:
            for attempt in range(1, 4):
                try:
                    res = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0
                    )
                    content = res.choices[0].message.content
                    self._extract_json(content)
                    return content
                except RateLimitError:
                    time.sleep(6)
                except Exception:
                    time.sleep(2)
        raise RuntimeError("Judge evaluation failed on both primary and fallback models.")

    async def a_generate(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nReturn ONLY a valid JSON object."

        for label, client, model in [
            ("primary", self.async_client, self.model_name),
            ("fallback", self.async_fallback_client, self.fallback_model)
        ]:
            for attempt in range(1, 4):
                try:
                    res = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0
                    )
                    content = res.choices[0].message.content
                    self._extract_json(content)
                    return content
                except RateLimitError:
                    await asyncio.sleep(6)
                except Exception:
                    await asyncio.sleep(2)
        raise RuntimeError("Async judge evaluation failed on both primary and fallback models.")




# ---------------------------------------------------------------------------
# 2. G-Eval Metrics with Explicit Rubrics
# ---------------------------------------------------------------------------
def create_geval_metrics(judge: GroqJudge):
    """
    Initializes DeepEval GEval metrics with explicit criteria, scoring steps,
    and rubrics for consistent, reproducible scoring.
    """
    correctness_metric = GEval(
        name="Answer Correctness",
        criteria=(
            "Evaluate whether the actual output is factually accurate, consistent with "
            "the expected output, and contains no contradictory or hallucinated claims."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        evaluation_steps=[
            "Extract every factual claim (numbers, names, dates, entities, relationships) made in the actual output.",
            "For each claim, check whether it is supported by, contradicted by, or absent from the expected output.",
            "Flag any claim in the actual output that does not appear in and cannot be reasonably inferred from the expected output as a possible hallucination.",
            "Treat differences in phrasing, order, or verbosity as irrelevant if the underlying facts match.",
            "Weigh contradictions of core facts (wrong numbers, wrong entities, reversed relationships) as more severe than omissions or stylistic differences.",
            "Assign a final score between 0.0 and 1.0 reflecting the proportion and severity of factual errors found."
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent with the expected output. No contradictions or fabricated details. Brevity is fine."),
            Rubric(score_range=(5, 8), expected_outcome="Core facts are correct, but there is one minor inaccuracy, an unsupported side-claim, or a small numerical/detail error that doesn't change the overall answer."),
            Rubric(score_range=(0, 4), expected_outcome="Contains a clear factual error, a fabricated/hallucinated claim, or a statement that directly contradicts the expected output."),
        ],
        threshold=0.7,
        model=judge
    )

    completeness_metric = GEval(
        name="Completeness",
        criteria=(
            "Evaluate whether the actual output thoroughly answers all parts of the question "
            "without omitting critical details, entities, or conditions mentioned in the expected output."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        evaluation_steps=[
            "From the input question and expected output, build a checklist of required elements: sub-questions, entities, conditions, constraints, and any multi-part requirements.",
            "Go through the checklist item by item and mark whether the actual output addresses each one.",
            "Distinguish between a fully missing requirement and a requirement that is only partially or vaguely addressed.",
            "Treat omission of a condition or constraint (e.g., a caveat, exception, or scope limit) as a completeness failure, not just a correctness one.",
            "Do not penalize the output for including correct extra information beyond what was asked, unless it displaces or obscures required content.",
            "Assign a final score between 0.0 and 1.0 based on the fraction and importance of checklist items fully addressed."
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Every required element, entity, and condition from the expected output is fully addressed. Nothing important is missing."),
            Rubric(score_range=(5, 8), expected_outcome="Most required elements are addressed, but one non-critical detail or condition is missing or underdeveloped."),
            Rubric(score_range=(0, 4), expected_outcome="One or more critical requirements, entities, or conditions from the expected output are missing entirely, leaving the answer substantially incomplete."),
        ],
        threshold=0.7,
        model=judge
    )

    return correctness_metric, completeness_metric




# ---------------------------------------------------------------------------
# 3. Application-Level Evaluation Runner
# ---------------------------------------------------------------------------
async def run_application_evaluation():
    print("=" * 70)
    print("Starting Phase 12 — End-to-End Application-Level Evaluation")
    print("=" * 70)

    # 1. Load Golden Dataset
    dataset_path = "data/golden/end_to_end/dataset.json"
    with open(dataset_path, "r") as f:
        golden_cases = json.load(f)

    print(f"\nLoaded {len(golden_cases)} end-to-end golden test cases from {dataset_path}.")

    # 2. Initialize Judge & Metrics
    judge = GroqJudge()
    correctness_metric, completeness_metric = create_geval_metrics(judge)

    results = []
    print("\nExecuting End-to-End Pipeline & Evaluating with G-Eval...\n")

    for idx, item in enumerate(golden_cases, start=1):
        question = item["question"]
        expected_output = item["expected_answer"]
        expected_behavior = item.get("expected_behavior", "answer_with_citations")

        print(f"[{idx}/{len(golden_cases)}] Question: '{question}'")

        # Execute through complete user-facing application endpoint
        req = ChatRequest(query=question, conversation_id=f"eval_case_{idx}")
        response = await chat_endpoint(req)
        actual_output = response.answer
        citations_count = len(response.sources)

        # Check behavior
        citations_pass = citations_count > 0 if expected_behavior == "answer_with_citations" else True

        # Build DeepEval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output
        )

        # Measure Answer Correctness
        correctness_metric.measure(test_case)
        c_score = correctness_metric.score
        c_reason = correctness_metric.reason

        # Measure Completeness
        completeness_metric.measure(test_case)
        comp_score = completeness_metric.score
        comp_reason = completeness_metric.reason

        # Log individual case result
        print(f"      • Correctness:  {c_score:.2f} ({'PASS' if c_score >= 0.7 else 'FAIL'}) — {c_reason}")
        print(f"      • Completeness: {comp_score:.2f} ({'PASS' if comp_score >= 0.7 else 'FAIL'}) — {comp_reason}")
        print(f"      • Citations:    {citations_count} sources ({'PASS' if citations_pass else 'FAIL'})")
        print("-" * 70)

        results.append({
            "id": item.get("id", f"case_{idx}"),
            "question": question,
            "actual_output": actual_output,
            "expected_output": expected_output,
            "correctness_score": c_score,
            "correctness_reason": c_reason,
            "completeness_score": comp_score,
            "completeness_reason": comp_reason,
            "citations_count": citations_count,
            "citations_pass": citations_pass
        })

        # Pause to respect Groq rate limits on free tier
        await asyncio.sleep(3)

    # 4. Summary Statistics
    avg_correctness = sum(r["correctness_score"] for r in results) / len(results)
    avg_completeness = sum(r["completeness_score"] for r in results) / len(results)
    citations_pass_rate = (sum(1 for r in results if r["citations_pass"]) / len(results)) * 100

    print("\n" + "=" * 70)
    print("📊 PHASE 12 APPLICATION EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Cases Evaluated:   {len(results)}")
    print(f"Average Answer Correctness:   {avg_correctness:.2f} (Threshold: 0.70)")
    print(f"Average Completeness:         {avg_completeness:.2f} (Threshold: 0.70)")
    print(f"Citations Pass Rate:          {citations_pass_rate:.1f}%")
    print("=" * 70)

    # Save results to file
    os.makedirs("evals/application", exist_ok=True)
    with open("evals/application/eval_results.json", "w") as f:
        json.dump({
            "summary": {
                "total_cases": len(results),
                "avg_correctness": avg_correctness,
                "avg_completeness": avg_completeness,
                "citations_pass_rate": citations_pass_rate
            },
            "results": results
        }, f, indent=2)

    print("\n✅ Results saved to evals/application/eval_results.json")
    assert avg_correctness >= 0.7, "Average Correctness must be >= 0.70"
    assert avg_completeness >= 0.7, "Average Completeness must be >= 0.70"


if __name__ == "__main__":
    asyncio.run(run_application_evaluation())
