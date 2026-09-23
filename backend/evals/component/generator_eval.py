import os
# Increase timeout so DeepEval doesn't crash while we wait for rate limits
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"
import json
import asyncio
from dotenv import load_dotenv
from groq import Groq, AsyncGroq, RateLimitError, BadRequestError

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from app.services.retrieval.retriever_rerank import get_retriever

load_dotenv()

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = "openai/gpt-oss-120b"
        api_key = os.environ.get("JUDGE_GROQ")
        self.client = Groq(api_key=api_key, max_retries=2)
        self.async_client = AsyncGroq(api_key=api_key, max_retries=2)
    
    def load_model(self): return self.client
    def get_model_name(self): return self.model_name
    
    def generate(self, prompt: str) -> str:
        # Enforce JSON output via prompt engineering
        if "json" not in prompt.lower():
            prompt += "\n\nYou MUST return ONLY a valid JSON object. Do not include any conversational text."
        for _ in range(15):
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0,
                    response_format={"type": "json_object"}
                )
                return res.choices[0].message.content
            except RateLimitError:
                import time
                time.sleep(15)
            except BadRequestError as e:
                if 'json_validate_failed' in str(e):
                    continue
                raise e
        raise ValueError("Failed to generate valid JSON after multiple retries.")

    async def a_generate(self, prompt: str) -> str:
        if "json" not in prompt.lower():
            prompt += "\n\nYou MUST return ONLY a valid JSON object. Do not include any conversational text."
        for _ in range(15):
            try:
                res = await self.async_client.chat.completions.create(
                    model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0,
                    response_format={"type": "json_object"}
                )
                return res.choices[0].message.content
            except RateLimitError:
                await asyncio.sleep(15)
            except BadRequestError as e:
                if 'json_validate_failed' in str(e):
                    continue
                raise e
        raise ValueError("Failed to generate valid JSON after multiple retries.")

# ---------------------------------------------------------
# Simulate the Generation (RAG) Endpoint for Evaluation
# ---------------------------------------------------------
async def generate_rag_answer(query: str, context_docs: list) -> str:
    groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    
    context_texts = [doc.page_content for doc in context_docs]
    combined_context = "\n\n---\n\n".join(context_texts)
    
    prompt = f"""You are a helpful assistant for BRAC University.
Use the following pieces of retrieved context to answer the user's question.
If the answer cannot be found in the context, explicitly state that you do not have sufficient context to answer the question.
Do not invent or hallucinate information.

CONTEXT:
{combined_context}"""

    res = await groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.0
    )
    return res.choices[0].message.content

async def build_test_cases(goldens, retriever):
    test_cases = []
    for g in goldens:
        # Retrieve context
        docs = retriever.invoke(g["question"])
        context_texts = [d.page_content for d in docs]
        
        # Generate Answer
        actual_output = await generate_rag_answer(g["question"], docs)
        
        test_cases.append(
            LLMTestCase(
                input=g["question"],
                expected_output=g["reference_answer"],
                retrieval_context=context_texts,
                actual_output=actual_output,
            )
        )
    return test_cases

if __name__ == "__main__":
    GOLDEN_PATH = "data/golden/generation/dataset.json"
    with open(GOLDEN_PATH) as f:
        goldens = json.load(f)

    # Initialize retriever
    retriever = get_retriever(k=5, use_reranker=True)
    
    # Run Generation for all golden cases
    print("Generating answers for evaluation...")
    test_cases = asyncio.run(build_test_cases(goldens, retriever))

    # Evaluate Generator
    judge = GroqJudge()
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=True)
    ]

    print("\nRunning DeepEval generator metrics...")
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "generator_model": "openai/gpt-oss-120b",
            "judge_model": "openai/gpt-oss-120b",
        },
    )
