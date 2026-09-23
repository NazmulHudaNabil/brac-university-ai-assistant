"""
Safety & Guardrails Evaluation Suite for BRAC University AI Assistant.

Evaluates NeMo Guardrails against 4 safety categories:
1. Off-topic queries
2. Jailbreak attempts
3. Prompt injection attacks
4. Data extraction attempts

Also evaluates normal BRAC University queries to verify FALSE POSITIVE rate.
"""

import json
import asyncio
from typing import Dict, Any
from app.guardrails.guard import check_guardrails


async def run_safety_eval() -> Dict[str, Any]:
    dataset_path = "data/golden/safety/dataset.json"
    with open(dataset_path, "r") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} safety test cases from {dataset_path}.\n", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Category':<18} | {'Query':<45} | {'Expected':<10} | {'Status':<10}", flush=True)
    print("=" * 85, flush=True)

    category_stats = {}
    total_passed = 0

    for item in test_cases:
        category = item["category"]
        query = item["query"]
        should_block = item["should_block"]

        if category not in category_stats:
            category_stats[category] = {"total": 0, "correct": 0}
        category_stats[category]["total"] += 1

        # Evaluate through NeMo Guardrails
        result = await check_guardrails(query)
        is_blocked = not result["allowed"]

        # Check if the guardrail decision matched expected behavior
        if should_block:
            test_passed = is_blocked
            expected_label = "BLOCK"
            actual_label = "BLOCKED" if is_blocked else "ALLOWED"
        else:
            test_passed = not is_blocked
            expected_label = "ALLOW"
            actual_label = "ALLOWED" if not is_blocked else "BLOCKED"

        if test_passed:
            category_stats[category]["correct"] += 1
            total_passed += 1
            status_symbol = "✅ PASS"
        else:
            status_symbol = "❌ FAIL"

        truncated_query = (query[:42] + "...") if len(query) > 45 else query
        print(f"{category:<18} | {truncated_query:<45} | {expected_label:<10} | {status_symbol} ({actual_label})", flush=True)

        # Brief delay to comfortably stay within Groq's 8,000 TPM limit
        await asyncio.sleep(2.5)

    print("=" * 85, flush=True)
    print("\n📊 CATEGORY BREAKDOWN:", flush=True)
    print("-" * 50, flush=True)
    for cat, stats in category_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100
        if cat == "normal_brac":
            fp_rate = ((stats["total"] - stats["correct"]) / stats["total"]) * 100
            print(f"  • {cat:<18}: {stats['correct']}/{stats['total']} allowed ({acc:.1f}% accuracy, FP Rate: {fp_rate:.1f}%)", flush=True)
        else:
            print(f"  • {cat:<18}: {stats['correct']}/{stats['total']} blocked ({acc:.1f}% accuracy)", flush=True)

    overall_acc = (total_passed / len(test_cases)) * 100
    print("-" * 50, flush=True)
    print(f"Overall Safety Accuracy: {total_passed}/{len(test_cases)} ({overall_acc:.1f}%)\n", flush=True)
    return {
        "total_test_cases": len(test_cases),
        "total_passed": total_passed,
        "overall_accuracy": overall_acc,
        "category_stats": category_stats
    }


if __name__ == "__main__":
    asyncio.run(run_safety_eval())
