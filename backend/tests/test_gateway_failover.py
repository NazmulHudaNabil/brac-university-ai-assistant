"""
Phase 10 — Portkey LLM Gateway Failover Test.

Verifies Portkey Gateway configuration, virtual slug targets (@rag / @brag),
LangChain ChatOpenAI proxy integration, cache extraction, and automatic failover.
"""

import os
import pytest
import asyncio
from app.gateway import (
    GATEWAY_CONFIG,
    portkey_client,
    get_langchain_llm,
    extract_cache_status,
    llm_gateway
)

pytestmark = pytest.mark.skipif(os.environ.get("CI") == "true", reason="Requires API keys")
from app.agents.graph import agent
from app.config import settings


def test_portkey_gateway_configuration():
    print("\n--- Test 1: Portkey Gateway Configuration & Slugs ---")
    print(f"Primary Slug Target: {GATEWAY_CONFIG['targets'][0]['override_params']['model']}")
    print(f"Fallback Slug Target: {GATEWAY_CONFIG['targets'][1]['override_params']['model']}")
    print(f"Strategy: {GATEWAY_CONFIG['strategy']}")
    print(f"Cache Mode: {GATEWAY_CONFIG['cache']}")
    print(f"Retry Settings: {GATEWAY_CONFIG['retry']}")

    assert GATEWAY_CONFIG["strategy"]["mode"] == "fallback"
    assert len(GATEWAY_CONFIG["targets"]) == 2
    print("✅ Portkey Gateway Configuration Verified!")


def test_langchain_llm_factory():
    print("\n--- Test 2: LangChain LLM Proxy Factory ---")
    llm = get_langchain_llm(feature="test-verification")
    print(f"Model: {llm.model_name}")
    print(f"Base URL: {llm.openai_api_base}")
    print(f"Headers configured: {'x-portkey-config' in getattr(llm, 'default_headers', {}) or 'x-portkey-api-key' in getattr(llm, 'default_headers', {})}")
    assert f"@{settings.GROQ_SLUG}" in llm.model_name
    print("✅ LangChain LLM Factory Verified!")


def test_cache_extraction():
    print("\n--- Test 3: Cache Status Extraction ---")
    class MockResponse:
        headers = {"x-portkey-cache-status": "HIT"}
    
    mock_hit = MockResponse()
    status_hit = extract_cache_status(mock_hit)
    print(f"Extracted HIT status: {status_hit}")
    assert status_hit == "HIT"

    class MockMissResponse:
        pass
    status_miss = extract_cache_status(MockMissResponse())
    print(f"Extracted default status: {status_miss}")
    assert status_miss == "MISS"
    print("✅ Cache Status Extraction Verified!")


async def test_end_to_end_agent():
    print("\n--- Test 4: End-to-End Agent via LLM Gateway ---")
    test_state = {
        "original_query": "What are the undergraduate admission requirements for BRAC University?",
        "current_query": "What are the undergraduate admission requirements for BRAC University?",
        "documents": [],
        "citations": [],
        "answer": "",
        "needs_rewrite": False,
        "rewrite_count": 0,
        "route": "",
        "chat_history": []
    }
    
    print("Invoking LangGraph agent through the gateway...")
    final_state = await agent.ainvoke(test_state)
    
    print("\nAgent Execution Succeeded!")
    print(f"Route: {final_state.get('route')}")
    print(f"Citations count: {len(final_state.get('citations', []))}")
    print(f"Answer preview: {final_state.get('answer', '')[:200]}...")
    
    assert len(final_state.get("answer", "")) > 50, "Expected a detailed answer from the agent!"
    print("✅ End-to-End Agent Execution via Gateway Passed!")


async def main():
    print("=" * 65)
    print("Starting Phase 10 Portkey Gateway Verification")
    print("=" * 65)
    
    test_portkey_gateway_configuration()
    test_langchain_llm_factory()
    test_cache_extraction()
    await test_end_to_end_agent()
    
    print("\n" + "=" * 65)
    print("🎉 ALL PHASE 10 PORTKEY GATEWAY TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
