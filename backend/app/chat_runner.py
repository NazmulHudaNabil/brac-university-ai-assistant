"""
Standalone runner for app.main.chat_endpoint.
Invoked by UI or CLI across Python environments.
"""

import sys
import json
import asyncio
from app.main import chat_endpoint, ChatRequest


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No query provided"}))
        return

    query = sys.argv[1]
    conv_id = sys.argv[2] if len(sys.argv) > 2 else "default"

    chat_history = None
    if len(sys.argv) > 3 and sys.argv[3]:
        try:
            chat_history = json.loads(sys.argv[3])
        except Exception:
            chat_history = None
    elif not sys.stdin.isatty():
        try:
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                chat_history = json.loads(stdin_data)
        except Exception:
            chat_history = None

    resp = await chat_endpoint(ChatRequest(query=query, conversation_id=conv_id, chat_history=chat_history))
    
    # Output single JSON line on stdout
    print(json.dumps({
        "answer": resp.answer,
        "sources": [s.model_dump() for s in resp.sources],
        "request_id": resp.request_id
    }))


if __name__ == "__main__":
    asyncio.run(main())
