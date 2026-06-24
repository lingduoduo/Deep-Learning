from __future__ import annotations

import json
from typing import Any

from memory_store import recall_memory


class MemoryAwareAgent:
    def __init__(self, client: Any, llm: Any, user_id: str):
        self.client = client
        self.llm = llm
        self.user_id = user_id
        self.messages: list[dict[str, Any]] = []

    def run_turn(self, user_message: str, turn_id: str) -> str:
        """
        Run recall before the LLM has a chance to rewrite the user's words.

        This is the key BM25 contribution. LLMs are good rewriting machines:
        a message like "postgres v15.3 + pgvector 0.5.1" may be normalized into
        "PostgreSQL database" before a tool call. That is often fine for dense
        retrieval, but it destroys exact tokens that BM25 can match directly.

        Verbatim pre-recall sends the original user message to Elasticsearch
        first, preserving version numbers, error codes, model names, and other
        high-signal tokens.
        """
        pre_recall_call_id = f"call_pre_{turn_id[:8]}"
        pre_recall_result = recall_memory(self.client, self.user_id, user_message, k=5)

        self.messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": pre_recall_call_id,
                        "function": {
                            "name": "recall_memory",
                            "arguments": json.dumps({"query": user_message}),
                        },
                    }
                ],
            }
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": pre_recall_call_id,
                "name": "recall_memory",
                "content": json.dumps(pre_recall_result, ensure_ascii=False),
            }
        )
        self.messages.append({"role": "user", "content": user_message})
        return self.llm.chat(self.messages)
