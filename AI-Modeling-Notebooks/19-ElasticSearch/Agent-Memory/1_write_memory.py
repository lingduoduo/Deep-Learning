from memory_store import EPISODIC_INDEX, create_client, write_memory


def write_sample_episodic_memory(client=None) -> str:
    """
    Store the raw user turn as episodic memory.

    This sample writes with refresh=True inside memory_store.write_memory so
    the agent can recall the just-written document in the same turn. That is a
    good learning default, but it trades write throughput for recall
    consistency. In a high-write production agent, prefer async indexing plus a
    small agent-side "just-written" register that injects new facts into the
    LLM context until Elasticsearch's native refresh catches up.
    """
    client = client or create_client()
    return write_memory(
        client,
        {
            "user_id": "bob",
            "text": "Original user message: Help me check the deployment logs from last Friday.",
            "memory_type": "message",
            "title": "User requested deployment logs",
            "timestamp": "2026-06-19T10:30:00+08:00",
            "metadata": {"conversation_id": "demo-agent-memory"},
        },
        index=EPISODIC_INDEX,
    )


if __name__ == "__main__":
    memory_id = write_sample_episodic_memory()
    print(f"Wrote episodic memory: {memory_id}")
