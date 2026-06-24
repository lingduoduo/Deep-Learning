from memory_store import supersede_fact


def correct_stale_fact(client, user_id: str, old_fact_id: str) -> str:
    """
    Replace a stale fact by linking a new fact to the old one.

    The old fact is never deleted. It is marked as superseded, which preserves
    the full version chain for audits or historical questions like "what was
    this user's database preference three years ago?"
    """
    return supersede_fact(
        client,
        user_id=user_id,
        old_fact_id=old_fact_id,
        new_text="Bob prefers daily deployment summaries from Kibana.",
        contradiction_type="harsh",
    )
