import json
from typing import Any

from memory_store import PROCEDURAL_INDEX, SEMANTIC_INDEX, write_memory


# Core consolidation prompt schema. In production, run this as a background
# batch job instead of calling it after every agent turn; otherwise each turn
# needs two LLM inferences, one for the reply and one for consolidation.
CONSOLIDATION_PROMPT = """
You are analyzing recent episodic memories to extract structured knowledge.

RECENT EPISODIC EVENTS (last 30):
{episodic_events}

EXISTING SEMANTIC FACTS (~50):
{existing_facts}

EXISTING PROCEDURALS (~20):
{existing_procedurals}

Output JSON:
{
  "new_facts": [
    {
      "fact": "...",
      "confidence": 0.0-1.0,
      "supporting_episode_ids": ["epi_123", "epi_145"],
      "supersedes_id": null
    }
  ],
  "new_procedures": [
    {
      "name": "...",
      "steps": ["step 1", "step 2"],
      "confidence": 0.0-1.0,
      "trigger_text": "when user asks about deployment"
    }
  ],
  "procedural_updates": [
    {
      "procedural_id": "proc_056",
      "success_delta": 1,
      "failure_delta": 0,
      "refined_steps": null
    }
  ]
}

Rules:
- Confidence >= 0.8 to create new procedural
- Mark supersedes_id when new fact contradicts existing one
- Include supporting_episode_ids for traceability
"""

DEFAULT_CONSOLIDATION_WINDOW_HOURS = 24
DEFAULT_MIN_EPISODIC_EVENTS_FOR_DYNAMIC_RUN = 100

def build_consolidation_prompt(
    episodic_events: list[dict[str, Any]],
    existing_facts: list[dict[str, Any]],
    existing_procedurals: list[dict[str, Any]],
) -> str:
    """
    Build the prompt for a background consolidation pass.

    Running consolidation after every user turn doubles LLM inference cost: the
    system pays once to answer and once to extract durable memories. A cheaper
    production pattern is to accumulate episodic events during the day and run
    one consolidation batch at night.

    That trades about 24 hours of memory freshness for roughly half the LLM
    inference cost. At higher traffic, trigger additional runs when the number
    of new episodic events in the last 24 hours exceeds a chosen threshold.
    """
    return CONSOLIDATION_PROMPT.format(
        episodic_events=json.dumps(episodic_events, indent=2),
        existing_facts=json.dumps(existing_facts, indent=2),
        existing_procedurals=json.dumps(existing_procedurals, indent=2),
    )


def apply_consolidation_result(
    client: Any,
    user_id: str,
    consolidation: dict[str, Any],
) -> dict[str, list[str]]:
    created = {"facts": [], "procedures": []}

    for fact in consolidation.get("new_facts", []):
        if fact.get("confidence", 0) < 0.5:
            continue
        fact_id = write_memory(
            client,
            {
                "user_id": user_id,
                "text": fact["fact"],
                "memory_type": "semantic",
                "confidence": fact.get("confidence", 0.5),
                "supporting_episode_ids": fact.get("supporting_episode_ids", []),
                "supersedes": fact.get("supersedes_id"),
            },
            SEMANTIC_INDEX,
        )
        created["facts"].append(fact_id)

    for procedure in consolidation.get("new_procedures", []):
        if procedure.get("confidence", 0) < 0.8:
            continue
        procedure_id = write_memory(
            client,
            {
                "user_id": user_id,
                "text": procedure["name"],
                "memory_type": "procedural",
                "confidence": procedure.get("confidence", 0.8),
                "steps": procedure.get("steps", []),
                "trigger_text": procedure.get("trigger_text", procedure["name"]),
                "success_count": 0,
                "failure_count": 0,
            },
            PROCEDURAL_INDEX,
        )
        created["procedures"].append(procedure_id)

    return created
