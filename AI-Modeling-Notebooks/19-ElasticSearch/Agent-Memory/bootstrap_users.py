from __future__ import annotations

import argparse
import json

from memory_store import create_client, create_user_api_key, ensure_memory_indices


def bootstrap_users(es_url: str, user_ids: list[str]) -> dict[str, dict]:
    """
    Create DLS-scoped Elasticsearch API keys for Agent Memory users.

    Run this with an administrative Elasticsearch client. Each generated API
    key includes a role descriptor that allows the user to read only:

    - documents whose user_id matches that user
    - shared catalog documents that do not have a user_id field

    The application should use the returned per-user key for memory reads and
    writes instead of sharing one broad cluster credential across tenants.
    """
    if not user_ids:
        raise ValueError("At least one user_id is required.")

    client = create_client(es_url)
    ensure_memory_indices(client)
    return {user_id: create_user_api_key(client, user_id) for user_id in user_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DLS-scoped Agent Memory API keys.")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("user_ids", nargs="+", help="User IDs to bootstrap, such as bob alice")
    args = parser.parse_args()

    api_keys = bootstrap_users(args.es_url, args.user_ids)
    print(json.dumps(api_keys, indent=2))


if __name__ == "__main__":
    main()
