# Agent Memory with Elasticsearch

This folder is a runnable Elasticsearch-backed memory sample for an agent.

It demonstrates:

- same-turn memory writes with `refresh=True`
- BM25 recall that can return episodic, semantic, and procedural memories in one query
- soft supersession for stale semantic facts
- verbatim pre-recall before an LLM call
- a concrete deployment-log assistant use case

## Run Elasticsearch

From `19-ElasticSearch`, start the local Elasticsearch container described in the parent README:

```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.3
```

Confirm it is reachable:

```bash
curl http://localhost:9200
```

## Run The Use Case

```bash
python 19-ElasticSearch/Agent-Memory/6_gbrain.py --recreate
```

`--recreate` deletes and recreates the four demo indices:

- `atlas-episodic`
- `atlas-semantic`
- `atlas-procedural`
- `atlas-catalog`

The demo seeds Bob's deployment-log memories, supersedes an outdated Grafana preference with a Kibana preference, recalls relevant memories from Elasticsearch, and prints the assistant response.

## Multi-Tenant Isolation With DLS

Atlas-style memory is multi-tenant by default: each user should only see their own memories plus shared catalog documents. Do not rely only on application code such as `if user_id == request.user_id`; that check is easy to forget in one query path.

Use Elasticsearch Document-Level Security as the primary isolation boundary. DLS requires a security-enabled Elasticsearch cluster and an administrative credential for bootstrapping users. Each user gets an API key whose role descriptor includes this DLS query:

```json
{
  "bool": {
    "should": [
      {"term": {"user_id": "bob"}},
      {"bool": {"must_not": {"exists": {"field": "user_id"}}}}
    ],
    "minimum_should_match": 1
  }
}
```

That allows documents owned by `bob` and documents without a `user_id`, which are treated as shared catalog records. The sample builds that role descriptor in `build_user_api_key_role_descriptor(...)` and mints keys through `bootstrap_users.py`:

```bash
python 19-ElasticSearch/Agent-Memory/bootstrap_users.py bob alice
```

At request time, create the Elasticsearch client with that user's API key:

```python
client = create_client("http://localhost:9200", api_key=("api-key-id", "api-key-secret"))
```

The application query still includes `_user_or_catalog_filter` as defense in depth. It is redundant on purpose; DLS protects the cluster, and the app filter protects against accidental broad queries during development.

Catalog documents participate in ranking through `CATALOG_SOURCE_PRIOR = 0.85`. This is a soft score prior, not a hard routing rule. Catalog hits are slightly de-boosted, but if a catalog document is clearly more relevant, Elasticsearch scoring or a downstream reranker can still select it.

## Step 2: One Query, Three Memory Types

The recall step sends one Elasticsearch search request across the memory indices. That single query can return different kinds of memory together:

- `episodic`: raw conversation events, such as a recent user request
- `semantic`: stable facts and preferences, such as the user's preferred summary style
- `procedural`: learned task steps, such as how to investigate a failed deployment

The agent does not need three separate retrieval calls. It can ask once, then use each hit's `memory_type` field to decide how to place the result in the LLM context.

Example response shape:

```json
[
  {
    "memory_type": "message",
    "text": "Bob asked to check failed deployment logs for the payments API last Friday."
  },
  {
    "memory_type": "semantic",
    "text": "Bob prefers concise deployment summaries with action items first."
  },
  {
    "memory_type": "procedural",
    "text": "Deployment log investigation workflow",
    "metadata": {
      "steps": ["check failed pods", "compare release diff", "summarize impact"]
    }
  }
]
```

## Step 3: Verbatim Pre-Recall Protects Exact Tokens

The BM25 leg is valuable because it preserves exact-token retrieval. LLMs are trained to rewrite and summarize, so a user message containing versions, error codes, or model names can be generalized before the agent calls `recall_memory`.

Example:

```text
User text:      postgres v15.3 + pgvector 0.5.1
LLM rewrite:    PostgreSQL database
```

If the rewritten query is sent to Elasticsearch, the exact tokens `v15.3` and `pgvector 0.5.1` may never match the indexed memory. Verbatim pre-recall bypasses that rewrite layer by sending the original user message directly to BM25 before the LLM produces a tool-call query.

The retrieval roles are split intentionally:

- BM25 captures exact tokens such as versions, error codes, product names, and model numbers.
- Dense retrieval captures semantic similarity when the wording changes.
- Extra LLM paraphrase expansion can add noise once those two legs already exist.

In ablation testing, generating two additional paraphrases with an LLM and fusing results after document-ID deduplication reduced performance. The likely reason is that BM25 already captures exact wording and dense retrieval already captures semantic rewrites; an extra paraphrase stage mostly introduces lower-precision matches.

## Step 4: Run Consolidation As A Batch Job

Consolidation turns raw episodic events into durable semantic facts and procedural memories. It is tempting to run consolidation at the end of every conversation turn, but that doubles LLM inference cost:

```text
one turn = one LLM call to answer + one LLM call to consolidate
```

For production systems, run consolidation in the background instead. A practical default is a nightly batch: accumulate one day of episodic events, then run a single consolidation job during off-peak hours.

The tradeoff is explicit:

- Memory freshness is delayed by about 24 hours.
- LLM consolidation cost drops by roughly half compared with per-turn consolidation.
- Agent replies stay fast because consolidation no longer blocks the active turn.

When traffic grows, make the batch schedule dynamic. For example, run consolidation when the number of new episodic events in the last 24 hours exceeds `N`; high event density automatically increases consolidation frequency, while quiet periods keep cost low.

## Step 5: Supersede Facts, Do Not Delete Them

Soft supersession keeps every version of a semantic fact. Updating a fact creates a chain:

```text
abc -> xyz -> pqr -> ...
```

Each old document is marked with `superseded_by`; each new document points back with `supersedes`. The default recall path hides superseded facts so the agent sees the current version. If you need the historical chain, call:

```python
recall_memory(client, user_id, query, include_superseded=True)
```

That lets you answer audit-style questions such as "what was this user's database preference three years ago?"

Harsh contradictions get a small confidence penalty. If the user says "I never said that," the replacement fact is stored at `0.9` confidence instead of `1.0`. The system treats the correction as likely true, but leaves room for later confirmation before restoring full confidence.

Consolidation is not mandatory in every deployment. For a small internal bot with 20 users, episodic memory plus manually marked semantic facts can be enough; asking users to say "remember this preference" may be cheaper than building an automatic extraction pipeline.

Atlas-style consolidation is designed for invisible memory management. If users do not know the memory system exists and will not manually maintain semantic facts, skipping consolidation effectively means you do not have a semantic layer.

## Production Notes

### Debug Painless scoring with `_explain`

Painless scripts are hard to debug from application code. A practical workflow is to reproduce the scoring query in Kibana Dev Tools with the Elasticsearch `_explain` API:

```http
GET atlas-semantic/_explain/<document_id>
{
  "query": {
    "function_score": {
      "query": {
        "multi_match": {
          "query": "failed deployment logs",
          "fields": ["text^3", "title^2", "trigger_text"]
        }
      },
      "script_score": {
        "script": {
          "source": "return _score * params.decay;",
          "params": {
            "decay": 0.72
          }
        }
      }
    }
  }
}
```

Feed `_explain` the query, document ID, and concrete script parameters. The response breaks scoring into explanation sections, so you can inspect each boost or decay multiplier instead of trying to print intermediate values from inside the application.

### Treat `refresh=True` as a consistency tradeoff

`refresh=True` makes newly written memories searchable immediately, which is useful for same-turn agent recall. The cost is lower write throughput because each write forces a refresh.

For high-write production workloads, such as hundreds of memory writes per second, move to async indexing and keep an agent-layer just-written register. The register should temporarily hold facts written during the current turn and inject them into the LLM context until Elasticsearch's native refresh interval makes those documents searchable.

## Fast Tests

The tests use a fake client so they can validate the Elasticsearch request shapes without requiring Docker:

```bash
python -m pytest 19-ElasticSearch/Agent-Memory/test_agent_memory.py -q
```
