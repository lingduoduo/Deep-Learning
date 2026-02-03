# Elasticsearch Overview

Elasticsearch is a **distributed, RESTful search and analytics engine** built on top of **Apache Lucene**.
 It is designed to be **fast, scalable, and easy to use**, and is one of the most popular **search-optimized databases** in production today.

Elasticsearch is widely used by companies such as **Netflix, Uber, Yelp**, and many others for search, analytics, and observability use cases.

------

## Getting Started: Running Elasticsearch Locally

You can quickly spin up a single-node Elasticsearch instance using Docker:

```
docker --version

docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.3
```

This setup is ideal for **local development and learning**, with security disabled and a single-node cluster configuration.

------

## Core Concepts and Query Flow

The sections below provide a **concise, structured explanation of core Elasticsearch concepts**, organized around how a **user query** flows through the system and how data is modeled, searched, and returned.

------

## 1. User Query (Search Request)

A **user query** is the input that initiates a search.
 In Elasticsearch, queries are expressed using the **Query DSL (Domain-Specific Language)** in JSON format.

### Key Ideas

- Queries describe **what to search for**
- Filters describe **constraints** (often cached and do not affect scoring)
- Queries can be combined, nested, boosted, or weighted

### Common Query Types

- **Full-text search**: `match`, `multi_match`
- **Exact match**: `term`, `terms`
- **Range queries**: `range`
- **Boolean logic**: `bool` (`must`, `should`, `filter`, `must_not`)

------

## 2. Sorting

**Sorting** controls the **order of search results**.

### Common Sorting Criteria

- Relevance score (`_score`) — default for full-text queries
- Numeric or date fields (e.g., price, timestamp)
- Keyword fields (lexicographic order)
- Script-based sorting (advanced use cases)

### Examples

- Sort by relevance first, then by recency
- Sort products by price (ascending or descending)

------

## 3. Facets (Aggregations)

In modern Elasticsearch, **facets are implemented as Aggregations**.

### Purpose

- Summarize search results
- Enable filtering and exploration (common in analytics and e-commerce)

### Common Aggregation Types

- **Terms aggregation** — categories, tags, brands
- **Range / Histogram** — price ranges, time buckets
- **Metrics** — count, average, min, max
- **Nested aggregations** — hierarchical summaries

### Examples

- Count of documents per category
- Average price per brand

------

## 4. Results (Lists of Results)

Search results are returned as a **ranked list of documents**.

### Typical Result Fields

- `_id` — document identifier
- `_score` — relevance score (if scoring applies)
- `_source` — original JSON document
- Highlighted fields (optional)
- Match explanations (optional)

### Pagination Options

- `from` + `size` — basic pagination
- `search_after` — deep pagination
- `scroll` — large batch processing

------

## 5. Documents

A **document** is the basic unit of data in Elasticsearch.

### Characteristics

- Stored as a **JSON blob**
- Represents one logical record (e.g., product, article, log)
- Schema-less at write time but governed by mappings

### Example Document

```
{
  "title": "Elasticsearch Basics",
  "author": "Emily",
  "published_date": "2024-01-01",
  "tags": ["search", "elasticsearch"],
  "views": 1024
}
```

------

## Cluster Architecture

Elasticsearch is a **distributed system** composed of multiple **nodes** working together as a cluster.
 Each node plays one or more specialized roles.

------

## Node Types

When you start an Elasticsearch cluster, you are actually starting multiple nodes.
 Nodes can be configured with different responsibilities.

### Master Node

- Coordinates the cluster
- Performs cluster-level operations:
  - Creating or deleting indices
  - Adding or removing nodes
- Only **one active master** exists at any time

### Data Node

- Stores indexed data
- Executes search and aggregation queries, heavily i/o (read/write) requests
- The primary workhorse of Elasticsearch
- Large clusters typically have many data nodes

### Coordinating Node

- Acts as the **query router**
- Receives search requests from clients
- Distributes queries to data nodes
- Gathers and merges results
- Often considered the **frontend** of the cluster

### Ingest Node

- Handles **data ingestion and transformation**
- Executes ingest pipelines (e.g., parsing, enrichment, normalization)
- Useful for ETL-style preprocessing before indexing

### Machine Learning Node

- Runs Elasticsearch’s built-in ML features
- Used for anomaly detection and advanced analytics
- Might need to access GPU

------

## Node Configuration and Deployment

- A single Elasticsearch instance can serve **multiple roles**
- For example, a node can be both **master-eligible** and a **coordinating node**
- In production, roles are often **separated**:
  - Ingest nodes → CPU-heavy
  - Data nodes → high disk I/O and memory
  - Master nodes → stability and coordination

------

## Data Node Specialization

Data nodes can be specialized based on data lifecycle and access patterns:

- **Hot** — frequently queried, recently written data
- **Warm** — less frequently accessed
- **Cold** — rarely queried, mostly read-only
- **Frozen** — archival data, queried infrequently

This tiered approach enables **cost-efficient storage and querying**.

------

## Cluster Coordination

- When a cluster starts, a set of **master-eligible nodes** participate in a leader election
- One node becomes the active master
- Other master-eligible nodes remain on standby
- This ensures **high availability and fault tolerance**

------

## Where Search Happens

While ingest and coordinating nodes are important, **data nodes** are where search execution occurs:

- Queries are executed against shards on data nodes
- Results are scored and aggregated
- Final results are merged and returned to the client

Understanding **data nodes** is key to understanding Elasticsearch performance and scalability.