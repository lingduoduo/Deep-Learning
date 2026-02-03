# Elastic Search

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
The lists provide a concise, structured explanation of core Elasticsearch concepts, organized around how a **user query** flows through the system and how data is modeled, searched, and returned.

---

## 1. User Query (Search Request)

A **user query** is the input that initiates a search.  
In Elasticsearch, queries are expressed using the **Query DSL (Domain-Specific Language)** in JSON format.

### Key Ideas
- Queries describe **what to search for**
- Filters describe **constraints** (often cached, no scoring)
- Queries can be combined, nested, boosted, or weighted

### Common Query Types
- Full-text search: `match`, `multi_match`
- Exact match: `term`, `terms`
- Range queries: `range`
- Boolean logic: `bool` (`must`, `should`, `filter`, `must_not`)

---

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

---

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

---

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

---

## 5. Documents

A **document** is the basic unit of data in Elasticsearch.

### Characteristics
- Stored as a **JSON blob**
- Represents one logical record (e.g., product, article, log)
- Schema-less at write time but governed by mappings

### Example Document
```json
{
  "title": "Elasticsearch Basics",
  "author": "Emily",
  "published_date": "2024-01-01",
  "tags": ["search", "elasticsearch"],
  "views": 1024
}
