#!/usr/bin/env python3
"""
Optimized Elasticsearch Geohash Implementation

This script demonstrates optimized geohash-based spatial search in Elasticsearch,
including:
- Efficient geo_point indexing
- Geohash grid aggregations for bucketing
- Optimized queries for proximity and bounding box searches
- Performance considerations for large datasets
"""

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import Dict, List, Any
import time

class GeohashElasticsearch:
    def __init__(self, host: str = "http://localhost:9200"):
        self.es = Elasticsearch(host)
        self.index_name = "places_geohash"

    def create_index(self):
        """Create optimized index with geo_point mapping"""
        mapping = {
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "location": {
                        "type": "geo_point",
                        # Enable geohash indexing for faster queries
                        "ignore_malformed": True
                    },
                    "category": {"type": "keyword"},
                    "description": {"type": "text"},
                    "geohash": {"type": "keyword"}  # Store geohash for custom processing
                }
            },
            "settings": {
                "number_of_shards": 3,  # Adjust based on data size
                "number_of_replicas": 1,
                "index.codec": "best_compression"  # Optimize storage
            }
        }

        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        self.es.indices.create(index=self.index_name, body=mapping)
        print(f"Created index: {self.index_name}")

    def index_sample_data(self):
        """Index sample geospatial data"""
        import geohash

        places = [
            {"name": "Central Park", "location": {"lat": 40.7829, "lon": -73.9654}, "category": "park"},
            {"name": "Times Square", "location": {"lat": 40.7580, "lon": -73.9855}, "category": "landmark"},
            {"name": "Brooklyn Bridge", "location": {"lat": 40.7061, "lon": -73.9969}, "category": "bridge"},
            {"name": "Empire State Building", "location": {"lat": 40.7484, "lon": -73.9857}, "category": "landmark"},
            {"name": "Statue of Liberty", "location": {"lat": 40.6892, "lon": -74.0445}, "category": "landmark"},
        ]

        actions = []
        for place in places:
            # Generate geohash for additional indexing
            gh = geohash.encode(place["location"]["lat"], place["location"]["lon"], precision=7)
            place["geohash"] = gh

            actions.append({
                "_index": self.index_name,
                "_source": place
            })

        bulk(self.es, actions)
        self.es.indices.refresh(index=self.index_name)
        print(f"Indexed {len(places)} places")

    def proximity_search(self, lat: float, lon: float, distance: str = "5km"):
        """Optimized proximity search using geo_distance query"""
        query = {
            "query": {
                "bool": {
                    "must": {
                        "geo_distance": {
                            "distance": distance,
                            "location": {"lat": lat, "lon": lon}
                        }
                    }
                }
            },
            "sort": [
                {
                    "_geo_distance": {
                        "location": {"lat": lat, "lon": lon},
                        "order": "asc",
                        "unit": "km"
                    }
                }
            ],
            "size": 10
        }

        start_time = time.time()
        response = self.es.search(index=self.index_name, body=query)
        elapsed = time.time() - start_time

        print(".2f")
        return response

    def bounding_box_search(self, top_left: Dict, bottom_right: Dict):
        """Bounding box search for rectangular areas"""
        query = {
            "query": {
                "geo_bounding_box": {
                    "location": {
                        "top_left": top_left,
                        "bottom_right": bottom_right
                    }
                }
            },
            "size": 20
        }

        response = self.es.search(index=self.index_name, body=query)
        return response

    def geohash_aggregation(self, precision: int = 5):
        """Geohash grid aggregation for clustering points"""
        query = {
            "size": 0,
            "aggs": {
                "location_clusters": {
                    "geohash_grid": {
                        "field": "location",
                        "precision": precision
                    },
                    "aggs": {
                        "centroid": {
                            "geo_centroid": {
                                "field": "location"
                            }
                        },
                        "doc_count": {
                            "value_count": {
                                "field": "name"
                            }
                        }
                    }
                }
            }
        }

        response = self.es.search(index=self.index_name, body=query)
        return response

    def geotile_aggregation(self, precision: int = 10):
        """Geotile grid aggregation (alternative to geohash)"""
        query = {
            "size": 0,
            "aggs": {
                "tile_clusters": {
                    "geotile_grid": {
                        "field": "location",
                        "precision": precision
                    }
                }
            }
        }

        response = self.es.search(index=self.index_name, body=query)
        return response

    def optimized_geo_query(self, lat: float, lon: float, radius: str = "10km", category: str = None):
        """Combined query with filters and geo search"""
        must_clauses = [
            {
                "geo_distance": {
                    "distance": radius,
                    "location": {"lat": lat, "lon": lon}
                }
            }
        ]

        if category:
            must_clauses.append({"term": {"category": category}})

        query = {
            "query": {
                "bool": {
                    "must": must_clauses
                }
            },
            "sort": [
                {
                    "_geo_distance": {
                        "location": {"lat": lat, "lon": lon},
                        "order": "asc"
                    }
                }
            ],
            "_source": ["name", "location", "category"],
            "size": 50
        }

        response = self.es.search(index=self.index_name, body=query)
        return response

def main():
    geo_es = GeohashElasticsearch()

    # Setup
    geo_es.create_index()
    geo_es.index_sample_data()

    # Demonstrate queries
    print("\n=== Proximity Search ===")
    results = geo_es.proximity_search(40.7580, -73.9855, "2km")
    for hit in results["hits"]["hits"]:
        print(f"{hit['_source']['name']}: {hit['sort'][0]:.2f} km away")

    print("\n=== Geohash Aggregation ===")
    agg_results = geo_es.geohash_aggregation(precision=6)
    for bucket in agg_results["aggregations"]["location_clusters"]["buckets"][:5]:
        print(f"Geohash {bucket['key']}: {bucket['doc_count']} places")

    print("\n=== Optimized Geo Query ===")
    opt_results = geo_es.optimized_geo_query(40.7580, -73.9855, "5km", "landmark")
    for hit in opt_results["hits"]["hits"][:3]:
        print(f"{hit['_source']['name']} ({hit['_source']['category']})")

if __name__ == "__main__":
    main()