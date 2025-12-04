# src/vectordb_client.py
import os
from typing import List, Dict, Any
import requests


class VectorDBClient:
    def __init__(self):
        self.base_url = os.getenv("VECTORDB_URL", "http://research-tool-db:8003")
        self.api_key = os.getenv("VECTORDB_API_KEY", "dev-vectordb-key-12345")
        self.headers = {"X-API-Key": self.api_key}

    def ingest(self, collection_name: str, documents: List[Dict[str, Any]]):
        payload = {
            "collection_name": collection_name,
            "documents": documents,
        }
        resp = requests.post(
            f"{self.base_url}/ingest",
            json=payload,
            headers=self.headers,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def query(self, collection_name: str, query_text: str, n_results: int = 5, where: Dict[str, Any] | None = None):
        payload = {
            "query_text": query_text,
            "n_results": n_results,
            "collection_name": collection_name,
        }
        if where:
            payload["where"] = where

        resp = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers=self.headers,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
