# src/vectordb_client.py
import os
from typing import List, Dict, Any
import requests
from .config.config_manager import get_config

# Initialize config
config = get_config()


class VectorDBClient:
    def __init__(self):

        self.base_url = os.getenv("VECTORDB_URL", config.services.vectordb_url)
        self.api_key = os.getenv("VECTORDB_API_KEY", "dev-vectordb-key-12345")
        self.headers = {"X-API-Key": self.api_key}
        self.timeout = config.vectordb.timeout

    def ingest(self, collection_name: str, documents: List[Dict[str, Any]]):
        payload = {
            "collection_name": collection_name,
            "documents": documents,
        }
        resp = requests.post(
            f"{self.base_url}/ingest",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def query(self, collection_name: str, query_text: str, n_results: int = 5, where: Dict[str, Any] | None = None):
        """
        Query vector database.
        
        Args:
            collection_name: Collection to query
            query_text: Search query
            n_results: Number of results (default: 5, but typically overridden by caller)
            where: Optional metadata filters
        """
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
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()
