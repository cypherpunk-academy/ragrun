"""Thin async wrapper for Qdrant's HTTP API."""
from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import httpx


class QdrantClient:
    """Minimal client for the subset of Qdrant endpoints we need right now."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.headers = {"api-key": api_key} if api_key else None

    async def ensure_collection(
        self,
        name: str,
        *,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Create the collection if it does not exist."""

        payload = {
            "vectors": {"size": vector_size, "distance": distance},
            "hnsw_config": {"m": 64, "ef_construct": 512},
            "optimizers_config": {"default_segment_number": 2},
        }

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.put(f"{self.base_url}/collections/{name}", json=payload)
            if response.status_code in (200, 201):
                return
            if response.status_code == 409:
                # Collection already exists – treat as success.
                return
            raise RuntimeError(
                f"Failed to ensure Qdrant collection '{name}': {response.text}"
            )

    async def upsert_points(
        self,
        collection: str,
        points: Sequence[Mapping[str, object]],
        *,
        wait: bool = True,
    ) -> None:
        """Upsert a batch of points."""

        if not points:
            return

        payload = {"points": list(points)}
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.put(
                f"{self.base_url}/collections/{collection}/points?wait={'true' if wait else 'false'}",
                json=payload,
            )
            if response.status_code >= 400:
                details: str
                try:
                    details = response.text
                except Exception:
                    details = "<no response body>"
                raise RuntimeError(
                    f"Qdrant upsert failed ({response.status_code}): {details}"
                )

    async def delete_points(
        self,
        collection: str,
        point_ids: Iterable[str],
        *,
        wait: bool = True,
    ) -> None:
        """Delete a set of points by id."""

        ids: List[str] = list(point_ids)
        if not ids:
            return

        payload = {"points": ids, "wait": wait}
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.post(
                f"{self.base_url}/collections/{collection}/points/delete",
                json=payload,
            )
            response.raise_for_status()

    async def list_collections(self) -> List[Mapping[str, object]]:
        """List all collections in Qdrant."""

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.get(f"{self.base_url}/collections")
            response.raise_for_status()
            data = response.json()
            
            # Qdrant returns: {"result": {"collections": [...]}}
            collections_data = data.get("result", {}).get("collections", [])
            
            result: List[Mapping[str, object]] = []
            for coll in collections_data:
                name = coll.get("name", "")
                # Get detailed info for each collection
                detail_response = await client.get(f"{self.base_url}/collections/{name}")
                detail_response.raise_for_status()
                detail_data = detail_response.json()
                detail_result = detail_data.get("result", {})
                
                result.append({
                    "name": name,
                    "id": name,
                    "count": detail_result.get("points_count", 0),
                    "metadata": detail_result.get("config", {})
                })
            
            return result

