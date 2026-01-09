"""Thin async wrapper for Qdrant's HTTP API."""
from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple

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

    async def retrieve_points(
        self,
        collection: str,
        point_ids: Sequence[str],
        *,
        with_vectors: bool = False,
        with_payload: bool = True,
    ) -> List[Mapping[str, object]]:
        """Fetch specific points by id."""

        if not point_ids:
            return []
        payload = {
            "ids": list(point_ids),
            "with_vector": with_vectors,
            "with_payload": with_payload,
        }
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.post(
                f"{self.base_url}/collections/{collection}/points",
                json=payload,
            )
            if response.status_code == 404:
                # Collection does not exist yet: treat as empty so ingestion can create it later.
                return []
            response.raise_for_status()
            data = response.json()
            return data.get("result", []) or []

    async def set_payload(
        self,
        collection: str,
        updates: Sequence[Mapping[str, object]],
        *,
        wait: bool = True,
    ) -> None:
        """Update payload for existing points without touching vectors.

        Qdrant's set-payload applies the SAME payload to all provided point IDs.
        Here we send one request per payload update to allow per-point payloads.
        """

        if not updates:
            return

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            for upd in updates:
                point_id = upd.get("id")
                payload = upd.get("payload")
                if point_id is None or payload is None:
                    continue
                body = {
                    "points": [point_id],
                    "payload": payload,
                    "wait": wait,
                }
                response = await client.post(
                    f"{self.base_url}/collections/{collection}/points/payload",
                    json=body,
                )
                response.raise_for_status()

    async def scroll_points(
        self,
        collection: str,
        *,
        filter_: Mapping[str, object] | None = None,
        limit: int = 128,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Mapping[str, object]]:
        """Scroll a subset of points (used for inventories)."""

        payload: dict[str, object] = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vectors,
        }
        if filter_ is not None:
            payload["filter"] = filter_

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.post(
                f"{self.base_url}/collections/{collection}/points/scroll",
                json=payload,
            )
            if response.status_code == 404:
                return []
            response.raise_for_status()
            data = response.json()
            result = data.get("result", {})
            return result.get("points", []) or []

    async def scroll_points_page(
        self,
        collection: str,
        *,
        filter_: Mapping[str, object] | None = None,
        limit: int = 128,
        offset: object | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Tuple[List[Mapping[str, object]], object | None]:
        """Scroll one page of points, returning (points, next_page_offset).

        Qdrant returns `result.next_page_offset` which must be provided as `offset`
        to retrieve the next page. When it is absent/None, the scan is complete.
        """

        payload: dict[str, object] = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vectors,
        }
        if filter_ is not None:
            payload["filter"] = filter_
        if offset is not None:
            payload["offset"] = offset

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.post(
                f"{self.base_url}/collections/{collection}/points/scroll",
                json=payload,
            )
            if response.status_code == 404:
                return [], None
            response.raise_for_status()
            data = response.json()
            result = data.get("result", {}) or {}
            points = result.get("points", []) or []
            next_offset = result.get("next_page_offset")
            return points, next_offset

    async def scroll_all_points(
        self,
        collection: str,
        *,
        filter_: Mapping[str, object] | None = None,
        limit: int = 256,
        with_payload: bool = True,
        with_vectors: bool = False,
        max_pages: int = 10_000,
    ) -> List[Mapping[str, object]]:
        """Scroll all matching points (handles pagination).

        This is intentionally conservative (max_pages) to avoid infinite loops if
        a server returns a cyclic offset for some reason.
        """

        all_points: List[Mapping[str, object]] = []
        offset: object | None = None
        for _ in range(max_pages):
            points, offset = await self.scroll_points_page(
                collection,
                filter_=filter_,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            all_points.extend(points)
            if offset is None:
                break
        return all_points

    async def search_points(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int = 10,
        filter_: Mapping[str, object] | None = None,
        with_payload: bool = True,
    ) -> List[Mapping[str, object]]:
        """Vector search wrapper."""

        if not vector:
            return []

        payload: dict[str, object] = {
            "vector": vector,
            "limit": limit,
            "with_payload": with_payload,
        }
        if filter_ is not None:
            payload["filter"] = filter_

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.post(
                f"{self.base_url}/collections/{collection}/points/search", json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", []) or []

