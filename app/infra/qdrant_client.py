"""Thin async wrapper for Qdrant's HTTP API."""
from __future__ import annotations

import logging
from typing import Iterable, List, Mapping, Sequence, Tuple

import httpx

logger = logging.getLogger(__name__)


class QdrantClient:
    """Minimal client for the subset of Qdrant endpoints we need right now."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        # Single budget for connect/read/write/pool; avoids httpx.WriteTimeout on big upsert JSON.
        self._httpx_timeout = httpx.Timeout(timeout)
        self.headers = {"api-key": api_key} if api_key else None

    async def get_version(self) -> str:
        """Return Qdrant server version from GET /."""

        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
            response = await client.get(f"{self.base_url}/")
            response.raise_for_status()
            data = response.json()
            return str(data.get("version", "unknown"))

    async def get_collection_info(self, collection: str) -> dict[str, object] | None:
        """Return collection details (points_count, etc.) or None if not found."""

        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
            response = await client.get(f"{self.base_url}/collections/{collection}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return data.get("result", {}) or {}

    async def ensure_collection(
        self,
        name: str,
        *,
        vector_size: int,
        distance: str = "Cosine",
        sparse_vector_name: str | None = None,
    ) -> None:
        """Create the collection if it does not exist."""

        payload = {
            "vectors": {"size": vector_size, "distance": distance},
            "hnsw_config": {"m": 64, "ef_construct": 512},
            "optimizers_config": {"default_segment_number": 2},
        }
        if sparse_vector_name:
            payload["sparse_vectors"] = {
                sparse_vector_name: {"index": {"on_disk": False}}
            }

        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
            response = await client.put(f"{self.base_url}/collections/{name}", json=payload)
            if response.status_code in (200, 201):
                return
            if response.status_code == 409:
                # Collection already exists – treat as success.
                return
            raise RuntimeError(
                f"Failed to ensure Qdrant collection '{name}': {response.text}"
            )

    async def ensure_text_index(
        self,
        collection: str,
        *,
        field_name: str = "text",
        wait: bool = True,
    ) -> None:
        """Ensure a text payload index exists for sparse/BM25 search."""

        payload = {"field_name": field_name, "field_schema": {"type": "text"}}
        suffix = "?wait=true" if wait else ""

        url = f"{self.base_url}/collections/{collection}/index{suffix}"
        try:
            async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
                # Qdrant payload index creation uses PUT (POST may yield 405 with empty body).
                response = await client.put(url, json=payload)
                if response.status_code in (200, 201):
                    return
                if response.status_code == 409:
                    # Index already exists – treat as success.
                    return
                content_type = response.headers.get("content-type", "<unknown>")
                body = response.text if getattr(response, "text", None) else "<empty>"
                raise RuntimeError(
                    "Failed to ensure text index for "
                    f"'{collection}' (status={response.status_code}, content-type={content_type}, url={url}): {body}"
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to ensure text index for '{collection}' (url={url}): "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

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
        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
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
        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
            response = await client.post(
                f"{self.base_url}/collections/{collection}/points/delete",
                json=payload,
            )
            response.raise_for_status()

    async def list_collections(self) -> List[Mapping[str, object]]:
        """List all collections in Qdrant."""

        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
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
        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
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

        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
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

        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
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
        with_vector_names: Sequence[str] | None = None,
    ) -> Tuple[List[Mapping[str, object]], object | None]:
        """Scroll one page of points, returning (points, next_page_offset).

        Qdrant returns `result.next_page_offset` which must be provided as `offset`
        to retrieve the next page. When it is absent/None, the scan is complete.

        If ``with_vector_names`` is set, it is sent as Qdrant's ``with_vector`` (list
        of named vectors) and overrides ``with_vectors``.
        """

        if with_vector_names is not None:
            wv: object = list(with_vector_names)
        else:
            wv = with_vectors
        payload: dict[str, object] = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": wv,
        }
        if filter_ is not None:
            payload["filter"] = filter_
        if offset is not None:
            payload["offset"] = offset

        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
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

        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
            response = await client.post(
                f"{self.base_url}/collections/{collection}/points/search", json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", []) or []

    async def ensure_sparse_config(self, collection: str, *, vector_name: str = "text-sparse") -> bool:
        """Ensure sparse slot exists, without trying unsupported in-place schema mutation.

        Qdrant 1.11 does not support adding new named vectors to an existing collection
        via PATCH. We therefore only check whether the slot already exists.
        """

        info = await self.get_collection_info(collection)
        if info is None:
            return False
        params = info.get("config", {}).get("params", {}) if isinstance(info, Mapping) else {}
        sparse_cfg = params.get("sparse_vectors", {}) if isinstance(params, Mapping) else {}
        has_sparse = isinstance(sparse_cfg, Mapping) and vector_name in sparse_cfg
        if not has_sparse:
            logger.warning(
                "Collection '%s' has no sparse vector slot '%s'; sparse search/backfill disabled "
                "(Qdrant 1.11 cannot add this to an existing collection).",
                collection,
                vector_name,
            )
        return has_sparse

    async def update_vectors(
        self,
        collection: str,
        points: Sequence[Mapping[str, object]],
        *,
        wait: bool = True,
    ) -> None:
        """Update named vectors for existing points without touching payload.

        Each entry in `points` must have: {"id": str, "vector": {name: value}}.
        Used by the migration script to backfill sparse vectors.
        """

        if not points:
            return

        payload = {"points": list(points)}
        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
            response = await client.put(
                f"{self.base_url}/collections/{collection}/points/vectors"
                f"?wait={'true' if wait else 'false'}",
                json=payload,
            )
            if response.status_code >= 400:
                try:
                    details = response.text
                except Exception:
                    details = "<no response body>"
                raise RuntimeError(
                    f"Qdrant update_vectors failed ({response.status_code}): {details}"
                )

    async def search_sparse_points(
        self,
        *,
        collection: str,
        indices: List[int],
        values: List[float],
        limit: int = 10,
        filter_: Mapping[str, object] | None = None,
        with_payload: bool = True,
    ) -> List[Mapping[str, object]]:
        """BM25 sparse vector search using Qdrant's native sparse vector index.

        Requires the collection to have a 'text-sparse' named sparse vector slot
        (created via ensure_sparse_config) and points with pre-computed BM25 vectors.
        """

        if not indices:
            return []

        body: dict[str, object] = {
            "vector": {
                "name": "text-sparse",
                "vector": {"indices": indices, "values": values},
            },
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": False,
        }
        if filter_ is not None:
            body["filter"] = filter_

        async with httpx.AsyncClient(timeout=self._httpx_timeout, headers=self.headers) as client:
            response = await client.post(
                f"{self.base_url}/collections/{collection}/points/search",
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", []) or []
