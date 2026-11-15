"""Title search must not scan rag_chunks for queries the trigram indexes cannot serve."""
import asyncio
from unittest.mock import MagicMock

from app.services.app_search_service import _title_match_search


def test_title_search_skips_queries_shorter_than_a_trigram():
    engine = MagicMock()
    assert asyncio.run(_title_match_search("KI", ["book"], engine)) == []
    engine.connect.assert_not_called()
