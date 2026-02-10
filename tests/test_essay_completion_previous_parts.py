"""Test essay completion with previous_parts parameter."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.retrieval.graphs.essay_completion import _load_previous_parts


class TestLoadPreviousParts:
    """Test the _load_previous_parts function with provided_parts parameter."""

    def test_load_previous_parts_with_mood_index_1_returns_empty(self, tmp_path):
        """Test that mood_index=1 returns empty string."""
        assistant_dir = tmp_path / "assistant"
        result = _load_previous_parts(assistant_dir, "test-essay", 1)
        assert result == ""

    def test_load_previous_parts_with_provided_parts(self, tmp_path):
        """Test that provided_parts parameter bypasses filesystem read."""
        assistant_dir = tmp_path / "assistant"
        provided_text = "## Header 1\n\nText 1\n\n## Header 2\n\nText 2"
        
        # Should return provided_parts without reading filesystem
        result = _load_previous_parts(
            assistant_dir, 
            "test-essay", 
            3, 
            provided_parts=provided_text
        )
        
        assert result == provided_text
        # Verify no filesystem access was attempted (directory doesn't even exist)
        assert not (assistant_dir / "essays").exists()

    def test_load_previous_parts_missing_provided_parts_raises_error(self, tmp_path):
        """Test that mood_index>=2 requires provided_parts."""
        assistant_dir = tmp_path / "assistant"

        with pytest.raises(ValueError, match="previous_parts is required"):
            _load_previous_parts(assistant_dir, "test-essay", 2, provided_parts=None)

    def test_load_previous_parts_empty_provided_parts_raises_error(self, tmp_path):
        """Test that provided_parts must be non-empty for mood_index>=2."""
        assistant_dir = tmp_path / "assistant"

        with pytest.raises(ValueError, match="previous_parts must be non-empty"):
            _load_previous_parts(assistant_dir, "test-essay", 2, provided_parts="  \n")


class TestEssayCompletionAPIIntegration:
    """Integration tests for the full API flow."""

    @pytest.mark.asyncio
    async def test_api_accepts_previous_parts_parameter(self):
        """Test that the API accepts and uses previous_parts parameter."""
        from app.retrieval.api.essay_completion import EssayCompletionRequest
        
        # Test that the model accepts previous_parts
        request = EssayCompletionRequest(
            assistant="philo-von-freisinn",
            essay_slug="test-essay",
            essay_title="Test Essay",
            mood_index=2,
            current_text="Current part text",
            previous_parts="## Part 1\n\nPrevious part text"
        )
        
        assert request.previous_parts == "## Part 1\n\nPrevious part text"
        assert request.assistant == "philo-von-freisinn"

    @pytest.mark.asyncio
    async def test_api_previous_parts_optional(self):
        """Test that previous_parts is optional for mood_index=1."""
        from app.retrieval.api.essay_completion import EssayCompletionRequest
        
        # Test without previous_parts (should default to None)
        request = EssayCompletionRequest(
            assistant="philo-von-freisinn",
            essay_slug="test-essay",
            essay_title="Test Essay",
            mood_index=1,
            current_text="First part text"
        )
        
        assert request.previous_parts is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
