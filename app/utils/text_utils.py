import re
import html
from typing import Tuple

try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False


NORMALIZATION_VERSION = "html_strip_v1"


def strip_html_preserve_text(input_text: str) -> str:
    """
    Convert HTML to plain text for embedding:
    - Remove tags like <i>, <q>, etc.
    - Unescape HTML entities
    - Collapse excessive whitespace
    - Preserve sentence spacing reasonably
    """
    if not input_text:
        return ""

    text = input_text

    # Fast path if no tags present
    if "<" not in text and "&" not in text:
        return _collapse_whitespace(text)

    if _HAS_BS4:
        try:
            soup = BeautifulSoup(text, "lxml")
            text = soup.get_text(separator=" ")
        except Exception:
            # Fallback to regex if parser fails
            text = _regex_strip_tags(text)
    else:
        text = _regex_strip_tags(text)

    # Decode HTML entities
    text = html.unescape(text)

    return _collapse_whitespace(text)


_TAG_RE = re.compile(r"<[^>]+>")


def _regex_strip_tags(text: str) -> str:
    return _TAG_RE.sub(" ", text)


_WS_RE = re.compile(r"\s+")
_SOFT_HYPHEN_RE = re.compile("\u00ad")


def _collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def strip_soft_hyphens(text: str) -> str:
    """Remove Unicode soft hyphens (U+00AD) from text shown to clients."""
    if not text:
        return text
    return _SOFT_HYPHEN_RE.sub("", text)


