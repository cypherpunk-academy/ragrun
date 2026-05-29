"""LLM pricing service: loads config/pricing.json, refreshes EUR/USD at startup,
calculates cost per LLM call.

Exchange rate source: https://api.frankfurter.dev/v1 (no API key required).
Model prices are maintained manually in config/pricing.json.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_PRICING_PATH = Path(__file__).parent.parent.parent / "config" / "pricing.json"
_FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest?from=USD&to=EUR"

# Module-level singleton – populated once at startup via update_pricing()
_pricing: dict[str, Any] | None = None


def _load_from_disk() -> dict[str, Any]:
    try:
        return json.loads(_PRICING_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("config/pricing.json not found or unreadable; using empty config")
        return {"models": {}, "eur_per_usd": 0.92}


def get_pricing() -> dict[str, Any]:
    """Return the in-memory pricing config (loaded at startup)."""
    global _pricing
    if _pricing is None:
        _pricing = _load_from_disk()
    return _pricing


async def update_pricing() -> None:
    """Fetch current EUR/USD rate from frankfurter.app and persist to pricing.json.

    Called once during FastAPI lifespan startup. Failures are logged but never raised.
    """
    global _pricing
    _pricing = _load_from_disk()

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            resp = await client.get(_FRANKFURTER_URL)
            resp.raise_for_status()
            data = resp.json()
            eur_rate = float(data["rates"]["EUR"])
    except Exception as exc:
        logger.warning("Could not fetch EUR/USD rate: %s – using cached value %.4f", exc, _pricing.get("eur_per_usd", 0.92))
        return

    _pricing["eur_per_usd"] = eur_rate
    _pricing["eur_per_usd_updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        _PRICING_PATH.write_text(json.dumps(_pricing, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Pricing updated: EUR/USD=%.4f", eur_rate)
    except Exception as exc:
        logger.warning("Could not write pricing.json: %s", exc)


def calculate_cost(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> dict[str, float]:
    """Return {"cost_usd": float, "cost_eur": float} for a single LLM call.

    Returns zeros if pricing config is missing or tokens are None.
    """
    pricing = get_pricing()
    pt = prompt_tokens or 0
    ct = completion_tokens or 0
    if not pt and not ct:
        return {"cost_usd": 0.0, "cost_eur": 0.0}

    models = pricing.get("models", {})
    mp = models.get(model) or models.get("deepseek-chat")
    if not mp:
        return {"cost_usd": 0.0, "cost_eur": 0.0}

    usd = (pt / 1_000_000) * float(mp["input_per_1m"]) + (ct / 1_000_000) * float(mp["output_per_1m"])
    eur = usd * float(pricing.get("eur_per_usd", 0.92))
    return {
        "cost_usd": round(usd, 6),
        "cost_eur": round(eur, 6),
    }
