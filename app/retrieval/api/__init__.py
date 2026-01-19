"""Routers for retrieval agents."""
from __future__ import annotations

from fastapi import APIRouter

from .concept_explain_worldviews import router as worldviews_router
from .authentic_concept_explain import router as authentic_router

router = APIRouter()
router.include_router(worldviews_router, prefix="/agent/philo-von-freisinn")
router.include_router(authentic_router, prefix="/agent/philo-von-freisinn")

__all__ = ["router"]


