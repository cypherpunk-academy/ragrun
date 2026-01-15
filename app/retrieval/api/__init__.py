"""Routers for retrieval agents."""
from __future__ import annotations

from fastapi import APIRouter

from .philo_von_freisinn import router as philo_router
from .concept_explain_worldviews import router as worldviews_router

router = APIRouter()
router.include_router(philo_router, prefix="/agent/philo-von-freisinn")
router.include_router(worldviews_router, prefix="/agent/philo-von-freisinn")

__all__ = ["router"]


