"""Routers for retrieval agents."""
from __future__ import annotations

from fastapi import APIRouter

from .philo_von_freisinn import router as philo_router

router = APIRouter()
router.include_router(philo_router, prefix="/agent/philo-von-freisinn")

__all__ = ["router"]


