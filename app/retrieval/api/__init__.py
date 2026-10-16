"""Routers for retrieval agents."""
from __future__ import annotations

from fastapi import APIRouter

from .authentic_concept_explain import router as authentic_router
from .typology_explain import router as typology_explain_router
from .thought_explain import router as thought_explain_router

router = APIRouter()
router.include_router(authentic_router, prefix="/agent/philo-von-freisinn")
router.include_router(typology_explain_router, prefix="/agent/philo-von-freisinn")
router.include_router(thought_explain_router, prefix="/agent/philo-von-freisinn")

__all__ = ["router"]
