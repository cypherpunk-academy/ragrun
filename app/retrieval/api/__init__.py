"""Routers for retrieval agents."""
from __future__ import annotations

from fastapi import APIRouter

from .concept_explain_worldviews import router as worldviews_router
from .authentic_concept_explain import router as authentic_router
from .essay_create import router as essay_create_router
from .essay_finetune import router as essay_finetune_router
from .essay_completion import router as essay_completion_router
from .essay_evaluation import router as essay_evaluation_router
from .translate_to_worldview import router as translate_router

router = APIRouter()
router.include_router(worldviews_router, prefix="/agent/philo-von-freisinn")
router.include_router(authentic_router, prefix="/agent/philo-von-freisinn")
router.include_router(essay_create_router, prefix="/agent/philo-von-freisinn")
router.include_router(essay_finetune_router, prefix="/agent/philo-von-freisinn")
router.include_router(essay_completion_router, prefix="/agent/philo-von-freisinn")
router.include_router(essay_evaluation_router, prefix="/agent/philo-von-freisinn")
router.include_router(translate_router, prefix="/agent/sigrid-von-gleich")

__all__ = ["router"]


