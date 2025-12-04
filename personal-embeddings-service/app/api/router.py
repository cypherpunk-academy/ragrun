from fastapi import APIRouter
from app.api.endpoints import embeddings, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    embeddings.router,
    tags=["embeddings"]
)

api_router.include_router(
    health.router,
    tags=["health"]
) 