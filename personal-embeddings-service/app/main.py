from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from app.services.embedding_service import embedding_service
from app.api.router import api_router
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Personal Embeddings Service")
    try:
        await embedding_service.load_model()
        logger.info("Embedding service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Personal Embeddings Service")
    # Add any cleanup code here if needed

app = FastAPI(
    title="Personal Embeddings Service",
    description=f"Self-hosted {settings.model_name} embedding service for RAG applications",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "message": "Personal Embeddings Service",
        "model": settings.model_name,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/status")
async def status():
    """Quick status endpoint."""
    return {
        "service": "Personal Embeddings Service",
        "status": "running",
        "model_ready": embedding_service.ready
    } 