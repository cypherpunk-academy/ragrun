from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint with detailed service status.
    Returns comprehensive health information including model status.
    """
    try:
        health_result = await embedding_service.health_check()
        return health_result
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/health/simple")
async def simple_health_check() -> Dict[str, str]:
    """
    Simple health check endpoint that returns basic status.
    Useful for load balancers and simple monitoring.
    """
    try:
        if embedding_service.ready:
            return {"status": "healthy"}
        else:
            return {"status": "loading"}
    except Exception as e:
        logger.error(f"Simple health check failed: {str(e)}")
        return {"status": "unhealthy"}

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint to verify if the service is ready to accept requests.
    """
    try:
        if not embedding_service.ready:
            raise HTTPException(
                status_code=503,
                detail="Service not ready - model still loading"
            )
        
        # Perform a quick test to ensure everything is working
        health_result = await embedding_service.health_check()
        
        if health_result.get("status") == "healthy":
            return {
                "status": "ready",
                "model": health_result.get("model"),
                "device": health_result.get("device")
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Service not ready - health check failed"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        ) 