"""
Modal deployment for personal-embeddings-service.

Deploy:
    modal deploy personal-embeddings-service/modal_app.py

First-time model download (fills the Volume cache):
    modal run personal-embeddings-service/modal_app.py::download_models

The ASGI endpoint URL printed after deploy goes into:
    RAGRUN_EMBEDDINGS_BASE_URL (ragrun Railway env / .env.staging / .env.production)
"""

import os
import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "sentence-transformers==2.7.0",
        "torch==2.2.2",          # pinned: smaller wheel, CUDA 12.1 compatible
        "numpy==1.24.3",
        "pydantic==2.5.0",
        "httpx==0.25.2",
        "pydantic-settings==2.1.0",
        "tenacity==8.2.3",
        "huggingface-hub>=0.19.0,<1.0.0",
        "transformers==4.56.2",
        "tokenizers>=0.22,<0.24",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(
        "personal-embeddings-service/app",
        remote_path="/service/app",
    )
)

# ---------------------------------------------------------------------------
# App + Volume
# ---------------------------------------------------------------------------

app = modal.App(
    "personal-embeddings-service",
    image=image,
)

# Persistent volume — model weights survive container restarts and scale-out
model_volume = modal.Volume.from_name(
    "embeddings-model-cache",
    create_if_missing=True,
)

VOLUME_MOUNT = "/modal-models"

# Models to pre-download (space-separated list mirrors docker-compose ALLOWED_MODELS)
MODELS = [
    "intfloat/multilingual-e5-large",          # 1024-dim — production default
    "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",  # 768-dim
]
DEFAULT_MODEL = "intfloat/multilingual-e5-large"

# ---------------------------------------------------------------------------
# Helper: download models into the Volume (run once)
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    volumes={VOLUME_MOUNT: model_volume},
    timeout=1800,
)
def download_models():
    """Pre-populate the Volume with model weights. Run once after deploy."""
    from sentence_transformers import models as st_models, SentenceTransformer

    for model_name in MODELS:
        print(f"Downloading {model_name} …")
        transformer = st_models.Transformer(
            model_name,
            cache_dir=VOLUME_MOUNT,
        )
        pooling = st_models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        SentenceTransformer(modules=[transformer, pooling])
        print(f"  ✓ {model_name}")

    model_volume.commit()
    print("All models downloaded and committed to Volume.")


# ---------------------------------------------------------------------------
# ASGI app (FastAPI served via Modal)
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    volumes={VOLUME_MOUNT: model_volume},
    # Keep one container warm to avoid cold-start latency
    min_containers=1,
    # Scale-out limit — embeddings are stateless, safe to scale
    max_containers=3,
    # Longer timeout for batch encode requests
    timeout=120,
)
@modal.asgi_app()
def api():
    import sys
    sys.path.insert(0, "/service")

    os.environ.setdefault("EMBEDDINGS_MODEL_NAME", DEFAULT_MODEL)
    os.environ.setdefault("EMBEDDINGS_CACHE_DIR", VOLUME_MOUNT)
    import json
    os.environ.setdefault(
        "EMBEDDINGS_ALLOWED_MODELS",
        json.dumps(MODELS),
    )
    os.environ.setdefault("EMBEDDINGS_USE_HALF_PRECISION", "true")

    from app.main import app as fastapi_app
    return fastapi_app
