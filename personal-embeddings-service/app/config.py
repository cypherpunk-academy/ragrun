from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Model settings - use environment variables
    model_name: str = os.environ.get("EMBEDDINGS_MODEL", "T-Systems-onsite/cross-en-de-roberta-sentence-transformer")
    max_seq_length: int = 512
    batch_size: int = 32
    embedding_dimension: int = int(os.environ.get("EMBEDDINGS_DIMENSION", "768"))

    # Performance settings
    use_half_precision: bool = True  # Use float16 on GPU
    max_workers: int = 4
    cache_dir: str = "/app/models"

    # Service settings
    host: str = "0.0.0.0"
    port: int = 8001

    class Config:
        env_file = ".env"

settings = Settings() 