# Local Vector Database Development Setup

This document describes the local development environment for ChromaDB integration.

## Quick Start

1. **Set up environment:**
   ```bash
   # Copy environment configuration
   cp config/local_development.env .env
   
   # Or set environment variables manually
   export VECTOR_DB_TYPE=local
   export LOCAL_VECTOR_DB_PATH=./data/vector_db
   export LOCAL_VECTOR_COLLECTION_NAME=philosophical_768
   ```

2. **Run setup script:**
   ```bash
   python scripts/setup_local_vectordb.py
   ```

3. **Start personal embeddings service:**
   ```bash
   cd personal-embeddings-service
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
   ```

4. **Test the setup:**
   ```bash
   VECTOR_DB_TYPE=local python -c "
   from app.db.local_vector_db import vector_db
   print(f'Vector DB: {type(vector_db).__name__}')
   "
   ```

## Directory Structure

```
data/
├── vector_db/           # ChromaDB persistent storage
├── backups/             # Database backups
├── logs/                # Application logs
└── embeddings_cache/    # Embeddings cache

config/
└── local_development.env # Environment variables template
```

## Configuration Files

- `data/vector_db_config.yaml` - Vector database configuration
- `config/local_development.env` - Environment variables template
- `app/core/config.py` - Application configuration

## Switching Between Local and Pinecone

```bash
# Use local ChromaDB
export VECTOR_DB_TYPE=local

# Use Pinecone (rollback)
export VECTOR_DB_TYPE=pinecone
```

## Troubleshooting

1. **Import errors:** Run `pip install -r requirements.txt`
2. **Permission errors:** Check directory permissions in `data/`
3. **Embeddings service not running:** Start it on port 8001
4. **ChromaDB errors:** Delete `data/vector_db/` and reinitialize

## Development Workflow

1. Make changes to `LocalVectorStoreManager`
2. Test with: `python scripts/setup_local_vectordb.py`
3. Run application with: `VECTOR_DB_TYPE=local python -m uvicorn app.main:app`
