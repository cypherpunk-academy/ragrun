# Personal Embeddings Service

A self-hosted embedding service using T-Systems-onsite/cross-en-de-roberta-sentence-transformer model for high-quality German/English text embeddings. Designed specifically for RAG (Retrieval-Augmented Generation) applications with zero ongoing API costs, optimized for German philosophical content.

## 🚀 Features

-   **Self-hosted**: No external API dependencies, complete privacy control
-   **German/English-optimized embeddings**: Uses `T-Systems-onsite/cross-en-de-roberta-sentence-transformer` (768 dimensions)
-   **Fast and efficient**: Optimized for both single requests and batch processing
-   **Deterministic pooling**: Explicit mean-pooling pipeline avoids auto-fallback warnings
-   **German number handling**: Improved handling of different number formats (e.g., "12" vs "zwölf")
-   **Cross-lingual capabilities**: Handles both German and English with strong semantic understanding
-   **GPU support**: Automatic GPU detection and utilization
-   **RESTful API**: Easy integration with existing applications
-   **Comprehensive testing**: Built-in testing and benchmarking tools
-   **Production ready**: Docker containerization and health checks

## 🏗️ Architecture

```
personal-embeddings-service/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   └── embedding_model.py    # Model wrapper
│   ├── services/
│   │   ├── embedding_service.py  # Main service logic
│   │   └── batch_service.py      # Batch processing
│   └── api/
│       ├── router.py        # API router
│       └── endpoints/
│           ├── embeddings.py     # Embedding endpoints
│           └── health.py         # Health check endpoints
├── tests/
│   └── test_embedding_service.py
├── scripts/
│   ├── download_new_model.py    # Pre-download model
│   ├── test_model_comparison.py # Performance testing
│   └── benchmark.py         # Performance benchmarking
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 📋 Requirements

### Minimum Requirements

-   CPU: 4+ cores
-   RAM: 8GB
-   Storage: 10GB free space
-   Python: 3.11+

### Recommended for Best Performance

-   GPU: RTX 3060 or better (8GB+ VRAM)
-   CPU: 8+ cores
-   RAM: 16GB
-   SSD storage

## 🛠️ Installation

### Option 1: Docker (Recommended)

1. **Clone and navigate to the service directory:**

    ```bash
    cd personal-embeddings-service
    ```

2. **Build and run with Docker Compose:**

    ```bash
    docker-compose up --build
    ```

3. **For CPU-only deployment:**
    ```bash
    # Remove GPU configuration from docker-compose.yml
    docker-compose up --build
    ```

### Option 2: Local Installation

1. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Pre-download the model (optional but recommended):**

    ```bash
    python scripts/download_new_model.py --cache-dir ./models
    ```

3. **Run the service:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8001
    ```

## 📡 API Endpoints

### Base URL: `http://localhost:8001`

### Health Check

```bash
GET /api/v1/health
```

### Single/Batch Embeddings

```bash
POST /api/v1/embeddings
{
  "texts": "Your text here" | ["text1", "text2", ...]
}
```

### Large Batch Processing

```bash
POST /api/v1/embeddings/batch
{
  "texts": ["text1", "text2", ...],
  "chunk_size": 32
}
```

### Similarity Search

```bash
POST /api/v1/search
{
  "query": "search query",
  "documents": ["doc1", "doc2", ...],
  "top_k": 5
}
```

### Service Information

```bash
GET /api/v1/info
```

## 🧪 Testing

### Run Model Comparison Tests

```bash
python scripts/test_model_comparison.py
```

### With Performance Benchmarking

```bash
python scripts/test_embeddings.py --benchmark --benchmark-size 100
```

### Dedicated Benchmarking

```bash
python scripts/benchmark.py --output results.json --report report.txt
```

## 📊 Performance Examples

Based on testing with RTX 3070:

| Operation         | Batch Size | Processing Time | Throughput    |
| ----------------- | ---------- | --------------- | ------------- |
| Single Embedding  | 1          | ~50ms           | 20 texts/sec  |
| Small Batch       | 10         | ~80ms           | 125 texts/sec |
| Large Batch       | 100        | ~600ms          | 167 texts/sec |
| Similarity Search | 50 docs    | ~150ms          | -             |

## 🔧 Configuration

### Environment Variables

```bash
# Model settings
MODEL_NAME=T-Systems-onsite/cross-en-de-roberta-sentence-transformer
MAX_SEQ_LENGTH=512
BATCH_SIZE=32

# Performance settings
USE_HALF_PRECISION=true
MAX_WORKERS=4
CACHE_DIR=/app/models

# Service settings
HOST=0.0.0.0
PORT=8001
```

### Custom Configuration

Edit `app/config.py` to customize settings:

```python
class Settings(BaseSettings):
    model_name: str = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    batch_size: int = 32
    max_workers: int = 4
    # ... other settings
```

## 🐳 Docker Deployment

### Basic Deployment

```bash
docker-compose up -d
```

### With Custom Configuration

```bash
# Create .env file
echo "BATCH_SIZE=64" > .env
echo "MAX_WORKERS=8" >> .env

docker-compose up -d
```

### Health Check

```bash
curl http://localhost:8001/api/v1/health
```

## 🔍 Usage Examples

### Python Client Example

```python
import httpx

async def get_embeddings(texts):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/api/v1/embeddings",
            json={"texts": texts}
        )
        return response.json()

# Single text
result = await get_embeddings("Hello world")
embedding = result["embeddings"][0]

# Multiple texts
result = await get_embeddings(["Text 1", "Text 2", "Text 3"])
embeddings = result["embeddings"]
```

### curl Examples

```bash
# Single embedding
curl -X POST "http://localhost:8001/api/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{"texts": "Hello world"}'

# Batch embeddings
curl -X POST "http://localhost:8001/api/v1/embeddings" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'

# Similarity search
curl -X POST "http://localhost:8001/api/v1/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "machine learning",
       "documents": [
         "Deep learning is powerful",
         "Traditional ML methods",
         "Computer vision applications"
       ],
       "top_k": 2
     }'
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Download Fails**

    ```bash
    # Pre-download manually
    python scripts/download_new_model.py
    ```

2. **GPU Not Detected**

    ```bash
    # Check CUDA availability
    python -c "import torch; print(torch.cuda.is_available())"
    ```

3. **Out of Memory Errors**

    - Reduce `batch_size` in config
    - Use CPU-only mode
    - Increase system RAM

4. **Slow Performance**
    - Ensure GPU is being used
    - Check system resources
    - Optimize batch sizes

### Logs

```bash
# Docker logs
docker-compose logs -f embeddings-service

# Local logs
tail -f logs/embeddings-service.log
```

## 🔮 Integration with RAG Server

To integrate with your main RAG server, update the configuration:

```python
# In your RAG server configuration
LOCAL_EMBEDDING_SERVICE_URL = "http://localhost:8001"
EMBEDDINGS_DIMENSION = 768
EMBEDDINGS_MODEL = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
```

## 📈 Monitoring

### Health Endpoints

-   `/api/v1/health` - Detailed health check
-   `/api/v1/health/simple` - Simple status
-   `/api/v1/ready` - Readiness check
-   `/status` - Quick status

### Prometheus `/metrics`

`GET /metrics` is unauthenticated. Prometheus scrapes it on the Docker network at `embedding-service:8001/metrics`. The body never contains request text, vectors, or user data. `/metrics` and `/api/v1/health/simple` are excluded from HTTP instrumentation so scrapes and health polls do not inflate those series. The process runs as a single uvicorn worker, so the counters stay in that process.

`prometheus-fastapi-instrumentator` adds the usual HTTP series: request count, status codes, and duration per route.

Custom series use the prefix `embedder_`:

-   `embedder_requests_total` — embedding requests, labels `endpoint` and `status`.
-   `embedder_texts_per_request` — histogram of how many texts one request submitted.
-   `embedder_encode_seconds` — histogram of encode time only, not the HTTP round trip. Buckets run from 0.05 s to 10 s.
-   `embedder_model_loaded` — gauge, 1 when that model is loaded and 0 otherwise. Label `model`.
-   `embedder_model_load_seconds` — gauge, seconds the last model load took.

`batch_service` has no queue, so there is no `embedder_batch_queue_size`.

### Metrics to Monitor

-   Response times
-   Throughput (texts/second)
-   Memory usage
-   GPU utilization
-   Error rates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

-   Microsoft for the multilingual-e5-large model
-   Sentence Transformers library
-   FastAPI framework
-   All contributors and testers
