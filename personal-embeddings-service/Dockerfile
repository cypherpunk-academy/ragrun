FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create model cache directory
RUN mkdir -p /app/models

# Pre-download the T-Systems model
ENV MODEL_NAME="T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
ENV TRANSFORMERS_CACHE="/app/models"
ENV HF_HOME="/app/models"

# Pre-download model (optional - can also download at runtime)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer', cache_folder='/app/models')"

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"] 