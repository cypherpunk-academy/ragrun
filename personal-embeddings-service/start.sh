#!/bin/bash

# Personal Embeddings Service - Quick Start Script

echo "🚀 Personal Embeddings Service - Quick Start"
echo "============================================="

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "✅ Docker and Docker Compose found"
    
    echo ""
    echo "🐳 Starting with Docker Compose..."
    echo "This will:"
    echo "  - Build the Docker image"
    echo "  - Download the multilingual-e5-large model (~2.5GB)"
    echo "  - Start the service on http://localhost:8001"
    echo ""
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔨 Building and starting the service..."
        docker-compose up --build
    else
        echo "❌ Cancelled by user"
        exit 1
    fi
    
elif command -v python3 &> /dev/null; then
    echo "✅ Python 3 found"
    echo "⚠️  Docker not found, using local Python installation"
    
    echo ""
    echo "🐍 Starting with local Python..."
    echo "This will:"
    echo "  - Install Python dependencies"
    echo "  - Download the multilingual-e5-large model (~2.5GB)"
    echo "  - Start the service on http://localhost:8001"
    echo ""
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
        
        echo "📥 Pre-downloading model..."
        python scripts/download_model.py --cache-dir ./models
        
        echo "🚀 Starting service..."
        uvicorn app.main:app --host 0.0.0.0 --port 8001
    else
        echo "❌ Cancelled by user"
        exit 1
    fi
    
else
    echo "❌ Neither Docker nor Python 3 found"
    echo "Please install Docker or Python 3.11+ to continue"
    exit 1
fi 