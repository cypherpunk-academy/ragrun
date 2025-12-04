#!/usr/bin/env python3
"""
Verification script for Personal Embeddings Service setup.
Checks if all files are in place and configuration is correct.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_directory_structure():
    """Check if all required directories and files exist."""
    print("🔍 Checking directory structure...")
    
    required_files = [
        ("requirements.txt", "Requirements file"),
        ("Dockerfile", "Docker configuration"),
        ("docker-compose.yml", "Docker Compose configuration"),
        ("README.md", "Documentation"),
        ("app/main.py", "Main application"),
        ("app/config.py", "Configuration"),
        ("app/models/embedding_model.py", "Embedding model"),
        ("app/services/embedding_service.py", "Embedding service"),
        ("app/services/batch_service.py", "Batch service"),
        ("app/api/router.py", "API router"),
        ("app/api/endpoints/embeddings.py", "Embeddings endpoints"),
        ("app/api/endpoints/health.py", "Health endpoints"),
        ("scripts/download_model.py", "Model download script"),
        ("scripts/test_embeddings.py", "Test script"),
        ("scripts/benchmark.py", "Benchmark script"),
        ("tests/test_embedding_service.py", "Unit tests"),
        ("start.sh", "Quick start script"),
    ]
    
    all_present = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_present = False
    
    return all_present

def check_python_version():
    """Check Python version compatibility."""
    print("\n🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.11+)")
        return False

def check_docker_availability():
    """Check if Docker is available."""
    print("\n🐳 Checking Docker availability...")
    
    docker_available = os.system("docker --version > /dev/null 2>&1") == 0
    compose_available = os.system("docker-compose --version > /dev/null 2>&1") == 0
    
    if docker_available and compose_available:
        print("✅ Docker and Docker Compose available")
        return True
    elif docker_available:
        print("⚠️  Docker available, but Docker Compose missing")
        return False
    else:
        print("❌ Docker not available")
        return False

def check_configuration():
    """Check configuration files."""
    print("\n⚙️  Checking configuration...")
    
    # Check if config.py has correct settings
    try:
        sys.path.insert(0, 'app')
        from config import settings
        
        print(f"✅ Model name: {settings.model_name}")
        print(f"✅ Batch size: {settings.batch_size}")
        print(f"✅ Max workers: {settings.max_workers}")
        print(f"✅ Cache directory: {settings.cache_dir}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def print_next_steps(docker_available, python_ok):
    """Print next steps based on availability."""
    print("\n" + "="*50)
    print("🚀 NEXT STEPS")
    print("="*50)
    
    if docker_available:
        print("\n🐳 Recommended: Docker Deployment")
        print("   cd personal-embeddings-service")
        print("   ./start.sh")
        print("   # OR manually:")
        print("   docker-compose up --build")
        
    elif python_ok:
        print("\n🐍 Python Local Deployment")
        print("   cd personal-embeddings-service")
        print("   ./start.sh")
        print("   # OR manually:")
        print("   pip install -r requirements.txt")
        print("   python scripts/download_model.py")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8001")
    
    else:
        print("\n❌ Setup Required")
        print("   Please install Docker or Python 3.11+")
    
    print("\n📚 After starting the service:")
    print("   - Service will be available at: http://localhost:8001")
    print("   - API documentation: http://localhost:8001/docs")
    print("   - Health check: http://localhost:8001/api/v1/health")
    print("   - Run tests: python scripts/test_embeddings.py")
    print("   - Run benchmarks: python scripts/benchmark.py")

def main():
    """Main verification function."""
    print("🔍 Personal Embeddings Service - Setup Verification")
    print("="*55)
    
    # Check directory structure
    structure_ok = check_directory_structure()
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check Docker availability
    docker_available = check_docker_availability()
    
    # Check configuration
    config_ok = check_configuration()
    
    # Summary
    print("\n" + "="*50)
    print("📋 VERIFICATION SUMMARY")
    print("="*50)
    
    print(f"Directory structure: {'✅ OK' if structure_ok else '❌ ISSUES'}")
    print(f"Python version: {'✅ OK' if python_ok else '❌ ISSUES'}")
    print(f"Docker availability: {'✅ OK' if docker_available else '❌ NOT AVAILABLE'}")
    print(f"Configuration: {'✅ OK' if config_ok else '❌ ISSUES'}")
    
    if structure_ok and config_ok and (docker_available or python_ok):
        print("\n🎉 Setup verification PASSED!")
        print("✨ Ready to start the Personal Embeddings Service!")
    else:
        print("\n⚠️  Setup verification found issues.")
        print("Please resolve the issues above before starting the service.")
    
    # Print next steps
    print_next_steps(docker_available, python_ok)

if __name__ == "__main__":
    main() 