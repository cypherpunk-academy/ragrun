#!/bin/bash
# Setup development environment and run tests

# Make script exit on first error
set -e

# Print commands before executing them
set -x

# Go to project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "Setting up development environment in $PROJECT_ROOT"

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS or Linux
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    echo "Unsupported OS type: $OSTYPE"
    exit 1
fi

# Upgrade pip
pip install --upgrade pip

# Install the package in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run the tests
echo "Running tests..."
python -m pytest tests/assistants/test_template_processor.py -v
python -m pytest tests/assistants/test_pinecone_integration.py -v

echo "Setup completed successfully!" 