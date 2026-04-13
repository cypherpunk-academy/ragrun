# Swagger UI for Personal RAG Server OpenAPI Specifications

This directory contains all files needed to set up and run Swagger UI for the Personal RAG Server and Personal Embeddings Service OpenAPI specifications.

## Quick Start

### Option 1: Using Python (No Docker Required)

1. **Start Swagger UI with Python**:

    ```bash
    cd swagger-specs
    ./serve-swagger.py
    ```

2. **Access Swagger UI**:
   Open your browser and navigate to:

    ```
    http://localhost:8080
    ```

3. **Stop the server** by pressing `Ctrl+C` in the terminal.

### Option 2: Using Docker (If Docker is Available)

1. **Start Swagger UI with Docker**:

    ```bash
    cd swagger-specs
    docker-compose up -d
    ```

2. **Access Swagger UI**:
   Open your browser and navigate to:

    ```
    http://localhost:8080
    ```

3. **Switch between APIs** using the dropdown menu at the top of the page.

## Contents

-   `personal-rag-server-openapi.yaml`: OpenAPI specification for the Personal RAG Server
-   `personal-embeddings-service-openapi.yaml`: OpenAPI specification for the Personal Embeddings Service
-   `docker-compose.yml`: Docker Compose configuration for Swagger UI
-   `update-specs.sh`: Bash script to update specifications and restart Swagger UI
-   `update_specs.py`: Python script to update specifications and restart Swagger UI
-   `serve-swagger.py`: Python script to serve Swagger UI without Docker

## Updating Specifications

When you make changes to your API and need to update the OpenAPI specifications:

### If using Python server:

```bash
# Copy the updated specs to the swagger-specs directory
cp ../personal-rag-server-openapi.yaml ./
cp ../personal-embeddings-service-openapi.yaml ./

# Restart the Python server (stop with Ctrl+C and start again)
./serve-swagger.py
```

### If using Docker:

```bash
cd swagger-specs
./update-specs.sh
```

Or using Python:

```bash
cd swagger-specs
./update_specs.py
```

## Manual Updates

If you're manually updating the OpenAPI specifications:

1. Edit the YAML files directly in this directory
2. If using Docker, restart the container:
    ```bash
    cd swagger-specs
    docker-compose restart swagger-ui
    ```
3. If using Python server, restart the server (stop with Ctrl+C and start again)

## Using Swagger UI for API Testing

1. Expand any endpoint to see details
2. Click "Try it out" to test an endpoint
3. Fill in required parameters
4. Click "Execute" to make the API call

## Authentication

For endpoints that require authentication:

1. Click the "Authorize" button at the top
2. Enter your API key or JWT token
3. Click "Authorize" to apply

## Troubleshooting

-   If Swagger UI can't load your specification, check the YAML syntax
-   For CORS issues, ensure your API allows requests from the Swagger UI domain
-   If Docker container isn't starting, check if Docker is running with `docker info`
-   If the Python server fails, check if you have the required libraries installed
