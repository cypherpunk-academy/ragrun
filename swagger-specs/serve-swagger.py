#!/usr/bin/env python3
"""
Simple HTTP server to serve Swagger UI without Docker.
This script downloads Swagger UI and serves it with your OpenAPI specifications.
"""

import os
import sys
import shutil
import http.server
import socketserver
import urllib.request
import zipfile
import tempfile
import json
from pathlib import Path

# Configuration
PORT = 8080
SWAGGER_UI_VERSION = "5.9.0"  # Latest stable version as of now
SWAGGER_UI_DOWNLOAD_URL = f"https://github.com/swagger-api/swagger-ui/archive/v{SWAGGER_UI_VERSION}.zip"
SWAGGER_DIST_DIR = "swagger-ui-dist"

def download_swagger_ui():
    """Download and extract Swagger UI."""
    if os.path.exists(SWAGGER_DIST_DIR):
        print(f"Swagger UI already exists at {SWAGGER_DIST_DIR}")
        return
    
    print(f"Downloading Swagger UI v{SWAGGER_UI_VERSION}...")
    
    # Create a temporary file to download the zip
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
        temp_path = temp_file.name
        
    try:
        # Download the zip file
        urllib.request.urlretrieve(SWAGGER_UI_DOWNLOAD_URL, temp_path)
        
        # Extract the zip file
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            # Extract to a temporary directory
            extract_dir = tempfile.mkdtemp()
            zip_ref.extractall(extract_dir)
            
            # Copy the dist directory to the current directory
            swagger_dir = os.path.join(extract_dir, f"swagger-ui-{SWAGGER_UI_VERSION}")
            dist_dir = os.path.join(swagger_dir, "dist")
            
            if os.path.exists(dist_dir):
                shutil.copytree(dist_dir, SWAGGER_DIST_DIR)
                print(f"Swagger UI extracted to {SWAGGER_DIST_DIR}")
            else:
                print(f"Error: Could not find dist directory in {swagger_dir}")
                sys.exit(1)
    except Exception as e:
        print(f"Error downloading or extracting Swagger UI: {e}")
        sys.exit(1)
    finally:
        # Clean up
        os.unlink(temp_path)
        shutil.rmtree(extract_dir, ignore_errors=True)

def create_swagger_initializer():
    """Create a custom swagger-initializer.js file."""
    script_content = """
window.onload = function() {
  // Begin Swagger UI call region
  const ui = SwaggerUIBundle({
    urls: [
      {
        name: "Personal RAG Server API",
        url: "personal-rag-server-openapi.yaml"
      },
      {
        name: "Personal Embeddings Service API",
        url: "personal-embeddings-service-openapi.yaml"
      }
    ],
    dom_id: '#swagger-ui',
    deepLinking: true,
    displayOperationId: true,
    defaultModelsExpandDepth: 1,
    defaultModelExpandDepth: 1,
    defaultModelRendering: 'example',
    displayRequestDuration: true,
    docExpansion: 'list',
    filter: true,
    showExtensions: true,
    showCommonExtensions: true,
    presets: [
      SwaggerUIBundle.presets.apis,
      SwaggerUIStandalonePreset
    ],
    plugins: [
      SwaggerUIBundle.plugins.DownloadUrl
    ],
    layout: "StandaloneLayout"
  });
  window.ui = ui;
};
"""
    
    initializer_path = os.path.join(SWAGGER_DIST_DIR, "swagger-initializer.js")
    with open(initializer_path, 'w') as f:
        f.write(script_content)
    
    print("Created custom swagger-initializer.js")

def create_index_html():
    """Create a custom index.html file."""
    html_content = """<!-- HTML for static distribution bundle build -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Swagger UI - Personal RAG Server APIs</title>
    <link rel="stylesheet" type="text/css" href="./swagger-ui.css" />
    <link rel="stylesheet" type="text/css" href="index.css" />
    <link rel="icon" type="image/png" href="./favicon-32x32.png" sizes="32x32" />
    <link rel="icon" type="image/png" href="./favicon-16x16.png" sizes="16x16" />
  </head>

  <body>
    <div id="swagger-ui"></div>
    <script src="./swagger-ui-bundle.js" charset="UTF-8"> </script>
    <script src="./swagger-ui-standalone-preset.js" charset="UTF-8"> </script>
    <script src="./swagger-initializer.js" charset="UTF-8"> </script>
  </body>
</html>
"""
    
    index_path = os.path.join(SWAGGER_DIST_DIR, "index.html")
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print("Created custom index.html")

def copy_openapi_specs():
    """Copy OpenAPI specifications to the Swagger UI directory."""
    current_dir = Path.cwd()
    
    specs = [
        "personal-rag-server-openapi.yaml",
        "personal-embeddings-service-openapi.yaml"
    ]
    
    for spec in specs:
        source = current_dir / spec
        target = current_dir / SWAGGER_DIST_DIR / spec
        
        if source.exists():
            shutil.copy2(source, target)
            print(f"Copied {spec} to {SWAGGER_DIST_DIR}")
        else:
            print(f"Warning: {spec} not found")

def start_server():
    """Start the HTTP server."""
    os.chdir(SWAGGER_DIST_DIR)
    
    handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving Swagger UI at http://localhost:{PORT}")
        print("Both APIs should be available in the dropdown menu at the top")
        print("Press Ctrl+C to stop the server")
        httpd.serve_forever()

def main():
    download_swagger_ui()
    create_swagger_initializer()
    create_index_html()
    copy_openapi_specs()
    start_server()

if __name__ == "__main__":
    main() 