
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
