# Personal RAG Server

## Project Overview

The Personal RAG Server is a specialized Retrieval Augmented Generation system designed for private and efficient knowledge management, with particular focus on philosophical texts in German and English. It provides secure REST API access to interact with your document collection, enhanced with advanced semantic search capabilities.

### Key Features

-   **Local Vector DB (ChromaDB)**: Runs entirely locally by default (no external vector DB)
-   **Advanced Model Selection**: DeepSeek Reasoner as default for philosophical reasoning, with support for multiple LLM providers
-   **Hybrid Search Technology**: Combines dense and sparse vector representations for superior retrieval performance
-   **German Language Optimization**: Specialized embedding models for German philosophical texts
-   **Philosophical Question Detection**: Automatically routes philosophical queries to specialized models
-   **Category-Based Organization**: Organizes documents by philosophical worldviews and categories
-   **Document Management**: Comprehensive tools for uploading, updating, and managing documents
-   **Advanced Metadata**: Enhanced document metadata with philosophical concept tagging
-   **Command-Line Interface**: Unified CLI for all RAG and assistant management operations
-   **Diagnostic Tools**: Comprehensive system diagnostics and query performance analysis
-   **Template System**: Structured philosophical response templates with worldview-specific adaptations

### Core Technologies

-   **Vector Database**: Local ChromaDB (SQLite-backed) for efficient semantic search
-   **Embedding Models**: Specialized models for German philosophical texts
-   **LLM Integration**: DeepSeek Reasoner for advanced philosophical reasoning, with fallback support for other models
    (Legacy: previous Pinecone Assistants integration has been removed from the default runtime)
-   **API Framework**: FastAPI for modern, async Python REST API
-   **Storage**: PostgreSQL for application data and metadata

## System Architecture

The Personal RAG Server follows a modular architecture with clear separation of concerns to ensure maintainability and extensibility.

### Components

1. **API Layer**: FastAPI endpoints handle HTTP requests and provide OpenAPI documentation
2. **Service Layer**: Core business logic components include:

    - **RAG Service**: Orchestrates the RAG pipeline
    - **Assistant Manager**: Manages Pinecone Assistants for philosophical reasoning
    - **Embedding Service**: Generates vector embeddings using specialized models
    - **LLM Service**: Interfaces with DeepSeek for response generation
    - **Vector Store Manager**: Handles Pinecone vector database operations
    - **File Processor**: Processes and chunks documents
    - **Knowledge Base Scanner**: Scans and catalogs documents

3. **Database Layer**:

    - **PostgreSQL**: Stores metadata, conversations, and system data
    - **Pinecone**: Vector database for semantic search with hybrid vector support and assistant integration

4. **CLI Layer**:
    - **RAG CLI**: Command-line interface for system management
    - **Assistant CLI**: Comprehensive assistant management commands
    - **Diagnostic Tools**: Query testing and system analysis tools

### Data Flow

1. **Document Processing**:

    - Documents are uploaded via CLI or API
    - Text is processed, chunked, and enhanced with metadata
    - Both dense and sparse vectors are generated
    - Vectors and metadata are stored in Pinecone

2. **Query Processing**:

    - User query is analyzed for philosophical content
    - Query is converted to dense and sparse vectors
    - Hybrid search retrieves relevant document chunks
    - Retrieved context is passed to LLM with the query
    - LLM generates response based on context and query

3. **Assistant Interaction**:
    - Philosophical assistants maintain conversation context
    - Each assistant applies worldview-specific reasoning
    - DeepSeek Reasoner provides advanced philosophical analysis
    - Template system ensures structured, consistent responses

### Directory Structure

```
app/
├── api/                # REST API endpoints
├── core/               # Core configuration and utilities
├── db/                 # Database connections and operations
├── models/             # Pydantic data models
├── services/           # Business logic services
└── utils/              # Utility functions

assistants/
├── config/             # Assistant configurations by worldview
├── templates/          # Philosophical response templates
└── pinecone_assistant_manager.py  # Assistant management system

scripts/
├── data_import/        # Data import utilities
├── phase2/             # Hybrid search implementation
├── rag-cli/            # CLI tools for RAG and assistant management
└── testing/            # Testing and diagnostic tools
```

## Installation & Setup

### Requirements

-   **Python**: 3.9+ (3.11 recommended)
-   **PostgreSQL**: Running instance (local or remote)
-   **Pinecone**: Account with API key and Assistant API access
-   **DeepSeek**: API key for LLM access (Reasoner model recommended)
-   **Hardware**: Apple Silicon Mac recommended for optimized embedding generation

### Installation Steps

1. Clone the repository:

    ```
    git clone https://github.com/yourusername/personal-rag-server.git
    cd personal-rag-server
    ```

2. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Create a `.env` file with your configuration:

    ```
    python fix_env.py
    ```

4. Edit the `.env` file with your API keys and settings.

### Configuration

Create a `.env` file in the project root with the following settings:

```
# PostgreSQL Settings
RAGRUN_POSTGRES_DSN=postgresql+psycopg://ragrun:ragrun@localhost:15432/ragrun

# Vector DB (Local ChromaDB)
VECTOR_DB_TYPE=local
LOCAL_VECTOR_DB_PATH=./data/vector_db
LOCAL_VECTOR_COLLECTION_NAME=philosophical_768
EMBEDDINGS_DIMENSION=768

# Embedding Service Settings
LOCAL_EMBEDDING_SERVICE_URL=http://localhost:8001
EMBEDDINGS_MODEL=T-Systems-onsite/cross-en-de-roberta-sentence-transformer

# LLM Settings (optional)
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
LLM_PROVIDER=openai
```

(Optional legacy rollback) If you still need Pinecone, you can set `VECTOR_DB_TYPE=pinecone` and provide `PINECONE_*` variables. Note this is not used by default and the runtime is optimized for local ChromaDB.

### Starting the Server

1. Start the local embeddings service (recommended):

```
cd personal-embeddings-service
docker-compose up -d
```

2. Start the FastAPI server:

```
uvicorn app.main:app --reload
```

3. Run the integration tests to verify setup:

    ```
    python -m scripts.testing.integration_tests
    ```

### Docker Compose Files

Use these compose files depending on your environment:

-   `docker-compose.yml`: Base stack for the Personal RAG Server, PostgreSQL, Qdrant, and embeddings service. Suitable for general usage.
-   `docker-compose.override.yml`: Development overrides auto-loaded by Docker Compose. Adds hot-reload, local bind mounts to absolute paths on your machine, and development CORS. Use with:
    -   `docker-compose up` (auto-includes this override).
-   `docker-compose.prod.yml`: Production overrides for hardened settings (no reload, restricted CORS, resource limits, named volumes). Start with:
    -   `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d`

Common commands:

```
# Dev
docker-compose up --build

# Prod
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Core Functionality

### RAG CLI Usage

The RAG CLI provides a unified interface for managing the RAG system and philosophical assistants. All commands are run using the `scripts/rag-cli.sh` script.

```bash
# Get general help
./scripts/rag-cli.sh --help

# Get help for a specific command group
./scripts/rag-cli.sh kb --help
./scripts/rag-cli.sh assistants --help
```

### Document Management

#### Uploading Documents

Upload documents from a directory to the vector database:

```bash
./scripts/rag-cli.sh kb upload --source /path/to/knowledge_base
```

You can specify categories to upload:

```bash
./scripts/rag-cli.sh kb upload --source /path/to/knowledge_base --categories Materialismus,Idealismus
```

#### Updating Documents

Update existing documents in the database:

```bash
./scripts/rag-cli.sh kb update --source /path/to/knowledge_base --categories Materialismus --delete-existing
```

#### Deleting Documents

Delete documents by category:

```bash
./scripts/rag-cli.sh kb delete --categories Materialismus,Idealismus
```

#### Viewing Statistics

Get statistics about the indexed content:

```bash
./scripts/rag-cli.sh kb stats
```

View statistics for specific categories:

```bash
./scripts/rag-cli.sh kb stats --categories Materialismus,Idealismus
```

### Philosophical Assistant Management

#### List Available Models

Show all supported LLM models for philosophical assistants:

```bash
./scripts/rag-cli.sh assistants models
```

This displays available models with DeepSeek Reasoner as the recommended default for philosophical reasoning.

#### Assistant Management

```bash
# List all assistants
./scripts/rag-cli.sh assistants list

# Create a new philosophical assistant (uses deepseek-reasoner by default)
./scripts/rag-cli.sh assistants create my-philosophy-ai Idealismus

# Create with specific model
./scripts/rag-cli.sh assistants create my-claude-ai Materialismus --model claude-3-5-sonnet

# Delete an assistant
./scripts/rag-cli.sh assistants delete my-test-assistant
```

#### Interactive Chat Sessions

```bash
# Single question
./scripts/rag-cli.sh assistants chat my-philosophy-ai "Erkläre bitte den Idealismus"

# Interactive session with history
./scripts/rag-cli.sh assistants chat my-philosophy-ai "Hallo" \
  --interactive \
  --history-file chat_history.json
```

#### Document Management for Assistants

```bash
# Upload documents to an assistant
./scripts/rag-cli.sh assistants add-files my-philosophy-ai \
  books/plato.pdf books/schelling.pdf \
  --worldview Idealismus

# List files uploaded to an assistant
./scripts/rag-cli.sh assistants list-files my-philosophy-ai

# Get context snippets without full responses
./scripts/rag-cli.sh assistants context my-philosophy-ai \
  "Was bedeutet Idealismus?" \
  --top-k 3 \
  --worldview-filter Idealismus
```

### Querying the System

You can query the system through the API or test queries via the CLI:

```bash
./scripts/rag-cli.sh search query "What is the concept of consciousness?" --category Materialismus
```

Search without category filter:

```bash
./scripts/rag-cli.sh search query "Compare consciousness in materialism and idealism"
```

### Testing the System

Run integration tests to verify system functionality:

```bash
python -m scripts.testing.integration_tests --verbose
```

Test a specific query:

```bash
python -m scripts.testing.integration_tests --query "What is the meaning of life?"
```

Test if a question is philosophical:

```bash
python -m scripts.testing.integration_tests --test-philosophical "Does free will exist?"
```

## Advanced Features

### Hybrid Search Capabilities

The system implements hybrid search combining both dense and sparse vector representations:

#### Dense Vectors (Semantic Search)

-   Capture meaning and context regardless of exact word usage
-   Use specialized German embedding models for better understanding of philosophical texts
-   Handle nuanced concepts and relationships between ideas

#### Sparse Vectors (Keyword Search)

-   Capture exact word matches and lexical information
-   Implement BM25 algorithm optimized for German philosophical texts
-   Address variations in number formats (e.g., "12" vs "zwölf")

#### Lexical strip for hybrid sparse (assistant-manifest)

For `generate-prompt` / `execute-prompt`, each assistant can define optional `lexical-strip-hybrid` in **ragkeep** `assistants/<slug>/assistant-manifest.yaml`: a YAML list of phrases removed from the **user prompt only for the sparse (BM25) branch** of hybrid retrieval. Dense embeddings still use the full user prompt. Phrases are applied longest-first; single words use word-boundary matching. Omit the key or use an empty list to disable. If redaction would leave nothing meaningful, the original prompt is kept for sparse.

#### Configurable Hybrid Search

```bash
# Use hybrid search with custom alpha value (dense vs sparse weighting)
./scripts/rag-cli.sh search query "Welches sind die 12 Weltanschauungen?" --alpha 0.6
```

The alpha parameter controls the weight balance between dense and sparse vectors:

-   `alpha=1.0`: 100% dense vectors (pure semantic search)
-   `alpha=0.0`: 100% sparse vectors (pure lexical search)
-   `alpha=0.5`: Equal weight to both (default)

### Philosophical Assistant Platform

The system includes four specialized philosophical assistants, each representing a distinct worldview:

#### Assistant Personas

1. **Idealismus (Aurelian I. Schelling)**:

    - Focuses on the primacy of ideas and spiritual forces
    - Emphasizes creative processes originating in the spiritual world
    - Uses elevated, enthusiastic language reflecting Schelling's philosophical style

2. **Materialismus (Aloys I. Freud)**:

    - Analyzes behavior through material and biological conditions
    - Avoids spiritual terminology, focusing on measurable phenomena
    - Adopts Freudian analytical approach to understanding human behavior

3. **Realismus (Arvid I. Steiner)**:

    - Balances spiritual and material perspectives
    - Emphasizes unity of perception and conceptual understanding
    - Focuses on karma, social development, and anthroposophical insights

4. **Spiritualismus (Amara I. Steiner)**:
    - Emphasizes spiritual hierarchies and angelic beings
    - Focuses on inner development and soul exploration
    - Discusses karma, reincarnation, and spiritual development

#### Model Selection for Philosophical Reasoning

The system uses **DeepSeek Reasoner** as the default model for philosophical assistants due to its:

-   **Exceptional reasoning capabilities** perfect for philosophical work
-   **Advanced chain-of-thought processing** for complex arguments
-   **Superior handling of abstract concepts** and logical reasoning
-   **Excellent German/English multilingual support**
-   **Cost-effective** performance while maintaining highest quality

Alternative models are available for specific use cases:

```bash
# Compare responses across different models
./scripts/rag-cli.sh assistants create materialismus-reasoner Materialismus  # Default: deepseek-reasoner
./scripts/rag-cli.sh assistants create materialismus-claude Materialismus --model claude-3-5-sonnet
./scripts/rag-cli.sh assistants create materialismus-gpt4o Materialismus --model gpt-4o
```

### Philosophical Question Detection

The system automatically detects philosophical questions and routes them to specialized LLM models:

```bash
# Test if a question is philosophical
./scripts/rag-cli.sh diagnostics test-philosophical "What is the meaning of life?"
```

Philosophical detection is based on:

-   Content analysis of the question
-   Presence of philosophical terms and concepts
-   Question structure and complexity

When a philosophical question is detected, the system uses the DeepSeek Reasoner model, which is optimized for philosophical reasoning.

### Template System

The assistant platform includes sophisticated template handling for structured philosophical responses:

#### Available Templates

1. **Gedankenfehler-Formulieren**: For correcting philosophical misconceptions
2. **Gedankenfehler-Aspekte**: For considering specific aspects in corrections
3. **Gedankenfehler-Glossar**: For creating philosophical glossaries
4. **Gedankenfehler-Wiederholen**: For creating variations of philosophical thoughts

Each template is adapted by the assistant to their specific worldview while maintaining the required structure.

### Metadata Enhancement

Documents are automatically enhanced with rich metadata to improve search and filtering:

#### Philosophical Concept Tagging

-   Automatic detection of philosophical concepts mentioned in documents
-   Tagging with standardized terminology
-   Cross-referencing between related concepts

#### Author Attribution

-   Automatic extraction of author information from filenames
-   Linking to author metadata and background information
-   Grouping works by author

#### Number Format Handling

-   Detection and normalization of number formats (digits and spelled-out)
-   Creation of search-optimized fields for both formats
-   Support for German number variations

### Category Management

Organize your document collection by philosophical worldviews and other categories:

```bash
# List all categories
./scripts/rag-cli.sh kb list-categories

# View documents in a category
./scripts/rag-cli.sh kb list-documents --category Idealismus
```

Categories can be nested and documents can belong to multiple categories, enabling flexible organization of complex philosophical texts.

## Troubleshooting

### Common Issues and Solutions

#### Document Not Found in Search Results

If a document isn't appearing in search results:

```bash
# Check if the document exists in the database
./scripts/rag-cli.sh diagnostics check-document --document-id "Rudolf_Steiner#Der_menschliche_und_der_kosmische_Gedanke_Zyklus_33_[GA_151]" --expected-category "Realismus"
```

**Solutions**:

-   Verify the document was uploaded with correct metadata
-   Try searching with alternative query formulations
-   Adjust the alpha parameter for hybrid search
-   Use the improved document check command:

```bash
./scripts/rag-cli.sh diagnostics check-document-improved --document-id "Rudolf_Steiner#Der_menschliche_und_der_kosmische_Gedanke_Zyklus_33_[GA_151]" --expected-category "Realismus" --query-match
```

#### Number Format Issues

If queries with different number formats (e.g., "12" vs "zwölf") yield different results:

**Solutions**:

-   Ensure hybrid search is enabled (alpha between 0.3-0.7)
-   Run the optimization tool to find optimal alpha:

```bash
python -m scripts.phase2.phase2_optimization_tuning --vectorizer bm25_vectorizer.pkl --output-dir optimization_results
```

#### Assistant-Related Issues

**Assistant Creation Fails**:

```bash
# Use dry-run to validate configuration
./scripts/rag-cli.sh assistants create test Idealismus --dry-run
```

**Assistant Chat Timeouts**:

-   Break complex philosophical questions into simpler parts
-   DeepSeek Reasoner handles complexity well but may need time for deep reasoning
-   Check assistant status with `list` command

**Model Selection Issues**:

```bash
# Check available models
./scripts/rag-cli.sh assistants models

# Verify model compatibility
./scripts/rag-cli.sh assistants create test Idealismus --model deepseek-reasoner --dry-run
```

#### Configuration Problems

If you encounter configuration issues:

```bash
# Run the integration tests to verify configuration
python -m scripts.testing.integration_tests
```

**Solutions**:

-   Verify environment variables in .env file
-   Check API keys for Pinecone and DeepSeek
-   Ensure the Pinecone index has the correct dimension (1024)
-   Verify Pinecone Assistant API access

### Diagnostic Tools

The system includes comprehensive diagnostic tools:

#### RAG Diagnosis

Run a full diagnostic on the RAG system:

```bash
./scripts/rag-cli.sh diagnostics diagnose-retrieval
```

This checks:

-   Document existence and retrieval
-   Query performance with variations
-   Category filter effectiveness
-   LLM response quality

#### Performance Analysis

Analyze search performance with different configurations:

```bash
python -m scripts.phase2.TODO_phase2_hybrid_test --vectorizer bm25_vectorizer.pkl --category Realismus_Test --output evaluation_report.md
```

### Performance Optimization Tips

1. **Adjust Hybrid Search Parameters**:

    - For philosophical terminology: alpha = 0.4
    - For general questions: alpha = 0.7
    - For queries with numbers: alpha = 0.5

2. **Optimize Assistant Interactions**:

    - Use DeepSeek Reasoner for complex philosophical discussions
    - Switch to faster models (deepseek-chat) for simple queries
    - Leverage context retrieval for fact-checking without full generation

3. **Optimize Chunk Size**:

    - Smaller chunks (500-800 chars) for precise retrieval
    - Larger chunks (1000-1500 chars) for more context
    - Adjust overlap (200-300 chars) to maintain context

4. **Batch Processing**:
    - Use larger batch sizes for document uploads
    - Process large collections with the parallel option:

```bash
./scripts/rag-cli.sh kb upload --source /path/to/knowledge_base --parallel --workers 4
```

5. **Pinecone Optimization**:
    - Increase `top_k` parameter for recall-focused applications
    - Use namespace-based organization for cleaner separation
    - Implement metadata filtering to narrow results

## Reference

### Command Reference

#### Knowledge Base Management Commands

| Command              | Description                           | Example                                                                         |
| -------------------- | ------------------------------------- | ------------------------------------------------------------------------------- |
| `kb upload`          | Upload documents to vector store      | `./scripts/rag-cli.sh kb upload --source /path/to/docs`                         |
| `kb update`          | Update existing documents             | `./scripts/rag-cli.sh kb update --source /path/to/docs --categories Philosophy` |
| `kb delete`          | Delete documents by category          | `./scripts/rag-cli.sh kb delete --categories Materialismus`                     |
| `kb stats`           | View statistics about indexed content | `./scripts/rag-cli.sh kb stats`                                                 |
| `kb list-categories` | List available categories             | `./scripts/rag-cli.sh kb list-categories`                                       |
| `kb list-documents`  | List documents in a category          | `./scripts/rag-cli.sh kb list-documents --category Idealismus`                  |

#### Philosophical Assistant Commands

| Command                   | Description                            | Example                                                                             |
| ------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------- |
| `assistants models`       | List available LLM models              | `./scripts/rag-cli.sh assistants models`                                            |
| `assistants list`         | List all assistants                    | `./scripts/rag-cli.sh assistants list --output-format json`                         |
| `assistants create`       | Create new philosophical assistant     | `./scripts/rag-cli.sh assistants create my-ai Idealismus --model deepseek-reasoner` |
| `assistants delete`       | Delete an assistant                    | `./scripts/rag-cli.sh assistants delete my-test-assistant`                          |
| `assistants chat`         | Interactive chat with assistant        | `./scripts/rag-cli.sh assistants chat my-ai "Hello" --interactive`                  |
| `assistants list-files`   | List files uploaded to assistant       | `./scripts/rag-cli.sh assistants list-files my-ai`                                  |
| `assistants add-files`    | Upload files to assistant              | `./scripts/rag-cli.sh assistants add-files my-ai file.pdf --worldview Idealismus`   |
| `assistants remove-files` | Remove files from assistant            | `./scripts/rag-cli.sh assistants remove-files my-ai file-id-123`                    |
| `assistants context`      | Get context snippets without responses | `./scripts/rag-cli.sh assistants context my-ai "query" --top-k 5`                   |

#### Search Commands

| Command           | Description                | Example                                                           |
| ----------------- | -------------------------- | ----------------------------------------------------------------- |
| `search query`    | Search for documents       | `./scripts/rag-cli.sh search query "What is consciousness?"`      |
| `search optimize` | Optimize search parameters | `./scripts/rag-cli.sh search optimize --query-type philosophical` |

#### Diagnostic Commands

| Command                          | Description                  | Example                                                                  |
| -------------------------------- | ---------------------------- | ------------------------------------------------------------------------ |
| `diagnostics diagnose-retrieval` | Run full system diagnosis    | `./scripts/rag-cli.sh diagnostics diagnose-retrieval`                    |
| `diagnostics check-document`     | Check document existence     | `./scripts/rag-cli.sh diagnostics check-document --document-id "doc_id"` |
| `diagnostics test-philosophical` | Test philosophical detection | `./scripts/rag-cli.sh diagnostics test-philosophical "What is truth?"`   |

### API Endpoints

#### RAG Endpoints

| Endpoint                      | Method | Description                                  |
| ---------------------------- | ------ | -------------------------------------------- |
| `/api/v1/rag/query`          | POST   | Generate RAG response for a query            |
| `/api/v1/rag/search`         | POST   | Search for documents                         |
| `/api/v1/rag/documents`      | POST   | Add documents to the system                  |
| `/api/v1/rag/retrieve`       | POST   | Retrieve top-k chunks for a prompt           |
| `/api/v1/rag/collections`    | GET    | List vector collections                       |
| `/api/v1/rag/books/titles`   | GET    | List distinct book titles (optionally author) |
| `/api/v1/rag/store-chunks`   | POST   | Persist JSONL to `rag_chunks` (`collection_name` = `rag_partition`: assistant collection or `__shared__`) |
| `/api/v1/rag/embed-chunks`   | POST   | Embed from `rag_chunks` (optional `shared_source_ids`, `source_ids`, **`chunk_types`**) → Qdrant + `vector_chunks` |
| `/api/v1/rag/delete-chunks`  | POST   | Delete chunks by filter or all                |

#### Auth and System

| Endpoint       | Method | Description                 |
| -------------- | ------ | --------------------------- |
| `/`            | GET    | Root status                 |
| `/health`      | GET    | Detailed health check       |
| `/info`        | GET    | Application info            |
| `/api/v1/auth/login`    | POST   | User login                |
| `/api/v1/auth/register` | POST   | User registration         |

> Note: Legacy “assistants” endpoints are no longer part of the default runtime and may be removed entirely in a future revision.

#### Assistants API (OpenAI-compatible + Pinecone Extensions)

| Endpoint                               | Method         | Description                 |
| -------------------------------------- | -------------- | --------------------------- |
| `/api/v1/assistants`                   | GET/POST       | List/create assistants      |
| `/api/v1/assistants/{assistant_id}`    | GET/PUT/DELETE | Get/update/delete assistant |
| `/api/v1/threads`                      | GET/POST       | List/create threads         |
| `/api/v1/threads/{thread_id}/messages` | GET/POST       | List/create messages        |
| `/api/v1/threads/{thread_id}/runs`     | POST           | Run assistant on thread     |

#### System Endpoints

| Endpoint             | Method | Description              |
| -------------------- | ------ | ------------------------ |
| `/api/v1/health`     | GET    | System health check      |
| `/api/v1/auth/token` | POST   | Get authentication token |
| `/docs`              | GET    | OpenAPI documentation    |

### Configuration Options

#### Environment Variables

| Variable                      | Description                     | Default                       |
| ----------------------------- | ------------------------------- | ----------------------------- |
| `RAGRUN_POSTGRES_DSN`         | PostgreSQL DSN                  | `postgresql+psycopg://ragrun:ragrun@localhost:15432/ragrun` |
| `PINECONE_API_KEY`            | Pinecone API key                | -                             |
| `PINECONE_ENVIRONMENT`        | Pinecone environment            | -                             |
| `PINECONE_INDEX_NAME`         | Pinecone index name             | `rag-server-hybrid`           |
| `PINECONE_HOST`               | Pinecone host URL               | -                             |
| `DEEPSEEK_API_KEY`            | DeepSeek API key                | -                             |
| `DEEPSEEK_API_URL`            | DeepSeek API URL                | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL`              | Default DeepSeek model          | `deepseek-chat`               |
| `DEEPSEEK_PHILOSOPHY_MODEL`   | Model for philosophical queries | `deepseek-reasoner`           |
| `LOCAL_EMBEDDING_SERVICE_URL` | Local embedding service URL     | `http://localhost:8001`       |
| `EMBEDDINGS_DIMENSION`        | Embedding vector dimension      | `1024`                        |
| `EMBEDDINGS_MODEL`            | Embedding model name            | `multilingual-e5-large`       |

#### Assistant Model Selection

| Model               | Description                        | Best For                          |
| ------------------- | ---------------------------------- | --------------------------------- |
| `deepseek-reasoner` | Advanced reasoning model (default) | Complex philosophical discussions |
| `deepseek-chat`     | General purpose model              | Quick Q&A, mixed content          |
| `claude-3-5-sonnet` | Anthropic's reasoning model        | Alternative reasoning, analysis   |
| `gpt-4o`            | OpenAI's general model             | General use, fallback             |
| `gpt-4o-mini`       | Lightweight OpenAI model           | Simple queries, testing           |

#### Hybrid Search Parameters

| Parameter       | Description                   | Recommended Values       |
| --------------- | ----------------------------- | ------------------------ |
| `alpha`         | Dense vs sparse weight        | 0.3-0.7 (default: 0.5)   |
| `top_k`         | Number of results to retrieve | 10-30 (default: 15)      |
| `chunk_size`    | Document chunk size           | 500-1500 (default: 1000) |
| `chunk_overlap` | Overlap between chunks        | 100-300 (default: 200)   |

## Workflow Examples

### Complete Setup with Philosophical Assistants

```bash
# 1. Upload philosophical documents
./scripts/rag-cli.sh kb upload --source /path/to/philosophical_texts

# 2. Check available models (deepseek-reasoner will be default)
./scripts/rag-cli.sh assistants models

# 3. Create philosophical assistants for each worldview
./scripts/rag-cli.sh assistants create idealism-reasoner Idealismus
./scripts/rag-cli.sh assistants create materialism-reasoner Materialismus
./scripts/rag-cli.sh assistants create realism-reasoner Realismus
./scripts/rag-cli.sh assistants create spiritualism-reasoner Spiritualismus

# 4. Upload relevant texts to each assistant
./scripts/rag-cli.sh assistants add-files idealism-reasoner \
  texts/plato_republic.pdf texts/schelling_nature.pdf \
  --worldview Idealismus

# 5. Start philosophical discussion
./scripts/rag-cli.sh assistants chat idealism-reasoner \
  "Erkläre den Unterschied zwischen Platons Ideenlehre und Schellings Naturphilosophie" \
  --interactive
```

### Model Comparison for Philosophical Reasoning

```bash
# Create assistants with different models for comparison
./scripts/rag-cli.sh assistants create ethics-reasoner Materialismus  # Uses deepseek-reasoner
./scripts/rag-cli.sh assistants create ethics-claude Materialismus --model claude-3-5-sonnet
./scripts/rag-cli.sh assistants create ethics-gpt4o Materialismus --model gpt-4o

# Test complex philosophical question across models
question="How does materialism explain consciousness and free will in light of quantum mechanics?"
./scripts/rag-cli.sh assistants chat ethics-reasoner "$question"
./scripts/rag-cli.sh assistants chat ethics-claude "$question"
./scripts/rag-cli.sh assistants chat ethics-gpt4o "$question"

# Compare context retrieval capabilities
./scripts/rag-cli.sh assistants context ethics-reasoner \
  "consciousness materialism quantum" --top-k 5
```

This comprehensive Personal RAG Server provides a complete platform for philosophical reasoning, combining advanced retrieval capabilities with sophisticated conversational AI to create an unparalleled tool for philosophical research and discussion.
