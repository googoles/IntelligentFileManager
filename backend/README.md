# Research File Manager - Backend

This directory contains the backend database foundation for the AI-powered Research File Manager MVP.

## Overview

The backend provides a comprehensive database layer built with SQLAlchemy that supports:

- **Project Management**: Organize research projects with metadata and ontology rules
- **File Indexing**: Track files with content, metadata, and vector embeddings
- **Session Management**: Robust database session handling with context managers
- **Configuration**: Environment-based settings for different deployment scenarios
- **Utilities**: Helper functions for file processing and database operations

## Architecture

### Core Components

1. **`database.py`** - SQLAlchemy models and session management
   - `Project` model with UUID primary keys and ontology support
   - `File` model with content, metadata, and embedding storage
   - `DatabaseManager` for connection pooling and initialization
   - Context managers for automatic session cleanup

2. **`config.py`** - Configuration management
   - Environment-based settings (development/production/testing)
   - File processing limits and supported extensions
   - Project templates and organization rules
   - Logging configuration

3. **`utils.py`** - Utility functions
   - File metadata extraction and hash calculation
   - Text content reading with encoding detection
   - Duplicate file detection
   - Project statistics and cleanup operations

4. **`setup.py`** - Database initialization and setup
   - Automated table creation
   - Sample data generation
   - Installation verification

## Database Schema

### Projects Table
```sql
CREATE TABLE projects (
    id VARCHAR PRIMARY KEY,           -- UUID
    name VARCHAR(255) NOT NULL,       -- Project name
    path TEXT NOT NULL UNIQUE,        -- File system path
    created_at TIMESTAMP NOT NULL,    -- Creation time
    ontology JSON                     -- Project-specific rules
);
```

### Files Table
```sql
CREATE TABLE files (
    id VARCHAR PRIMARY KEY,           -- UUID
    project_id VARCHAR NOT NULL,      -- Foreign key to projects
    path TEXT NOT NULL,               -- File system path
    name VARCHAR(255) NOT NULL,       -- File name
    type VARCHAR(50) NOT NULL,        -- File extension
    content TEXT,                     -- Extracted text content
    metadata JSON,                    -- File metadata (size, hash, etc.)
    embedding TEXT,                   -- JSON-serialized vector embedding
    created_at TIMESTAMP NOT NULL     -- Creation time
);
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Database

```bash
python backend/setup.py
```

### 3. Basic Usage

```python
from backend import init_database, db_session, create_project, create_file

# Initialize database
init_database()

# Create a project
with db_session() as session:
    project = create_project(
        session,
        name="My Research",
        path="/path/to/research",
        ontology={"categories": ["data", "results", "papers"]}
    )
    
    # Add a file
    file_record = create_file(
        session,
        project_id=project.id,
        path="/path/to/research/paper.pdf",
        name="paper.pdf",
        file_type=".pdf",
        content="Extracted text content...",
        metadata={"size": 1024000, "hash": "abc123..."}
    )
```

## Features

### Database Models

- **UUID Primary Keys**: All models use UUID for distributed system compatibility
- **Proper Relationships**: Foreign key constraints with cascade deletes
- **JSON Fields**: Flexible metadata and ontology storage
- **Indexes**: Optimized for common query patterns
- **Type Hints**: Full typing support for better IDE integration

### Session Management

- **Context Managers**: Automatic session cleanup with `db_session()`
- **Connection Pooling**: Configurable pool sizes for different environments
- **Error Handling**: Proper rollback on exceptions
- **Transaction Management**: Automatic commit/rollback handling

### Configuration

- **Environment-Based**: Different settings for dev/prod/test
- **File Processing**: Configurable limits and supported extensions
- **Project Templates**: Predefined folder structures
- **Organization Rules**: Automatic file categorization rules

### Utilities

- **File Processing**: Content extraction with encoding detection
- **Metadata Extraction**: File size, hash, timestamps
- **Duplicate Detection**: Hash-based duplicate file finding
- **Statistics**: Comprehensive project analytics
- **Cleanup**: Automated removal of orphaned records

## Configuration Options

### Environment Variables

```bash
# Database
DATABASE_URL=sqlite:///data/db/research.db  # or PostgreSQL URL
ENVIRONMENT=development                      # development/production/testing

# File Processing  
MAX_FILE_SIZE=52428800                      # 50MB default
MAX_CONTENT_LENGTH=10000                    # Characters to extract
CHUNK_SIZE=500                              # Text chunking size

# Logging
LOG_LEVEL=INFO                              # DEBUG/INFO/WARNING/ERROR
LOG_FILE=logs/backend.log                   # Optional log file

# AI/ML
EMBEDDING_MODEL=all-MiniLM-L6-v2           # Sentence transformer model
CHROMA_DB_PATH=data/db/chroma              # Vector database path
```

### Supported File Types

- **Text Files**: `.txt`, `.md`, `.py`, `.js`, `.html`, `.css`, `.xml`, `.json`, etc.
- **Documents**: `.pdf`, `.doc`, `.docx`, `.odt`, `.rtf`
- **Data Files**: `.csv`, `.xlsx`, `.json`, `.xml`, `.parquet`
- **Images**: `.png`, `.jpg`, `.gif`, `.svg`, etc.

### Project Templates

- **Research**: `literature/`, `data/raw/`, `data/processed/`, `code/`, `results/`, `drafts/`, `notes/`
- **Analysis**: `data/`, `scripts/`, `outputs/`, `reports/`, `figures/`
- **Minimal**: `input/`, `output/`, `workspace/`

## Error Handling

The backend includes comprehensive error handling:

- **Database Errors**: Connection failures, constraint violations
- **File System Errors**: Permission issues, missing files
- **Configuration Errors**: Invalid settings, missing directories
- **Data Validation**: Schema validation, type checking

## Testing

Run the setup script to verify installation:

```bash
python backend/setup.py
```

This will:
1. Create required directories
2. Initialize database tables
3. Create sample data
4. Verify all components are working

## Integration with Frontend

The backend is designed to work with FastAPI for the web interface:

```python
from fastapi import FastAPI
from backend import init_database, db_session, Project, File

app = FastAPI()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()

@app.get("/projects")
async def list_projects():
    with db_session() as session:
        projects = session.query(Project).all()
        return [p.to_dict() for p in projects]
```

## Next Steps

After setting up the database foundation:

1. **File Watcher**: Implement automatic file monitoring (`file_watcher.py`)
2. **Semantic Search**: Add vector embedding and search (`search.py`)  
3. **File Organizer**: Create rule-based file organization (`organizer.py`)
4. **Web API**: Build FastAPI endpoints (`main.py`)
5. **Frontend**: Create React-based web interface

## Requirements

- Python 3.8+
- SQLAlchemy 2.0+
- FastAPI (for web API)
- Sentence Transformers (for embeddings)
- ChromaDB (for vector search)

See `requirements.txt` for complete dependency list.

## License

This project is part of the Research File Manager MVP and follows the same license terms.