# LLM Integration Guide

This guide explains how to use the Local LLM integration with Ollama in the Research File Manager.

## Overview

The Research File Manager now includes AI-powered features using locally running Large Language Models (LLMs) via Ollama. These features include:

- **Document Summarization**: Get AI-generated summaries of your files
- **Context-aware Querying**: Ask questions about file content
- **Intelligent File Organization**: Get AI suggestions for organizing files
- **Enhanced Search Results**: Automatic summaries for high-relevance search results

## Key Features

### 🛡️ Privacy-First Design
- **Complete Local Operation**: All AI processing happens on your machine
- **No Cloud Dependencies**: No data sent to external services
- **Graceful Degradation**: System works perfectly without Ollama installed

### ⚡ Performance Optimized
- **Lightweight Model**: Uses llama3.2:3b (3B parameters) for fast responses
- **Smart Caching**: Avoids redundant processing
- **Background Processing**: AI features don't block core functionality

## Installation

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows - Download from https://ollama.com/download
```

### 2. Install Python Dependencies
```bash
pip install ollama>=0.3.0
```

### 3. Pull the Default Model
```bash
ollama pull llama3.2:3b
```

### 4. Start Ollama Service
```bash
# The service usually starts automatically, but you can start it manually:
ollama serve
```

## Configuration

### Environment Variables
```bash
# Enable/disable LLM features
export LLM_ENABLED=true

# Change the model (optional)
export LLM_MODEL_NAME=llama3.2:3b

# Adjust timeout (seconds)
export LLM_TIMEOUT=60

# Adjust context length (characters)
export LLM_MAX_CONTEXT_LENGTH=4000

# Adjust creativity (0.0 = conservative, 1.0 = creative)
export LLM_TEMPERATURE=0.3

# Enable fallback mode when Ollama unavailable
export LLM_FALLBACK_ENABLED=true
```

### Alternative Models
You can use different models by changing the `LLM_MODEL_NAME`:
```bash
# Faster, smaller model (1B parameters)
export LLM_MODEL_NAME=llama3.2:1b

# Larger, more capable model (8B parameters - requires more RAM)
export LLM_MODEL_NAME=llama3.1:8b
```

## API Endpoints

### 1. Document Summarization
**POST** `/api/llm/summarize`

```json
{
  "file_id": "uuid-of-file",
  "content": "optional content override"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "summary": "Brief 2-3 sentence summary",
    "key_points": ["point 1", "point 2"],
    "main_topic": "topic category",
    "confidence_score": 0.85,
    "service_available": true
  },
  "service_available": true
}
```

### 2. Context-aware Querying
**POST** `/api/llm/query`

```json
{
  "query": "What are the main findings?",
  "file_id": "uuid-of-file",
  "content": "optional content override"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "answer": "The main findings show...",
    "confidence_score": 0.9,
    "source_found": true,
    "query": "What are the main findings?",
    "service_available": true
  },
  "service_available": true
}
```

### 3. Organization Suggestions
**POST** `/api/llm/suggest-organization`

```json
{
  "file_name": "research_data.csv",
  "content": "optional file content",
  "file_type": ".csv"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "suggested_folder": "data",
    "reasoning": "This appears to be research data...",
    "confidence_score": 0.8,
    "alternative_folders": ["results", "analysis"],
    "service_available": true
  },
  "service_available": true
}
```

### 4. Service Status
**GET** `/api/llm/status`

Response:
```json
{
  "enabled": true,
  "available": true,
  "model_name": "llama3.2:3b",
  "model_loaded": true,
  "ollama_client_available": true,
  "config": {
    "timeout": 60,
    "max_context_length": 4000,
    "temperature": 0.3,
    "fallback_enabled": true
  },
  "test_response": "OK",
  "last_tested": "2024-01-09T10:30:00"
}
```

## Enhanced Search Results

When LLM features are available, search results automatically include AI-generated summaries for high-relevance results (similarity > 0.7):

```json
{
  "file": {
    "id": "uuid",
    "name": "research_paper.pdf",
    "type": ".pdf",
    "path": "/path/to/file",
    "size": 1024000,
    "created_at": "2024-01-09T10:00:00"
  },
  "snippet": "This study investigates...",
  "score": 0.85,
  "summary": "AI-generated summary of the content",
  "llm_available": true
}
```

## Fallback Behavior

When Ollama is not available, the system provides fallback functionality:

- **Summaries**: Simple extractive summaries with document statistics
- **Queries**: Error message explaining LLM unavailability
- **Organization**: Rule-based classification using file extensions
- **Search**: Normal search results without AI summaries

## Testing the Integration

Run the included test script:
```bash
python3 test_llm_integration.py
```

This will test all LLM functionality and show whether Ollama is properly configured.

## Troubleshooting

### Ollama Not Starting
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Restart Ollama service
ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Pull missing model
ollama pull llama3.2:3b
```

### Performance Issues
- Try a smaller model: `llama3.2:1b`
- Reduce context length: `LLM_MAX_CONTEXT_LENGTH=2000`
- Increase timeout: `LLM_TIMEOUT=120`

### Memory Issues
- Use smaller model: `llama3.2:1b`
- Reduce concurrent requests
- Monitor system resources

## Security Considerations

- All processing happens locally - no data leaves your machine
- Models run in isolated environment
- File content is only processed when explicitly requested
- Summaries and responses are not stored permanently
- Original files are never modified

## Performance Tips

1. **Model Selection**: Start with `llama3.2:3b` for balanced performance
2. **Context Management**: Large files are automatically chunked
3. **Smart Processing**: Summaries only generated for high-relevance search results
4. **Background Processing**: LLM operations don't block the UI
5. **Caching**: Avoid repeated processing of the same content

## Model Recommendations

| Model | Size | RAM Required | Speed | Quality |
|-------|------|-------------|-------|---------|
| llama3.2:1b | 1.3GB | 4GB+ | Fast | Good |
| llama3.2:3b | 2.0GB | 8GB+ | Medium | Better |
| llama3.1:8b | 4.7GB | 16GB+ | Slow | Best |

Choose based on your hardware and performance requirements.