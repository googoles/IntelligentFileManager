# Research File Manager MVP

**AI-Powered File Organization & Semantic Search for Researchers**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 🎯 Overview

The Research File Manager is an intelligent file management system designed for researchers, academics, and data scientists. It combines **AI-powered semantic search** with **automatic file organization** and **OCR capabilities** to revolutionize how you manage your research projects.

### ✨ Key Features

- 🧠 **AI-Powered Intelligence**: Local LLM integration (Ollama) for document summarization and smart queries
- 👁️ **OCR Text Extraction**: Extract text from images and scanned PDFs automatically
- 🔍 **Semantic Search**: Find files using natural language queries with AI embeddings
- 🗂️ **Smart Organization**: Automatic file categorization based on content and naming patterns
- 📁 **Project Templates**: Pre-built structures for research, data science, and academic workflows
- 🔄 **Real-time Monitoring**: Automatic indexing and processing of new files
- 🌑 **Professional UI**: Modern black and white theme with excellent UX
- 🏠 **Privacy-First**: Complete local operation with no cloud dependencies

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**For Windows:**
```batch
# Clone the repository
git clone <repository-url>
cd IntelligentFileManager

# Run automated setup
python setup_mvp.py

# Start the application
run_mvp_windows.bat

# Open browser to http://localhost:8000
```

**For Linux/Mac:**
```bash
# Clone the repository  
git clone <repository-url>
cd IntelligentFileManager

# Run automated setup
python setup_mvp.py

# Start the application
./run_mvp.sh

# Open browser to http://localhost:8000
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install AI/OCR dependencies (optional)
pip install ollama>=0.3.0 easyocr==1.7.1 pdf2image==1.3.2

# 4. Create directories
mkdir -p data/projects data/db logs

# 5. Start the server
python backend/main.py
```

## 🤖 AI Features Setup (Optional)

### LLM Integration (Ollama)
```bash
# Install Ollama (visit https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model
ollama pull llama3.2:3b

# The application will automatically detect and use Ollama
```

### OCR Capabilities
OCR is automatically enabled when EasyOCR is installed. The system will:
- Process images (.png, .jpg, .gif, .bmp) automatically
- Extract text from scanned PDFs
- Make extracted text searchable
- Show confidence scores for OCR results

## 🏗️ System Architecture

### Backend Components
- **FastAPI Application** - REST API server with AI/OCR endpoints
- **Database Layer** - SQLAlchemy with SQLite (PostgreSQL ready)
- **File Monitor** - Real-time file system monitoring with automatic processing
- **Smart Organizer** - AI-powered file classification and organization
- **Semantic Search** - Vector embeddings with ChromaDB for similarity search
- **LLM Service** - Local Ollama integration for AI-powered features
- **OCR Processor** - EasyOCR pipeline for text extraction from images

### Frontend
- **Modern Web Interface** - Professional black and white theme
- **Real-time Updates** - Live project management and file processing
- **OCR Tools** - Dedicated interface for text extraction
- **AI Integration** - Document summaries and smart querying

## 📚 Usage Guide

### Creating Your First Project
1. Open http://localhost:8000
2. Navigate to "📁 Projects" tab
3. Enter project details:
   - **Name**: `ML_Research_2024`
   - **Path**: `/path/to/your/research`
   - **Template**: Choose from Research, Data Science, Minimal, or Software Dev
4. Click "Create Project"

The system creates organized folder structures automatically:

**Research Template:**
```
project_root/
├── literature/          # Research papers, reviews
├── data/
│   ├── raw/            # Original datasets
│   └── processed/      # Cleaned data
├── code/               # Analysis scripts
├── results/
│   ├── figures/        # Plots and visualizations
│   └── tables/         # Statistical outputs
├── drafts/             # Paper drafts
└── notes/              # Research notes
```

### Semantic Search Examples
```
Search Query: "machine learning experiments from last year"
→ Finds relevant papers, code, and data based on content similarity

Search Query: "statistical analysis python scripts"  
→ Locates analysis code and related documentation

Search Query: "figure generation visualization plots"
→ Discovers plotting scripts and generated images
```

### AI-Powered Features
- **Document Summaries**: Automatic AI summaries for search results
- **Content Queries**: Ask questions about specific files
- **Smart Organization**: AI suggestions for file categorization
- **OCR Processing**: Text extraction from images and scanned documents

## 🛠️ Configuration

### Environment Variables (.env)
```bash
# Server Settings
HOST=0.0.0.0
PORT=8000
ENV=development

# Database
DATABASE_URL=sqlite:///data/db/research.db

# AI Features
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_MODEL=llama3.2:3b
ENABLE_LLM_FEATURES=true

# OCR Settings
OCR_LANGUAGES=en
OCR_CONFIDENCE_THRESHOLD=50
ENABLE_OCR_PROCESSING=true

# Performance
MAX_FILE_SIZE_MB=100
CHUNK_SIZE=500
CACHE_SIZE=1000
```

### Project Templates
Customize templates by editing `backend/organizer.py`:
```python
PROJECT_TEMPLATES = {
    'research': ['literature', 'data/raw', 'data/processed', 'code', 'results/figures'],
    'data_science': ['data', 'notebooks', 'src', 'models', 'reports'],
    'custom': ['input', 'processing', 'output', 'documentation']
}
```

## 🔧 API Documentation

### Core Endpoints
```
GET    /                     # Web interface
GET    /health              # System health check
POST   /projects            # Create project
GET    /projects            # List projects
DELETE /projects/{id}       # Delete project
POST   /search              # Semantic search
POST   /organize            # Auto-organize files
```

### AI Integration Endpoints
```
POST   /api/llm/summarize         # AI document summarization
POST   /api/llm/query             # Ask questions about files
POST   /api/llm/suggest-organization # AI organization suggestions
GET    /api/llm/status            # LLM service status
```

### OCR Processing Endpoints
```
POST   /api/ocr/extract           # Process files with OCR
GET    /api/files/{id}/ocr-status # Check OCR processing status
POST   /api/files/{id}/ocr-reprocess # Reprocess with OCR
GET    /api/ocr/capabilities      # OCR service info
```

Interactive API documentation: http://localhost:8000/docs

## 🧪 Development

### Development Mode
```bash
# Start with auto-reload
source venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing the System
```bash
# Run system tests
python test_startup.py

# Clean start (reset databases)
clean_start.bat  # Windows
# ./clean_start.sh  # Linux/Mac
```

### Adding Custom Features
1. **Backend**: Add new endpoints in `backend/main.py`
2. **Frontend**: Extend UI in `frontend/index.html`
3. **AI Features**: Enhance `backend/llm_service.py`
4. **OCR**: Extend `backend/ocr_processor.py`

## 📊 Performance & Specifications

### Performance Targets
- **Search Response**: < 500ms for semantic queries
- **File Indexing**: > 100 files/second processing
- **OCR Processing**: 1-3 seconds per image
- **LLM Response**: 2-5 seconds per query
- **Memory Usage**: < 512MB for 10K files
- **Startup Time**: < 30 seconds

### Supported File Types
- **Text Documents**: .txt, .md, .py, .js, .json, .csv, .xml, .html
- **Research Papers**: .pdf, .doc, .docx, .tex
- **Data Files**: .csv, .xlsx, .json, .xml, .yaml
- **Images**: .png, .jpg, .jpeg, .gif, .bmp, .tiff (OCR processed)
- **Code**: .py, .js, .r, .java, .cpp, .c, .h, .ipynb

## 🗂️ Project Structure
```
IntelligentFileManager/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── database.py          # Database models & operations
│   ├── file_watcher.py      # File monitoring system
│   ├── organizer.py         # File organization logic
│   ├── search.py            # Semantic search engine
│   ├── llm_service.py       # AI integration (Ollama)
│   ├── ocr_processor.py     # OCR text extraction
│   └── config.py            # Configuration management
├── frontend/
│   └── index.html           # Modern web interface
├── data/
│   ├── projects/            # User project directories
│   └── db/                  # Databases (SQLite, ChromaDB)
├── dev_todo/                # Development status reports
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
├── setup_mvp.py            # Automated setup script
├── run_mvp.sh              # Start script (Linux/Mac)
├── run_mvp_windows.bat     # Start script (Windows)
└── README.md               # This file
```

## 🐛 Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# ChromaDB compilation issues (Windows)
pip install chromadb --no-build-isolation
# Or use the simplified search fallback

# Missing dependencies
pip install -r requirements.txt --upgrade
```

**Runtime Issues:**
```bash
# Port already in use
lsof -ti:8000 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :8000   # Windows (find PID, then taskkill)

# Database issues
rm -rf data/db/  # Delete and recreate
mkdir -p data/db

# AI/OCR not working
# Check if Ollama is running: ollama list
# Check OCR installation: python -c "import easyocr"
```

**Performance Issues:**
```bash
# Clear caches
rm -rf data/db/chroma/
rm -rf data/db/embeddings/

# Reduce chunk size in .env
CHUNK_SIZE=250
```

### Getting Help
- Check logs in `logs/app.log`
- View API documentation at `/docs`
- Test system status with `python test_startup.py`
- Review configuration in `.env` file

## 🛣️ Roadmap

### Current Status: Production Ready ✅
- [x] Complete MVP with core features
- [x] AI-powered document summarization  
- [x] OCR text extraction pipeline
- [x] Modern black/white UI theme
- [x] Real-time project management
- [x] Semantic search with embeddings

### Next Version Features 🚧
- [ ] Knowledge graph visualization (D3.js)
- [ ] Desktop app packaging (Electron)
- [ ] Advanced analytics dashboard
- [ ] Real-time collaboration features
- [ ] Mobile companion app
- [ ] Advanced ML learning patterns

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **UI/UX Enhancements**: Improved visualizations and interactions
- **AI Features**: Enhanced LLM integration and smart suggestions
- **Performance**: Optimization and caching improvements
- **File Support**: Additional file type processors
- **Documentation**: Tutorials and guides
- **Testing**: Automated test coverage

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- Powered by [Sentence-Transformers](https://www.sbert.net/) for semantic search
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- AI features by [Ollama](https://ollama.ai) for local LLM processing
- OCR capabilities by [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text extraction
- Inspired by modern research workflows and academic needs

---

**Research File Manager MVP - Transform your research workflow with AI-powered file intelligence** 🔬

*For questions, issues, or feature requests, please check the logs or create an issue in the repository.*