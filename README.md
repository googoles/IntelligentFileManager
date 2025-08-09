# Research File Manager MVP

**AI-Powered File Organization & Semantic Search for Researchers**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

The Research File Manager MVP is an intelligent file management system designed specifically for researchers, academics, and data scientists. It combines AI-powered semantic search with automatic file organization to help you manage your research projects efficiently.

### ✨ Key Features

- 🔍 **Semantic Search**: Find files using natural language queries powered by AI embeddings
- 🗂️ **Automatic Organization**: AI categorizes and organizes files based on content and naming patterns  
- 📁 **Project Templates**: Pre-built folder structures for research, data science, and academic workflows
- 🔄 **Real-time Monitoring**: Automatically indexes new files as they're added to your projects
- 🌐 **Modern Web Interface**: Intuitive, responsive design for desktop and mobile
- 🏠 **Privacy-First**: Complete local operation with no cloud dependencies required

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone or download the project
git clone <repository-url>
cd IntelligentFileManager

# Run the automated setup
python setup_mvp.py

# Start the application
./run_mvp.sh        # Linux/Mac
run_mvp.bat         # Windows

# Open browser to http://localhost:8000
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/projects data/db logs

# Start the server
python backend/main.py
```

## 📊 System Architecture

### Backend Components
- **FastAPI Application** (`backend/main.py`) - REST API server with all endpoints
- **Database Layer** (`backend/database.py`) - SQLAlchemy models and session management
- **File Monitor** (`backend/file_watcher.py`) - Real-time file system monitoring
- **Smart Organizer** (`backend/organizer.py`) - AI-powered file classification
- **Semantic Search** (`backend/search.py`) - Vector embeddings and similarity search

### Frontend
- **Single-Page App** (`frontend/index.html`) - Modern web interface with tabs for Search, Projects, and Organization

### Data Storage
- **SQLite Database** - File metadata, project info, and relationships
- **ChromaDB Vector Store** - Document embeddings for semantic search
- **Local File System** - Organized project files and folders

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, FastAPI, SQLAlchemy
- **AI/ML**: Sentence-Transformers, ChromaDB, all-MiniLM-L6-v2 model
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Database**: SQLite (default), PostgreSQL (production-ready)
- **File Monitoring**: Watchdog
- **Server**: Uvicorn ASGI server

## 📚 Usage Examples

### Creating a Research Project
1. Open http://localhost:8000
2. Go to "📁 Projects" tab
3. Enter project name: `ML_Research_2024`
4. Set path: `/home/user/research/ml_project`
5. Choose "Research Project" template
6. Click "Create Project"

### Semantic Search
```
Search Query: "machine learning experiments from last year"
Results: Finds relevant papers, code, and data files based on content similarity
```

### Auto-Organization
- Select a project in the "🗂️ Auto-Organize" tab
- Choose preview mode to see suggestions
- Enable auto-move to automatically organize files into:
  - `literature/` - Papers and documents
  - `data/` - Datasets and raw files  
  - `code/` - Scripts and programs
  - `results/` - Analysis outputs and figures

## 🏗️ Project Templates

### Research Template
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

### Data Science Template
```
project_root/
├── data/               # Datasets
├── notebooks/          # Jupyter notebooks  
├── src/                # Source code
├── models/             # Trained models
├── reports/            # Analysis reports
└── config/             # Configuration files
```

## 🔧 Configuration

Edit `.env` file to customize:

```bash
# Server settings
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=sqlite:///data/db/research.db

# AI Model settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500

# File processing
MAX_FILE_SIZE_MB=100
SUPPORTED_TEXT_EXTENSIONS=.txt,.md,.py,.js,.json,.csv
```

## 🧪 Development

### Development Mode
```bash
# Start with auto-reload
./run_dev.sh

# Access API documentation
http://localhost:8000/docs
```

### Testing
```bash
# Run tests (when available)
python -m pytest tests/

# Manual testing with sample data
python backend/demo_file_watcher.py
```

## 📊 Performance Targets

- **Search Response Time**: < 500ms
- **File Indexing Speed**: > 100 files/second
- **Memory Usage**: < 512MB for 10K files
- **Startup Time**: < 30 seconds

## 🗂️ File Structure

```
IntelligentFileManager/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── database.py          # Database models
│   ├── file_watcher.py      # File monitoring
│   ├── organizer.py         # File organization
│   ├── search.py            # Semantic search
│   └── __init__.py
├── frontend/
│   └── index.html           # Web interface
├── data/
│   ├── projects/            # User projects
│   └── db/                  # Databases
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
├── setup_mvp.py            # Automated setup
├── run_mvp.sh              # Start script
├── CLAUDE.md               # Claude Code documentation
├── QUICK_START.md          # Quick start guide
└── README.md               # This file
```

## 🐛 Troubleshooting

### Common Issues

**Port 8000 already in use:**
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9
```

**Dependencies not installing:**
```bash
# Upgrade pip and try again
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Search not working:**
```bash
# Check if models downloaded
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Database issues:**
```bash
# Reset database
rm data/db/research.db
python backend/main.py  # Will recreate automatically
```

## 🛣️ Roadmap

### Current MVP Features ✅
- [x] Project creation and management
- [x] File monitoring and indexing  
- [x] Semantic search with embeddings
- [x] Automatic file organization
- [x] Web-based interface
- [x] Local-first operation

### Upcoming Features 🚧
- [ ] PDF content extraction
- [ ] OCR for image documents
- [ ] Local LLM integration (Ollama)
- [ ] Real-time collaboration
- [ ] Desktop app (Electron)
- [ ] Mobile-responsive improvements
- [ ] Advanced analytics dashboard

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- UI/UX improvements
- Additional file type support
- Performance optimizations
- Documentation and tutorials
- Testing and bug fixes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the backend
- Powered by [Sentence-Transformers](https://www.sbert.net/) for semantic search
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Inspired by modern research workflows and academic needs

---

**Happy Researching!** 🔬

For questions or support, please check the logs in `logs/app.log` or create an issue in the repository.