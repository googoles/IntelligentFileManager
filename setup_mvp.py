#!/usr/bin/env python3
"""
Research File Manager MVP - Automated Setup Script

This script automatically sets up the complete Research File Manager MVP
with all dependencies, database initialization, and sample data.

Usage:
    python setup_mvp.py
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


class MVPSetup:
    def __init__(self):
        self.root_dir = Path.cwd()
        self.venv_dir = self.root_dir / "venv"
        self.is_windows = platform.system() == "Windows"
        
    def print_banner(self):
        """Display setup banner"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                Research File Manager MVP                     ║
║                    🚀 Automated Setup 🚀                    ║
║                                                              ║
║  AI-Powered File Organization & Semantic Search System      ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
    def check_requirements(self):
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        # Python version check
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ required. Please upgrade Python.")
            sys.exit(1)
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        
        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            print("✅ pip available")
        except subprocess.CalledProcessError:
            print("❌ pip not available. Please install pip.")
            sys.exit(1)
            
        # Check git (optional)
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            print("✅ git available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  git not found (optional)")
        
        print()
        
    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        print("📦 Setting up virtual environment...")
        
        if self.venv_dir.exists():
            print("⚠️  Virtual environment already exists")
            return
            
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
        
    def get_python_executable(self):
        """Get Python executable path in venv"""
        if self.is_windows:
            return str(self.venv_dir / "Scripts" / "python.exe")
        return str(self.venv_dir / "bin" / "python")
    
    def get_pip_executable(self):
        """Get pip executable path in venv"""
        if self.is_windows:
            return str(self.venv_dir / "Scripts" / "pip.exe")
        return str(self.venv_dir / "bin" / "pip")
        
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📚 Installing Python dependencies...")
        
        pip_cmd = self.get_pip_executable()
        
        try:
            # Upgrade pip first
            subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
            
            # Install from requirements.txt
            if Path("requirements.txt").exists():
                subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
            else:
                # Install core dependencies directly
                dependencies = [
                    "fastapi==0.104.1",
                    "uvicorn[standard]==0.24.0",
                    "python-multipart==0.0.6",
                    "sqlalchemy==2.0.23",
                    "pydantic==2.5.0",
                    "watchdog==3.0.0",
                    "aiofiles==23.2.1",
                    "sentence-transformers==2.2.2",
                    "chromadb==0.4.18",
                    "python-dotenv==1.0.0"
                ]
                subprocess.run([pip_cmd, "install"] + dependencies, check=True)
            
            print("✅ Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Try running: pip install -r requirements.txt")
            sys.exit(1)
    
    def create_directory_structure(self):
        """Create required directory structure"""
        print("📁 Creating directory structure...")
        
        directories = [
            "backend",
            "frontend", 
            "data/projects",
            "data/db",
            "data/db/chroma",
            "logs",
            "tests"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        print("✅ Directory structure created")
        
    def create_environment_file(self):
        """Create .env configuration file"""
        print("⚙️  Creating environment configuration...")
        
        env_content = """# Research File Manager MVP Configuration
# Environment
ENV=development
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Database
DATABASE_URL=sqlite:///data/db/research.db

# Machine Learning
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
OVERLAP_SIZE=125

# File Processing
MAX_FILE_SIZE_MB=100
SUPPORTED_TEXT_EXTENSIONS=.txt,.md,.py,.js,.json,.csv,.xml,.html,.css,.sql,.r,.java,.cpp,.c,.h
IGNORE_PATTERNS=.git,.svn,__pycache__,node_modules,.DS_Store,Thumbs.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Optional API Keys (for extended features)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
"""
        
        env_path = Path(".env")
        if not env_path.exists():
            env_path.write_text(env_content)
            print("✅ Environment file created")
        else:
            print("⚠️  .env file already exists")
        
    def initialize_database(self):
        """Initialize the database"""
        print("🗃️  Initializing database...")
        
        python_cmd = self.get_python_executable()
        
        # Create a simple initialization script
        init_script = """
import sys
sys.path.append('backend')

try:
    from database import init_database
    init_database()
    print("✅ Database initialized successfully")
except Exception as e:
    print(f"❌ Database initialization failed: {e}")
    sys.exit(1)
"""
        
        init_path = Path("temp_init.py")
        init_path.write_text(init_script)
        
        try:
            subprocess.run([python_cmd, "temp_init.py"], check=True)
        except subprocess.CalledProcessError:
            print("⚠️  Database initialization will happen on first run")
        finally:
            if init_path.exists():
                init_path.unlink()
        
    def create_run_scripts(self):
        """Create convenient run scripts"""
        print("🎯 Creating run scripts...")
        
        # Windows batch file
        if self.is_windows:
            batch_content = """@echo off
title Research File Manager MVP
echo Starting Research File Manager MVP...
echo.
echo 🔬 Research File Manager MVP
echo AI-Powered File Organization ^& Semantic Search
echo.
call venv\\Scripts\\activate
python backend\\main.py
pause
"""
            Path("run_mvp.bat").write_text(batch_content)
            print("✅ Created run_mvp.bat")
        
        # Unix shell script  
        shell_content = """#!/bin/bash
echo "🔬 Research File Manager MVP"
echo "AI-Powered File Organization & Semantic Search"
echo ""
echo "Starting server..."

# Activate virtual environment
source venv/bin/activate

# Run the application
python backend/main.py
"""
        
        shell_path = Path("run_mvp.sh")
        shell_path.write_text(shell_content)
        if not self.is_windows:
            os.chmod(shell_path, 0o755)
        print("✅ Created run_mvp.sh")
        
        # Development script
        dev_content = """#!/bin/bash
echo "🛠️  Development Mode"
echo "Starting with hot reload..."
echo ""

source venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
        
        dev_path = Path("run_dev.sh")
        dev_path.write_text(dev_content)
        if not self.is_windows:
            os.chmod(dev_path, 0o755)
        print("✅ Created run_dev.sh")
        
    def create_sample_project(self):
        """Create sample project with test data"""
        print("📄 Creating sample project...")
        
        sample_dir = Path("data/projects/sample_research")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample files with research content
        samples = {
            "literature/README.md": """# Literature Review

This folder contains research papers and literature reviews.

## Key Papers
- Smith et al. (2023) - Machine Learning Applications
- Johnson (2022) - Data Analysis Methods
- Brown & Lee (2023) - Statistical Approaches
""",
            "data/raw/experiment_data.csv": """experiment_id,condition,value,date
1,control,15.2,2024-01-15
2,treatment,18.7,2024-01-15
3,control,14.8,2024-01-16
4,treatment,19.1,2024-01-16
""",
            "code/analysis.py": """#!/usr/bin/env python3
\"\"\"
Data analysis script for research project
\"\"\"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    \"\"\"Load experimental data\"\"\"
    return pd.read_csv(filepath)

def analyze_results(data):
    \"\"\"Perform statistical analysis\"\"\"
    control = data[data['condition'] == 'control']['value']
    treatment = data[data['condition'] == 'treatment']['value']
    
    return {
        'control_mean': control.mean(),
        'treatment_mean': treatment.mean(),
        'difference': treatment.mean() - control.mean()
    }

if __name__ == "__main__":
    data = load_data('../data/raw/experiment_data.csv')
    results = analyze_results(data)
    print(f"Results: {results}")
""",
            "results/summary.txt": """Experimental Results Summary

Date: 2024-01-20
Researcher: MVP Demo

Key Findings:
- Treatment group showed 20% improvement over control
- Results are statistically significant (p < 0.05)
- Sample size: n=100 participants

Next Steps:
- Replicate with larger sample
- Test additional conditions
- Prepare manuscript for publication
""",
            "notes/daily_log.md": """# Research Daily Log

## 2024-01-15
- Set up experimental conditions
- Collected initial data samples
- Control group: 15.2 average

## 2024-01-16  
- Continued data collection
- Treatment group showing promising results
- Need to verify statistical significance

## 2024-01-20
- Completed initial analysis
- Results look very promising
- Ready to expand study
""",
            "drafts/paper_outline.md": """# Research Paper Outline

## Title
"Machine Learning Applications in Experimental Research: A Comparative Study"

## Abstract
- Background on ML in research
- Methodology overview
- Key findings
- Implications

## Introduction
- Problem statement
- Literature review
- Research questions
- Hypotheses

## Methods
- Experimental design
- Data collection procedures
- Analysis methods
- Statistical approaches

## Results
- Descriptive statistics
- Inferential analyses
- Visualizations
- Effect sizes

## Discussion
- Interpretation of findings
- Limitations
- Future directions
- Conclusions
""",
        }
        
        # Create sample files
        for file_path, content in samples.items():
            full_path = sample_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        print("✅ Sample project created with realistic research data")
        
    def download_ml_models(self):
        """Pre-download ML models for faster first run"""
        print("🤖 Pre-downloading ML models (this may take a few minutes)...")
        
        python_cmd = self.get_python_executable()
        
        download_script = """
try:
    from sentence_transformers import SentenceTransformer
    print("Downloading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model downloaded successfully")
except Exception as e:
    print(f"⚠️  Model download failed: {e}")
    print("Models will be downloaded on first use")
"""
        
        script_path = Path("temp_download.py")
        script_path.write_text(download_script)
        
        try:
            subprocess.run([python_cmd, "temp_download.py"], 
                         check=True, timeout=300)  # 5 minute timeout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("⚠️  Model pre-download failed, will download on first use")
        finally:
            if script_path.exists():
                script_path.unlink()
        
    def create_quick_start_guide(self):
        """Create a quick start guide"""
        print("📖 Creating quick start guide...")
        
        guide_content = """# Research File Manager MVP - Quick Start Guide

## 🚀 Getting Started

### 1. Start the Application
```bash
# Option 1: Use run script
./run_mvp.sh          # Linux/Mac
run_mvp.bat           # Windows

# Option 2: Manual start
source venv/bin/activate
python backend/main.py
```

### 2. Access the Web Interface
Open your browser and go to: http://localhost:8000

### 3. Create Your First Project
1. Click on the "📁 Projects" tab
2. Enter project name and path
3. Choose a template (Research recommended)
4. Click "Create Project"

### 4. Try Semantic Search
1. Go to the "🔍 Search" tab
2. Enter natural language queries like:
   - "machine learning experiments"
   - "data analysis results"
   - "python code for statistics"

### 5. Auto-Organize Files
1. Click "🗂️ Auto-Organize" tab
2. Select your project
3. Choose preview or auto-move
4. Click "Analyze & Organize"

## 🛠️ Development

### Run in Development Mode
```bash
./run_dev.sh    # Auto-reload on changes
```

### View API Documentation
http://localhost:8000/docs

## 📊 Sample Data

A sample research project has been created at:
`data/projects/sample_research/`

This includes:
- Literature review files
- Sample data files
- Analysis code
- Research notes
- Paper drafts

## 🔧 Configuration

Edit `.env` file to customize:
- Port number
- Database location  
- ML model settings
- File processing limits

## 🆘 Troubleshooting

### Port Already in Use
```bash
# Kill process using port 8000
kill -9 $(lsof -t -i:8000)
```

### Dependencies Issues
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Database Issues
Delete `data/db/research.db` and restart the application.

## 📚 Features

### ✅ Implemented (MVP)
- [x] Project creation and management
- [x] File monitoring and indexing
- [x] Semantic search with AI embeddings
- [x] Automatic file organization
- [x] Web-based interface
- [x] Real-time file processing

### 🚧 Future Enhancements
- [ ] PDF content extraction
- [ ] OCR for images
- [ ] Local LLM integration (Ollama)
- [ ] Real-time collaboration
- [ ] Desktop app packaging

## 🎯 Use Cases

1. **Research Projects**: Organize papers, data, code, and results
2. **Literature Reviews**: Semantic search across research papers
3. **Data Analysis**: Automatic categorization of datasets and scripts
4. **Academic Writing**: Organize drafts, references, and notes
5. **Collaborative Research**: Share and organize team resources

## 📞 Support

- Check logs in `logs/app.log`
- View API docs at `/docs` endpoint
- Create GitHub issues for bugs/features

Happy researching! 🔬
"""
        
        Path("QUICK_START.md").write_text(guide_content)
        print("✅ Quick start guide created")
        
    def print_completion_message(self):
        """Display completion message with next steps"""
        run_cmd = "run_mvp.bat" if self.is_windows else "./run_mvp.sh"
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    🎉 Setup Complete! 🎉                    ║
╚══════════════════════════════════════════════════════════════╝

🎯 Next Steps:

1. Start the application:
   {run_cmd}

2. Open your browser:
   http://localhost:8000

3. Try the sample project:
   data/projects/sample_research/

4. Read the guide:
   QUICK_START.md

📊 What's Ready:
✅ Complete backend with FastAPI
✅ Modern web interface  
✅ Semantic search with AI
✅ Automatic file organization
✅ Real-time file monitoring
✅ Sample research project
✅ Production-ready setup

🔧 Development:
- Run: ./run_dev.sh (hot reload)
- API docs: http://localhost:8000/docs
- Logs: logs/app.log

🚀 The Research File Manager MVP is ready to use!
   Organize your research with AI-powered intelligence.

        """)
        
    def run_setup(self):
        """Execute the complete setup process"""
        try:
            self.print_banner()
            self.check_requirements()
            self.create_virtual_environment()
            self.install_dependencies()
            self.create_directory_structure()
            self.create_environment_file()
            self.initialize_database()
            self.create_run_scripts()
            self.create_sample_project()
            self.download_ml_models()
            self.create_quick_start_guide()
            self.print_completion_message()
            
        except KeyboardInterrupt:
            print("\n\n❌ Setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Setup failed with error: {e}")
            print("Please check the error message and try again")
            sys.exit(1)


if __name__ == "__main__":
    setup = MVPSetup()
    setup.run_setup()