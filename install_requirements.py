"""
Install requirements for the Research File Manager project.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for the semantic search system."""
    
    print("🔧 Installing Research File Manager requirements...")
    
    # Core requirements for semantic search functionality
    requirements = [
        "sqlalchemy>=2.0.0",
        "sentence-transformers>=2.2.0", 
        "chromadb>=0.4.0",
        "numpy>=1.24.0",
        "fastapi>=0.104.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.4.0"
    ]
    
    for requirement in requirements:
        try:
            print(f"Installing {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print(f"✅ {requirement} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {requirement}: {e}")
            return False
    
    print("\n🎉 All requirements installed successfully!")
    return True

if __name__ == "__main__":
    success = install_requirements()
    sys.exit(0 if success else 1)