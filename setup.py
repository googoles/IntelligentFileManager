#!/usr/bin/env python3
"""
Setup script for Research File Manager MVP
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    required_packages = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0", 
        "watchdog>=3.0.0",
        "sqlalchemy>=2.0.0",
        "aiofiles>=23.0.0",
        "python-multipart>=0.0.6"
    ]
    
    for package in required_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True


def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "data/db",
        "data/projects", 
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def main():
    """Main setup function"""
    print("🚀 Research File Manager MVP Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return 1
    
    # Create directories
    create_directories()
    
    print("\n✅ Setup completed successfully!")
    print("\nYou can now run:")
    print("  python3 test_file_watcher.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())