"""
Setup script for the Research File Manager backend.

This script handles database initialization, migration, and basic setup tasks.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from database import init_database, db_session, Project, File, Base
from config import get_config


def setup_database(database_url: Optional[str] = None) -> bool:
    """
    Initialize the database with all required tables.
    
    Args:
        database_url: Optional database URL override
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print("🗄️  Initializing database...")
        
        # Initialize database manager
        db_manager = init_database(database_url)
        print(f"✅ Database initialized: {db_manager.database_url}")
        
        # Verify tables were created
        with db_session() as session:
            # Test basic operations
            project_count = session.query(Project).count()
            file_count = session.query(File).count()
            print(f"📊 Current database stats:")
            print(f"   - Projects: {project_count}")
            print(f"   - Files: {file_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


def create_sample_data() -> bool:
    """
    Create sample data for testing and demonstration.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print("📝 Creating sample data...")
        
        with db_session() as session:
            # Check if sample data already exists
            existing_project = session.query(Project).filter(
                Project.name == "Sample Research Project"
            ).first()
            
            if existing_project:
                print("⏭️  Sample data already exists, skipping...")
                return True
            
            # Create sample project
            from database import create_project, create_file
            
            project = create_project(
                session,
                name="Sample Research Project",
                path="./data/projects/sample_research",
                ontology={
                    "description": "A sample research project for testing",
                    "file_types": ["pdf", "txt", "md", "csv", "py"],
                    "categories": ["literature", "data", "code", "results"]
                }
            )
            
            # Create sample files
            sample_files = [
                {
                    "name": "readme.md",
                    "path": "./data/projects/sample_research/readme.md",
                    "type": ".md",
                    "content": "# Sample Research Project\n\nThis is a sample project for testing the Research File Manager.",
                    "metadata": {"size": 100, "category": "documentation"}
                },
                {
                    "name": "analysis.py",
                    "path": "./data/projects/sample_research/code/analysis.py",
                    "type": ".py", 
                    "content": "import pandas as pd\n\ndef analyze_data(df):\n    return df.describe()",
                    "metadata": {"size": 75, "category": "code"}
                },
                {
                    "name": "results.csv",
                    "path": "./data/projects/sample_research/data/results.csv",
                    "type": ".csv",
                    "content": "metric,value\naccuracy,0.95\nprecision,0.92\nrecall,0.89",
                    "metadata": {"size": 60, "category": "results"}
                }
            ]
            
            for file_data in sample_files:
                create_file(
                    session,
                    project_id=project.id,
                    path=file_data["path"],
                    name=file_data["name"],
                    file_type=file_data["type"],
                    content=file_data["content"],
                    metadata=file_data["metadata"]
                )
            
            print(f"✅ Created sample project '{project.name}' with {len(sample_files)} files")
            
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False


def setup_directories() -> bool:
    """
    Create required directory structure.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print("📁 Setting up directory structure...")
        
        config = get_config()
        
        # Create main directories
        directories = [
            Path("data"),
            Path("data/db"),
            Path("data/projects"), 
            Path("logs"),
            Path(config.CHROMA_DB_PATH).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory setup failed: {e}")
        return False


def verify_installation() -> bool:
    """
    Verify that all components are working correctly.
    
    Returns:
        True if all checks pass, False otherwise
    """
    try:
        print("🔍 Verifying installation...")
        
        # Test database connection
        with db_session() as session:
            session.query(Project).first()
            print("   ✅ Database connection working")
        
        # Test configuration
        config = get_config()
        assert config.DATABASE_URL is not None
        print("   ✅ Configuration loaded")
        
        # Test directory structure
        required_dirs = [Path("data"), Path("data/db")]
        for directory in required_dirs:
            assert directory.exists(), f"Directory {directory} not found"
        print("   ✅ Directory structure verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Research File Manager Backend Setup")
    print("=" * 50)
    
    # Setup logging
    config = get_config()
    
    success = True
    
    # Step 1: Setup directories
    if not setup_directories():
        success = False
    
    # Step 2: Initialize database
    if success and not setup_database():
        success = False
    
    # Step 3: Create sample data (optional)
    if success:
        create_sample_data()  # Don't fail on sample data issues
    
    # Step 4: Verify installation
    if success and not verify_installation():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Backend setup completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt") 
        print("2. Start the application: python -m backend.database")
        print("3. Or run tests: pytest tests/")
    else:
        print("❌ Backend setup failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()