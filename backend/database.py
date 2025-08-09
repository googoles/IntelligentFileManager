"""
Database models and session management for the Research File Manager.

This module defines SQLAlchemy models for Projects and Files, along with
database initialization and session management functionality.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid
import json
import logging
import os
from pathlib import Path

from sqlalchemy import (
    create_engine, 
    Column, 
    String, 
    Text, 
    DateTime, 
    ForeignKey,
    JSON,
    Index,
    event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import Engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base class for all models
Base = declarative_base()


class Project(Base):
    """
    Project model representing a research project container.
    
    A Project contains multiple files and maintains metadata about the
    research structure and organization rules.
    """
    __tablename__ = 'projects'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    path = Column(Text, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ontology = Column(JSON, default=dict)  # JSON field for project-specific rules
    
    # Relationship to files
    files = relationship("File", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Project(id='{self.id}', name='{self.name}', path='{self.path}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'path': self.path,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'ontology': self.ontology or {},
            'file_count': len(self.files) if self.files else 0
        }


class File(Base):
    """
    File model representing an indexed file within a project.
    
    Stores file metadata, content, embeddings, and relationships to projects.
    """
    __tablename__ = 'files'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, ForeignKey('projects.id'), nullable=False, index=True)
    path = Column(Text, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    type = Column(String(50), nullable=False, index=True)  # File extension
    content = Column(Text)  # Extracted text content
    file_metadata = Column(JSON, default=dict)  # File metadata (size, modified, etc.) - renamed to avoid SQLAlchemy conflict
    embedding = Column(Text)  # JSON-serialized vector embedding
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship to project
    project = relationship("Project", back_populates="files")
    
    # Add composite index for common queries
    __table_args__ = (
        Index('ix_files_project_type', 'project_id', 'type'),
        Index('ix_files_project_created', 'project_id', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<File(id='{self.id}', name='{self.name}', type='{self.type}', path='{self.path}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert file to dictionary representation."""
        return {
            'id': self.id,
            'project_id': self.project_id,
            'path': self.path,
            'name': self.name,
            'type': self.type,
            'content_length': len(self.content) if self.content else 0,
            'metadata': self.metadata or {},
            'has_embedding': bool(self.embedding),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @property
    def embedding_vector(self) -> Optional[List[float]]:
        """Get embedding as a list of floats."""
        if not self.embedding:
            return None
        try:
            return json.loads(self.embedding)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error parsing embedding for file {self.id}: {e}")
            return None
    
    @embedding_vector.setter
    def embedding_vector(self, vector: List[float]) -> None:
        """Set embedding from a list of floats."""
        if vector is None:
            self.embedding = None
        else:
            try:
                self.embedding = json.dumps(vector)
            except (TypeError, ValueError) as e:
                logger.error(f"Error serializing embedding for file {self.id}: {e}")
                raise


class DatabaseManager:
    """
    Database manager for handling connections, sessions, and initialization.
    
    Provides centralized database management with proper error handling
    and connection pooling.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database URL. If None, defaults to SQLite in data/db/
        """
        if database_url is None:
            # Create data directory if it doesn't exist
            data_dir = Path("data/db")
            data_dir.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{data_dir}/research.db"
        
        self.database_url = database_url
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        
        # Initialize database tables
        self._init_db()
    
    def _create_engine(self) -> Engine:
        """Create database engine with appropriate settings."""
        if self.database_url.startswith("sqlite"):
            engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},  # For SQLite
                echo=False  # Set to True for SQL debugging
            )
            
            # Enable foreign key support for SQLite
            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrency
                cursor.close()
                
        else:
            # PostgreSQL or other databases
            engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                echo=False
            )
        
        return engine
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            SQLAlchemy session instance
        """
        return self.SessionLocal()
    
    def close(self) -> None:
        """Close database engine."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def add_file(self, 
                 project_id: str,
                 file_path: str, 
                 name: str,
                 file_type: str,
                 content: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[File]:
        """
        Add a file to the database (convenience method for file watcher).
        
        Args:
            project_id: Project ID
            file_path: Path to the file
            name: File name
            file_type: File type/extension
            content: File content
            metadata: File metadata
            
        Returns:
            Created File object or None if failed
        """
        session = self.get_session()
        try:
            # Check if file already exists
            existing = session.query(File).filter(File.path == file_path).first()
            if existing:
                # Update existing file
                existing.content = content
                existing.metadata = metadata or {}
                session.commit()
                session.refresh(existing)
                return existing
            else:
                # Create new file
                return create_file(
                    session=session,
                    project_id=project_id,
                    path=file_path,
                    name=name,
                    file_type=file_type,
                    content=content,
                    metadata=metadata
                )
        except Exception as e:
            logger.error(f"Error adding file to database: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from the database by path.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if file was deleted, False if not found
        """
        session = self.get_session()
        try:
            file_record = session.query(File).filter(File.path == file_path).first()
            if file_record:
                session.delete(file_record)
                session.commit()
                logger.debug(f"Deleted file from database: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file from database: {e}")
            session.rollback()
            return False
        finally:
            session.close()


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def init_database(database_url: Optional[str] = None) -> DatabaseManager:
    """
    Initialize the global database manager.
    
    Args:
        database_url: Database URL. If None, uses SQLite default
        
    Returns:
        DatabaseManager instance
    """
    global db_manager
    
    if db_manager is None:
        db_manager = DatabaseManager(database_url)
        logger.info(f"Database initialized with URL: {database_url or 'SQLite default'}")
    
    return db_manager


def get_db_session() -> Session:
    """
    Get database session from global manager.
    
    Returns:
        SQLAlchemy session instance
        
    Raises:
        RuntimeError: If database is not initialized
    """
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    return db_manager.get_session()


def create_project(
    session: Session,
    name: str,
    path: str,
    ontology: Optional[Dict[str, Any]] = None
) -> Project:
    """
    Create a new project.
    
    Args:
        session: Database session
        name: Project name
        path: Project path
        ontology: Optional ontology rules
        
    Returns:
        Created project instance
        
    Raises:
        ValueError: If project with same path already exists
    """
    # Check if project with same path exists
    existing = session.query(Project).filter(Project.path == path).first()
    if existing:
        raise ValueError(f"Project with path '{path}' already exists")
    
    project = Project(
        name=name,
        path=path,
        ontology=ontology or {}
    )
    
    session.add(project)
    session.commit()
    session.refresh(project)
    
    logger.info(f"Created project: {name} at {path}")
    return project


def create_file(
    session: Session,
    project_id: str,
    path: str,
    name: str,
    file_type: str,
    content: Optional[str] = None,
    file_metadata: Optional[Dict[str, Any]] = None,  # Changed parameter name
    embedding_vector: Optional[List[float]] = None
) -> File:
    """
    Create a new file record.
    
    Args:
        session: Database session
        project_id: Parent project ID
        path: File path
        name: File name
        file_type: File type/extension
        content: Extracted text content
        file_metadata: File metadata (renamed from metadata)
        embedding_vector: Vector embedding
        
    Returns:
        Created file instance
    """
    file_record = File(
        project_id=project_id,
        path=path,
        name=name,
        type=file_type,
        content=content,
        file_metadata=file_metadata or {}  # Use file_metadata column
    )
    
    if embedding_vector:
        file_record.embedding_vector = embedding_vector
    
    session.add(file_record)
    session.commit()
    session.refresh(file_record)
    
    logger.debug(f"Created file record: {name} ({file_type})")
    return file_record


def get_project_by_id(session: Session, project_id: str) -> Optional[Project]:
    """Get project by ID."""
    return session.query(Project).filter(Project.id == project_id).first()


def get_project_by_path(session: Session, path: str) -> Optional[Project]:
    """Get project by path."""
    return session.query(Project).filter(Project.path == path).first()


def get_file_by_id(session: Session, file_id: str) -> Optional[File]:
    """Get file by ID."""
    return session.query(File).filter(File.id == file_id).first()


def get_files_by_project(session: Session, project_id: str) -> List[File]:
    """Get all files in a project."""
    return session.query(File).filter(File.project_id == project_id).all()


# Aliases for compatibility
def get_all_projects(session: Session) -> List[Project]:
    """Get all projects (alias for backwards compatibility)."""
    return session.query(Project).all()


def get_project_files(session: Session, project_id: str) -> List[File]:
    """Get all files for a project (alias for get_files_by_project)."""
    return get_files_by_project(session, project_id)


def get_files_by_type(session: Session, project_id: str, file_type: str) -> List[File]:
    """Get files by type in a project."""
    return session.query(File).filter(
        File.project_id == project_id,
        File.type == file_type
    ).all()


def delete_project(session: Session, project_id: str) -> bool:
    """
    Delete a project and all its files.
    
    Args:
        session: Database session
        project_id: Project ID to delete
        
    Returns:
        True if project was deleted, False if not found
    """
    project = get_project_by_id(session, project_id)
    if not project:
        return False
    
    session.delete(project)
    session.commit()
    
    logger.info(f"Deleted project: {project.name} ({project_id})")
    return True


def delete_file(session: Session, file_id: str) -> bool:
    """
    Delete a file record.
    
    Args:
        session: Database session
        file_id: File ID to delete
        
    Returns:
        True if file was deleted, False if not found
    """
    file_record = get_file_by_id(session, file_id)
    if not file_record:
        return False
    
    session.delete(file_record)
    session.commit()
    
    logger.debug(f"Deleted file record: {file_record.name} ({file_id})")
    return True


# Context manager for database sessions
class db_session:
    """
    Context manager for database sessions with automatic cleanup.
    
    Usage:
        with db_session() as session:
            # Use session here
            pass
    """
    
    def __enter__(self) -> Session:
        self.session = get_db_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
        finally:
            self.session.close()


# For backwards compatibility and direct access
def get_session() -> Session:
    """Get database session (alias for get_db_session)."""
    return get_db_session()


if __name__ == "__main__":
    # Example usage and testing
    print("Initializing database...")
    init_database()
    
    # Test basic operations
    with db_session() as session:
        # Create a test project
        project = create_project(
            session,
            name="Test Project",
            path="./test_data",
            ontology={"file_types": ["pdf", "txt", "md"]}
        )
        print(f"Created project: {project}")
        
        # Create a test file
        file_record = create_file(
            session,
            project_id=project.id,
            path="./test_data/readme.txt",
            name="readme.txt",
            file_type=".txt",
            content="This is a test file for the research file manager.",
            file_metadata={"size": 100, "last_modified": datetime.utcnow().timestamp()}
        )
        print(f"Created file: {file_record}")
        
        # Query data
        projects = session.query(Project).all()
        print(f"Found {len(projects)} project(s)")
        
        files = get_files_by_project(session, project.id)
        print(f"Found {len(files)} file(s) in project")
    
    print("Database operations completed successfully!")