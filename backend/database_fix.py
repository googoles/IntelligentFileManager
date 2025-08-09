#!/usr/bin/env python3
"""
Database compatibility fix for the metadata -> file_metadata column rename.

This module provides wrapper functions to handle the column name change
and ensure backward compatibility.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.exc import SQLAlchemyError

# Import the base models
from database import Base, Project, File, DATABASE_URL, engine, SessionLocal

# Database session management
@contextmanager
def db_session():
    """Provide a transactional scope for database operations"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


# CRUD Operations with fixed column names
def create_project(
    session: Session,
    name: str,
    path: str,
    ontology: Optional[Dict] = None
) -> Project:
    """Create a new project"""
    project = Project(
        name=name,
        path=path,
        ontology=ontology or {}
    )
    session.add(project)
    session.flush()
    return project


def create_file(
    session: Session,
    project_id: str,
    path: str,
    name: str,
    file_type: str,
    content: Optional[str] = None,
    file_metadata: Optional[Dict] = None,  # Changed from metadata to file_metadata
    embedding: Optional[str] = None
) -> File:
    """Create a new file record with proper column name"""
    file_record = File(
        project_id=project_id,
        path=path,
        name=name,
        type=file_type,
        content=content,
        file_metadata=file_metadata or {},  # Use file_metadata instead of metadata
        embedding=embedding
    )
    session.add(file_record)
    session.flush()
    return file_record


def get_all_projects(session: Session) -> List[Project]:
    """Get all projects"""
    return session.query(Project).all()


def get_project_by_id(session: Session, project_id: str) -> Optional[Project]:
    """Get a project by ID"""
    return session.query(Project).filter(Project.id == project_id).first()


def get_project_files(session: Session, project_id: str) -> List[File]:
    """Get all files for a project"""
    return session.query(File).filter(File.project_id == project_id).all()


def get_file_by_id(session: Session, file_id: str) -> Optional[File]:
    """Get a file by ID"""
    return session.query(File).filter(File.id == file_id).first()


def update_file_metadata(
    session: Session,
    file_id: str,
    file_metadata: Dict  # Changed from metadata to file_metadata
) -> Optional[File]:
    """Update file metadata with proper column name"""
    file_record = get_file_by_id(session, file_id)
    if file_record:
        file_record.file_metadata = file_metadata  # Use file_metadata
        session.flush()
    return file_record


def init_database():
    """Initialize the database with all tables"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully with file_metadata column")


# Export all functions for compatibility
__all__ = [
    'db_session',
    'create_project',
    'create_file',
    'get_all_projects',
    'get_project_by_id', 
    'get_project_files',
    'get_file_by_id',
    'update_file_metadata',
    'init_database',
    'Project',
    'File'
]