"""
Research File Manager Backend Package

This package provides comprehensive file monitoring, organization, and search
capabilities for research projects.
"""

from .database import db_manager, Project, File, DatabaseManager
from .file_watcher import (
    FileHandler, 
    ProjectWatcher, 
    FileWatcherManager,
    file_watcher_manager,
    start_watching,
    stop_watching,
    stop_all_watchers
)

__version__ = "1.0.0"
__author__ = "Research File Manager Team"

# Export main components
__all__ = [
    # Database components
    "db_manager",
    "DatabaseManager", 
    "Project",
    "File",
    
    # File watcher components
    "FileHandler",
    "ProjectWatcher", 
    "FileWatcherManager",
    "file_watcher_manager",
    "start_watching",
    "stop_watching", 
    "stop_all_watchers",
]