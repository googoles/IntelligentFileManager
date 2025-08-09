"""
Utility functions for the Research File Manager backend.

This module provides helper functions for common database operations,
file processing, and data validation.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import mimetypes
import logging

from .database import Session, File, Project, get_files_by_project
from .config import get_config

logger = logging.getLogger(__name__)
config = get_config()


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file for duplicate detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash as hexadecimal string
    """
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except (OSError, IOError) as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file metadata
    """
    try:
        file_stat = os.stat(file_path)
        file_path_obj = Path(file_path)
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        metadata = {
            'size': file_stat.st_size,
            'created': file_stat.st_ctime,
            'modified': file_stat.st_mtime,
            'accessed': file_stat.st_atime,
            'extension': file_path_obj.suffix.lower(),
            'mime_type': mime_type,
            'is_hidden': file_path_obj.name.startswith('.'),
            'hash': calculate_file_hash(file_path),
            'readable': os.access(file_path, os.R_OK),
            'writable': os.access(file_path, os.W_OK)
        }
        
        return metadata
        
    except (OSError, IOError) as e:
        logger.error(f"Error getting metadata for {file_path}: {e}")
        return {}


def read_text_file(file_path: str, max_length: Optional[int] = None) -> str:
    """
    Read text content from a file with encoding detection.
    
    Args:
        file_path: Path to the file
        max_length: Maximum characters to read (None for no limit)
        
    Returns:
        File content as string
    """
    if max_length is None:
        max_length = config.MAX_CONTENT_LENGTH
    
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read(max_length)
                return content
        except (UnicodeDecodeError, OSError):
            continue
    
    logger.warning(f"Could not decode file {file_path} with any encoding")
    return ""


def is_text_file(file_path: str) -> bool:
    """
    Check if a file is a text file based on extension and content.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is text, False otherwise
    """
    file_ext = Path(file_path).suffix.lower()
    
    # Check by extension
    if file_ext in config.SUPPORTED_TEXT_EXTENSIONS:
        return True
    
    # Check by MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('text/'):
        return True
    
    # Check by reading a small portion
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(8192)  # Read first 8KB
        
        # Check for null bytes (binary indicator)
        if b'\x00' in sample:
            return False
        
        # Try to decode as text
        try:
            sample.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
            
    except (OSError, IOError):
        return False


def find_duplicate_files(session: Session, project_id: str) -> List[Tuple[str, List[File]]]:
    """
    Find duplicate files in a project based on content hash.
    
    Args:
        session: Database session
        project_id: Project ID to search
        
    Returns:
        List of tuples (hash, list of duplicate files)
    """
    files = get_files_by_project(session, project_id)
    hash_groups = {}
    
    for file in files:
        if file.metadata and 'hash' in file.metadata:
            file_hash = file.metadata['hash']
            if file_hash:
                if file_hash not in hash_groups:
                    hash_groups[file_hash] = []
                hash_groups[file_hash].append(file)
    
    # Return only groups with multiple files
    duplicates = [(hash_val, files) for hash_val, files in hash_groups.items() if len(files) > 1]
    return duplicates


def get_project_statistics(session: Session, project_id: str) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a project.
    
    Args:
        session: Database session
        project_id: Project ID
        
    Returns:
        Dictionary with project statistics
    """
    files = get_files_by_project(session, project_id)
    
    if not files:
        return {
            'total_files': 0,
            'total_size': 0,
            'file_types': {},
            'has_content': 0,
            'has_embedding': 0,
            'last_updated': None
        }
    
    stats = {
        'total_files': len(files),
        'total_size': 0,
        'file_types': {},
        'has_content': 0,
        'has_embedding': 0,
        'last_updated': max(f.created_at for f in files) if files else None
    }
    
    for file in files:
        # File type distribution
        file_type = file.type
        stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
        
        # Size calculation
        if file.metadata and 'size' in file.metadata:
            stats['total_size'] += file.metadata['size']
        
        # Content availability
        if file.content:
            stats['has_content'] += 1
        
        # Embedding availability
        if file.embedding:
            stats['has_embedding'] += 1
    
    return stats


def clean_old_files(session: Session, project_id: str, days: int = 30) -> int:
    """
    Clean up old file records that no longer exist on disk.
    
    Args:
        session: Database session
        project_id: Project ID
        days: Files older than this many days will be checked
        
    Returns:
        Number of cleaned up files
    """
    from datetime import timedelta
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    files = session.query(File).filter(
        File.project_id == project_id,
        File.created_at < cutoff_date
    ).all()
    
    cleaned_count = 0
    
    for file in files:
        if not os.path.exists(file.path):
            session.delete(file)
            cleaned_count += 1
            logger.info(f"Cleaned up missing file record: {file.path}")
    
    if cleaned_count > 0:
        session.commit()
        logger.info(f"Cleaned up {cleaned_count} old file records")
    
    return cleaned_count


def validate_project_path(path: str) -> Tuple[bool, str]:
    """
    Validate a project path for creation.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path_obj = Path(path)
    
    # Check if path is absolute
    if not path_obj.is_absolute():
        return False, "Path must be absolute"
    
    # Check if parent directory exists
    if not path_obj.parent.exists():
        return False, f"Parent directory does not exist: {path_obj.parent}"
    
    # Check if we can create the directory
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        return False, f"Cannot create directory: {e}"
    
    # Check if we can write to the directory
    if not os.access(path, os.W_OK):
        return False, "Directory is not writable"
    
    return True, ""


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def batch_update_file_metadata(session: Session, project_id: str) -> int:
    """
    Update metadata for all files in a project by reading from disk.
    
    Args:
        session: Database session  
        project_id: Project ID
        
    Returns:
        Number of files updated
    """
    files = get_files_by_project(session, project_id)
    updated_count = 0
    
    for file in files:
        if os.path.exists(file.path):
            try:
                # Get fresh metadata
                new_metadata = get_file_metadata(file.path)
                
                # Update if metadata is different
                if file.metadata != new_metadata:
                    file.metadata = new_metadata
                    updated_count += 1
                    logger.debug(f"Updated metadata for {file.name}")
                    
            except Exception as e:
                logger.error(f"Error updating metadata for {file.path}: {e}")
    
    if updated_count > 0:
        session.commit()
        logger.info(f"Updated metadata for {updated_count} files")
    
    return updated_count


def export_project_data(session: Session, project_id: str) -> Dict[str, Any]:
    """
    Export all project data for backup or migration.
    
    Args:
        session: Database session
        project_id: Project ID
        
    Returns:
        Dictionary with all project data
    """
    # Get project
    project = session.query(Project).filter(Project.id == project_id).first()
    if not project:
        return {}
    
    # Get all files
    files = get_files_by_project(session, project_id)
    
    export_data = {
        'project': project.to_dict(),
        'files': [file.to_dict() for file in files],
        'statistics': get_project_statistics(session, project_id),
        'export_timestamp': datetime.utcnow().isoformat(),
        'version': '1.0'
    }
    
    return export_data


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test file size formatting
    print(f"1024 bytes: {format_file_size(1024)}")
    print(f"1048576 bytes: {format_file_size(1048576)}")
    
    # Test path validation
    is_valid, error = validate_project_path("/tmp/test_project")
    print(f"Path validation: {is_valid}, {error}")
    
    print("Utility tests completed!")