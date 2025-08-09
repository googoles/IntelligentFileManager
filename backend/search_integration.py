"""
Integration utilities for semantic search with other system components.

This module provides integration points between the semantic search functionality
and other components like the file watcher, database operations, and background tasks.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import time

from database import File, Project, db_session, get_file_by_id
from search import get_semantic_search, SemanticSearch
from config import config

# Configure logging
logger = logging.getLogger(__name__)


class SearchIndexManager:
    """
    Manages semantic search indexing operations and integrations.
    
    This class provides high-level operations for managing the search index,
    including automatic indexing, background processing, and maintenance tasks.
    """
    
    def __init__(self):
        """Initialize the search index manager."""
        self.search_engine: Optional[SemanticSearch] = None
        self._indexing_queue: List[str] = []  # File IDs to index
        self._processing = False
        
    def get_search_engine(self) -> SemanticSearch:
        """Get or initialize the search engine."""
        if self.search_engine is None:
            self.search_engine = get_semantic_search()
        return self.search_engine
    
    async def index_file_async(self, file_id: str, force_reindex: bool = False) -> bool:
        """
        Asynchronously index a single file.
        
        Args:
            file_id: ID of the file to index
            force_reindex: Whether to reindex if already indexed
            
        Returns:
            True if indexing was successful
        """
        try:
            search_engine = self.get_search_engine()
            
            with db_session() as session:
                file_record = get_file_by_id(session, file_id)
                if not file_record:
                    logger.warning(f"File {file_id} not found for indexing")
                    return False
                
                # Check if file has content to index
                if not file_record.content or not file_record.content.strip():
                    logger.debug(f"File {file_record.name} has no content to index")
                    return True
                
                # Perform indexing
                success = search_engine.index_file(file_record, force_reindex)
                
                if success:
                    logger.info(f"Successfully indexed file: {file_record.name}")
                else:
                    logger.error(f"Failed to index file: {file_record.name}")
                
                return success
                
        except Exception as e:
            logger.error(f"Error indexing file {file_id}: {e}")
            return False
    
    def queue_file_for_indexing(self, file_id: str) -> None:
        """
        Add a file to the indexing queue for background processing.
        
        Args:
            file_id: ID of the file to queue for indexing
        """
        if file_id not in self._indexing_queue:
            self._indexing_queue.append(file_id)
            logger.debug(f"Queued file {file_id} for indexing")
    
    async def process_indexing_queue(self, batch_size: int = 10) -> Dict[str, Any]:
        """
        Process the indexing queue in batches.
        
        Args:
            batch_size: Number of files to process in each batch
            
        Returns:
            Processing statistics
        """
        if self._processing:
            logger.warning("Indexing queue is already being processed")
            return {"status": "already_processing"}
        
        self._processing = True
        start_time = time.time()
        successful = 0
        failed = 0
        
        try:
            logger.info(f"Processing indexing queue: {len(self._indexing_queue)} files")
            
            while self._indexing_queue:
                # Process batch
                batch = self._indexing_queue[:batch_size]
                self._indexing_queue = self._indexing_queue[batch_size:]
                
                # Process files in batch
                tasks = [
                    self.index_file_async(file_id) 
                    for file_id in batch
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count results
                for result in results:
                    if isinstance(result, Exception):
                        failed += 1
                        logger.error(f"Indexing task failed: {result}")
                    elif result:
                        successful += 1
                    else:
                        failed += 1
                
                # Small delay between batches to prevent overwhelming the system
                if self._indexing_queue:
                    await asyncio.sleep(0.1)
            
            processing_time = time.time() - start_time
            
            stats = {
                "status": "completed",
                "successful": successful,
                "failed": failed,
                "processing_time": processing_time,
                "files_per_second": (successful + failed) / processing_time if processing_time > 0 else 0
            }
            
            logger.info(f"Queue processing completed: {stats}")
            return stats
            
        finally:
            self._processing = False
    
    def handle_file_created(self, file_path: str) -> None:
        """
        Handle file creation event from file watcher.
        
        Args:
            file_path: Path of the created file
        """
        try:
            with db_session() as session:
                # Find file record by path
                file_record = session.query(File).filter(File.path == file_path).first()
                
                if file_record:
                    logger.info(f"File created: {file_path}, queuing for indexing")
                    self.queue_file_for_indexing(file_record.id)
                else:
                    logger.debug(f"File {file_path} not found in database")
                    
        except Exception as e:
            logger.error(f"Error handling file creation {file_path}: {e}")
    
    def handle_file_modified(self, file_path: str) -> None:
        """
        Handle file modification event from file watcher.
        
        Args:
            file_path: Path of the modified file
        """
        try:
            with db_session() as session:
                file_record = session.query(File).filter(File.path == file_path).first()
                
                if file_record:
                    logger.info(f"File modified: {file_path}, queuing for reindexing")
                    self.queue_file_for_indexing(file_record.id)
                    
        except Exception as e:
            logger.error(f"Error handling file modification {file_path}: {e}")
    
    def handle_file_deleted(self, file_path: str) -> None:
        """
        Handle file deletion event from file watcher.
        
        Args:
            file_path: Path of the deleted file
        """
        try:
            with db_session() as session:
                file_record = session.query(File).filter(File.path == file_path).first()
                
                if file_record:
                    logger.info(f"File deleted: {file_path}, removing embeddings")
                    search_engine = self.get_search_engine()
                    search_engine._remove_file_embeddings(file_record.id)
                    
        except Exception as e:
            logger.error(f"Error handling file deletion {file_path}: {e}")
    
    async def reindex_project(self, project_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Reindex all files in a project.
        
        Args:
            project_id: ID of the project to reindex
            force: Whether to force reindexing of existing files
            
        Returns:
            Reindexing statistics
        """
        try:
            search_engine = self.get_search_engine()
            
            with db_session() as session:
                # Get project
                project = session.query(Project).filter(Project.id == project_id).first()
                if not project:
                    raise ValueError(f"Project {project_id} not found")
                
                # Get all files with content
                files = session.query(File).filter(
                    File.project_id == project_id,
                    File.content.isnot(None)
                ).all()
                
                logger.info(f"Reindexing {len(files)} files for project: {project.name}")
                
                # Batch index files
                stats = search_engine.batch_index_files(files, force_reindex=force)
                
                return stats
                
        except Exception as e:
            logger.error(f"Error reindexing project {project_id}: {e}")
            return {"error": str(e)}
    
    def get_indexing_status(self) -> Dict[str, Any]:
        """
        Get current indexing status and statistics.
        
        Returns:
            Indexing status information
        """
        try:
            search_engine = self.get_search_engine()
            collection_stats = search_engine.get_collection_stats()
            
            return {
                "queue_length": len(self._indexing_queue),
                "processing": self._processing,
                "collection_stats": collection_stats,
                "next_files": self._indexing_queue[:5] if self._indexing_queue else []
            }
            
        except Exception as e:
            logger.error(f"Error getting indexing status: {e}")
            return {"error": str(e)}


# Global search index manager instance
search_index_manager: Optional[SearchIndexManager] = None


def get_search_index_manager() -> SearchIndexManager:
    """
    Get the global search index manager instance.
    
    Returns:
        SearchIndexManager instance
    """
    global search_index_manager
    
    if search_index_manager is None:
        search_index_manager = SearchIndexManager()
        logger.info("Search index manager initialized")
    
    return search_index_manager


# Integration functions for file watcher
def on_file_created(file_path: str) -> None:
    """
    File watcher callback for file creation events.
    
    Args:
        file_path: Path of the created file
    """
    manager = get_search_index_manager()
    manager.handle_file_created(file_path)


def on_file_modified(file_path: str) -> None:
    """
    File watcher callback for file modification events.
    
    Args:
        file_path: Path of the modified file
    """
    manager = get_search_index_manager()
    manager.handle_file_modified(file_path)


def on_file_deleted(file_path: str) -> None:
    """
    File watcher callback for file deletion events.
    
    Args:
        file_path: Path of the deleted file
    """
    manager = get_search_index_manager()
    manager.handle_file_deleted(file_path)


# Background task functions
async def periodic_queue_processing(interval: int = 30) -> None:
    """
    Periodically process the indexing queue.
    
    Args:
        interval: Processing interval in seconds
    """
    manager = get_search_index_manager()
    
    while True:
        try:
            if manager._indexing_queue and not manager._processing:
                logger.info("Processing indexing queue (periodic)")
                await manager.process_indexing_queue()
            
            await asyncio.sleep(interval)
            
        except asyncio.CancelledError:
            logger.info("Periodic queue processing cancelled")
            break
        except Exception as e:
            logger.error(f"Error in periodic queue processing: {e}")
            await asyncio.sleep(interval)


async def maintenance_tasks() -> None:
    """
    Perform periodic maintenance tasks for the search system.
    """
    logger.info("Starting search system maintenance tasks")
    
    try:
        manager = get_search_index_manager()
        search_engine = manager.get_search_engine()
        
        # Get collection statistics
        stats = search_engine.get_collection_stats()
        logger.info(f"Search collection stats: {stats}")
        
        # TODO: Add other maintenance tasks like:
        # - Cleanup orphaned embeddings
        # - Optimize collection performance
        # - Generate search usage reports
        
    except Exception as e:
        logger.error(f"Error in maintenance tasks: {e}")


# Utility functions for application startup
async def initialize_search_system() -> bool:
    """
    Initialize the search system and perform startup tasks.
    
    Returns:
        True if initialization was successful
    """
    try:
        logger.info("Initializing search system...")
        
        # Initialize search index manager
        manager = get_search_index_manager()
        search_engine = manager.get_search_engine()
        
        # Get initial statistics
        stats = search_engine.get_collection_stats()
        logger.info(f"Search system initialized with {stats.get('total_embeddings', 0)} embeddings")
        
        # Start background tasks if running in async context
        try:
            # Start periodic queue processing
            asyncio.create_task(periodic_queue_processing())
            logger.info("Background indexing task started")
        except RuntimeError:
            logger.info("Background tasks will be started by the main application")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize search system: {e}")
        return False


if __name__ == "__main__":
    # Example usage and testing
    async def test_integration():
        """Test the search integration functionality."""
        print("Testing search integration...")
        
        # Initialize system
        success = await initialize_search_system()
        print(f"System initialization: {'✅' if success else '❌'}")
        
        # Test manager
        manager = get_search_index_manager()
        status = manager.get_indexing_status()
        print(f"Initial status: {status}")
        
        # Simulate file events
        manager.handle_file_created("/test/file1.txt")
        manager.handle_file_created("/test/file2.txt")
        
        status = manager.get_indexing_status()
        print(f"After queueing files: {status}")
        
        print("Integration test completed")
    
    # Run test
    asyncio.run(test_integration())