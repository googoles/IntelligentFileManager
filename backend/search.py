"""
Semantic search functionality with embeddings for the Research File Manager.

This module implements semantic search using sentence-transformers and ChromaDB
for vector storage, enabling intelligent similarity-based file search and discovery.
"""

import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import uuid
from dataclasses import dataclass
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from sqlalchemy.orm import Session

from database import File, Project, get_db_session, db_session
from config import config

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Search result containing file information and similarity score.
    """
    file_id: str
    file_name: str
    file_path: str
    file_type: str
    project_id: str
    project_name: str
    content_snippet: str
    similarity_score: float
    chunk_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TextChunk:
    """
    Text chunk with metadata for embedding storage.
    """
    content: str
    file_id: str
    project_id: str
    chunk_index: int
    start_pos: int
    end_pos: int
    
    @property
    def chunk_id(self) -> str:
        """Generate unique ID for this chunk."""
        return f"{self.file_id}_{self.chunk_index}"


class SemanticSearch:
    """
    Semantic search engine using sentence-transformers and ChromaDB.
    
    Provides functionality for indexing file content as vector embeddings
    and performing similarity-based searches across documents.
    """
    
    def __init__(
        self,
        model_name: str = None,
        chroma_path: str = None,
        collection_name: str = "file_embeddings"
    ):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name: Name of the sentence-transformer model
            chroma_path: Path to ChromaDB persistence directory
            collection_name: Name of the ChromaDB collection
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.chroma_path = chroma_path or config.CHROMA_DB_PATH
        self.collection_name = collection_name
        self.chunk_size = config.CHUNK_SIZE
        
        # Initialize model and database
        self.model: Optional[SentenceTransformer] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.collection: Optional[Collection] = None
        
        # Performance tracking
        self._indexing_stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'processing_time': 0.0
        }
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the embedding model and ChromaDB."""
        try:
            # Load embedding model
            logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()
            
            self.model = SentenceTransformer(self.model_name)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            # Initialize ChromaDB
            self._init_chromadb()
            
        except Exception as e:
            logger.error(f"Error initializing semantic search: {e}")
            raise
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure ChromaDB directory exists
            chroma_dir = Path(self.chroma_path)
            chroma_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(
                    allow_reset=False,
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection (this handles both cases automatically)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "File content embeddings for semantic search"}
            )
            logger.info(f"Initialized collection: {self.collection_name}")
            
            # Get collection stats
            collection_count = self.collection.count()
            logger.info(f"Collection contains {collection_count} embeddings")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _split_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Text content to split
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        text = text.strip()
        chunk_size = self.chunk_size
        
        # Simple overlap strategy for better context preservation
        overlap = min(50, chunk_size // 4)  # 25% overlap
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Find end position
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_breaks = ['. ', '! ', '? ', '\n\n']
                best_break = -1
                
                for i in range(max(0, end - 100), end):
                    for break_char in sentence_breaks:
                        if text[i:i+len(break_char)] == break_char:
                            best_break = i + len(break_char)
                
                if best_break > start + chunk_size // 2:  # Ensure minimum chunk size
                    end = best_break
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunk = TextChunk(
                    content=chunk_content,
                    file_id="",  # Will be set when indexing
                    project_id="",  # Will be set when indexing
                    chunk_index=chunk_index,
                    start_pos=start,
                    end_pos=end
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            # Safety check to prevent infinite loops
            if start >= len(text):
                break
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            # Filter empty texts
            non_empty_texts = [text for text in texts if text and text.strip()]
            
            if not non_empty_texts:
                return np.array([])
            
            # Generate embeddings
            embeddings = self.model.encode(
                non_empty_texts,
                show_progress_bar=len(non_empty_texts) > 10,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def index_file(
        self, 
        file_record: File,
        force_reindex: bool = False
    ) -> bool:
        """
        Index a file by generating embeddings for its content chunks.
        
        Args:
            file_record: File database record
            force_reindex: Whether to reindex if already indexed
            
        Returns:
            True if indexing was successful
        """
        try:
            start_time = time.time()
            
            # Check if file already indexed (unless force_reindex)
            if not force_reindex:
                existing_count = len(self.collection.get(
                    where={"file_id": file_record.id}
                )["ids"])
                
                if existing_count > 0:
                    logger.debug(f"File {file_record.name} already indexed with {existing_count} chunks")
                    return True
            else:
                # Remove existing embeddings for this file
                self._remove_file_embeddings(file_record.id)
            
            # Check if file has content
            if not file_record.content or not file_record.content.strip():
                logger.debug(f"File {file_record.name} has no content to index")
                return True
            
            # Split content into chunks
            chunks = self._split_text(file_record.content)
            
            if not chunks:
                logger.debug(f"No chunks created for file {file_record.name}")
                return True
            
            # Set file and project IDs for chunks
            for chunk in chunks:
                chunk.file_id = file_record.id
                chunk.project_id = file_record.project_id
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self._generate_embeddings(chunk_texts)
            
            if len(embeddings) == 0:
                logger.warning(f"No embeddings generated for file {file_record.name}")
                return True
            
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "file_id": chunk.file_id,
                    "project_id": chunk.project_id,
                    "file_name": file_record.name,
                    "file_path": file_record.path,
                    "file_type": file_record.type,
                    "chunk_index": chunk.chunk_index,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                    "content_length": len(chunk.content),
                    "indexed_at": time.time()
                }
                
                # Add file metadata if available
                if file_record.metadata:
                    metadata.update(file_record.metadata)
                
                metadatas.append(metadata)
            
            # Store embeddings in ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=chunk_texts,
                ids=ids
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self._indexing_stats['files_processed'] += 1
            self._indexing_stats['chunks_created'] += len(chunks)
            self._indexing_stats['embeddings_generated'] += len(embeddings)
            self._indexing_stats['processing_time'] += processing_time
            
            logger.info(
                f"Indexed file {file_record.name}: {len(chunks)} chunks, "
                f"{len(embeddings)} embeddings in {processing_time:.2f}s"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing file {file_record.name}: {e}")
            return False
    
    def search(
        self,
        query: str,
        project_id: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Perform semantic search across indexed files.
        
        Args:
            query: Search query string
            project_id: Optional project ID to limit search scope
            file_types: Optional list of file types to filter
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        try:
            if not query or not query.strip():
                return []
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query.strip()])
            
            if len(query_embedding) == 0:
                return []
            
            # Build where filter
            where_filter = {}
            if project_id:
                where_filter["project_id"] = project_id
            
            if file_types:
                where_filter["file_type"] = {"$in": file_types}
            
            # Perform similarity search
            search_results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(top_k * 2, 100),  # Get more results for filtering
                where=where_filter if where_filter else None
            )
            
            # Process results
            results = []
            
            if not search_results["ids"] or not search_results["ids"][0]:
                return results
            
            # Get file information from database
            with db_session() as session:
                for i, chunk_id in enumerate(search_results["ids"][0]):
                    # Extract similarity score (ChromaDB returns distances, convert to similarity)
                    distance = search_results["distances"][0][i]
                    similarity = 1.0 - distance  # Convert distance to similarity
                    
                    if similarity < similarity_threshold:
                        continue
                    
                    metadata = search_results["metadatas"][0][i]
                    document = search_results["documents"][0][i]
                    
                    # Get file record for additional information
                    file_record = session.query(File).filter(
                        File.id == metadata["file_id"]
                    ).first()
                    
                    if not file_record:
                        continue
                    
                    # Get project name
                    project_name = "Unknown"
                    if file_record.project:
                        project_name = file_record.project.name
                    
                    # Create search result
                    result = SearchResult(
                        file_id=metadata["file_id"],
                        file_name=metadata["file_name"],
                        file_path=metadata["file_path"],
                        file_type=metadata["file_type"],
                        project_id=metadata["project_id"],
                        project_name=project_name,
                        content_snippet=document,
                        similarity_score=similarity,
                        chunk_index=metadata.get("chunk_index"),
                        metadata=metadata
                    )
                    
                    results.append(result)
            
            # Sort by similarity score and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []
    
    def find_similar_files(
        self,
        file_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        exclude_same_file: bool = True
    ) -> List[SearchResult]:
        """
        Find files similar to a given file based on content embeddings.
        
        Args:
            file_id: ID of the reference file
            top_k: Maximum number of similar files to return
            similarity_threshold: Minimum similarity score
            exclude_same_file: Whether to exclude the reference file from results
            
        Returns:
            List of SearchResult objects for similar files
        """
        try:
            # Get embeddings for the reference file
            file_embeddings = self.collection.get(
                where={"file_id": file_id}
            )
            
            if not file_embeddings["embeddings"]:
                logger.warning(f"No embeddings found for file {file_id}")
                return []
            
            # Use the first chunk's embedding as representative
            # (Could be improved by averaging all chunks)
            reference_embedding = file_embeddings["embeddings"][0]
            
            # Search for similar content
            search_results = self.collection.query(
                query_embeddings=[reference_embedding],
                n_results=top_k * 3,  # Get more results for filtering
            )
            
            # Process results
            results = []
            seen_files = set()
            
            if not search_results["ids"] or not search_results["ids"][0]:
                return results
            
            with db_session() as session:
                for i, chunk_id in enumerate(search_results["ids"][0]):
                    distance = search_results["distances"][0][i]
                    similarity = 1.0 - distance
                    
                    if similarity < similarity_threshold:
                        continue
                    
                    metadata = search_results["metadatas"][0][i]
                    document = search_results["documents"][0][i]
                    
                    current_file_id = metadata["file_id"]
                    
                    # Skip if same file (if requested)
                    if exclude_same_file and current_file_id == file_id:
                        continue
                    
                    # Skip if we've already included this file
                    if current_file_id in seen_files:
                        continue
                    
                    seen_files.add(current_file_id)
                    
                    # Get file record
                    file_record = session.query(File).filter(
                        File.id == current_file_id
                    ).first()
                    
                    if not file_record:
                        continue
                    
                    # Get project name
                    project_name = "Unknown"
                    if file_record.project:
                        project_name = file_record.project.name
                    
                    result = SearchResult(
                        file_id=current_file_id,
                        file_name=metadata["file_name"],
                        file_path=metadata["file_path"],
                        file_type=metadata["file_type"],
                        project_id=metadata["project_id"],
                        project_name=project_name,
                        content_snippet=document,
                        similarity_score=similarity,
                        chunk_index=metadata.get("chunk_index"),
                        metadata=metadata
                    )
                    
                    results.append(result)
                    
                    if len(results) >= top_k:
                        break
            
            # Sort by similarity
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar files: {e}")
            return []
    
    def batch_index_files(
        self,
        file_records: List[File],
        force_reindex: bool = False,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Index multiple files in batches for better performance.
        
        Args:
            file_records: List of File records to index
            force_reindex: Whether to reindex existing files
            batch_size: Number of files to process in each batch
            
        Returns:
            Dictionary with indexing statistics
        """
        start_time = time.time()
        successful = 0
        failed = 0
        
        logger.info(f"Starting batch indexing of {len(file_records)} files")
        
        for i in range(0, len(file_records), batch_size):
            batch = file_records[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_records) + batch_size - 1)//batch_size}")
            
            for file_record in batch:
                try:
                    if self.index_file(file_record, force_reindex):
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Failed to index {file_record.name}: {e}")
                    failed += 1
        
        total_time = time.time() - start_time
        
        stats = {
            'total_files': len(file_records),
            'successful': successful,
            'failed': failed,
            'processing_time': total_time,
            'files_per_second': len(file_records) / total_time if total_time > 0 else 0,
            'indexing_stats': self._indexing_stats.copy()
        }
        
        logger.info(
            f"Batch indexing completed: {successful} successful, {failed} failed "
            f"in {total_time:.2f}s ({stats['files_per_second']:.1f} files/sec)"
        )
        
        return stats
    
    def _remove_file_embeddings(self, file_id: str) -> bool:
        """
        Remove all embeddings for a specific file.
        
        Args:
            file_id: ID of the file to remove embeddings for
            
        Returns:
            True if embeddings were removed successfully
        """
        try:
            # Get all chunk IDs for this file
            existing_data = self.collection.get(
                where={"file_id": file_id}
            )
            
            if existing_data["ids"]:
                self.collection.delete(ids=existing_data["ids"])
                logger.debug(f"Removed {len(existing_data['ids'])} embeddings for file {file_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing embeddings for file {file_id}: {e}")
            return False
    
    def remove_project_embeddings(self, project_id: str) -> bool:
        """
        Remove all embeddings for a specific project.
        
        Args:
            project_id: ID of the project to remove embeddings for
            
        Returns:
            True if embeddings were removed successfully
        """
        try:
            existing_data = self.collection.get(
                where={"project_id": project_id}
            )
            
            if existing_data["ids"]:
                self.collection.delete(ids=existing_data["ids"])
                logger.info(f"Removed {len(existing_data['ids'])} embeddings for project {project_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing embeddings for project {project_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            total_embeddings = self.collection.count()
            
            # Get sample of metadata for analysis
            sample_data = self.collection.get(limit=100)
            
            stats = {
                'total_embeddings': total_embeddings,
                'collection_name': self.collection_name,
                'model_name': self.model_name,
                'chunk_size': self.chunk_size,
                'chroma_path': self.chroma_path,
                'indexing_stats': self._indexing_stats.copy()
            }
            
            # Analyze file types and projects if we have data
            if sample_data["metadatas"]:
                file_types = {}
                projects = {}
                
                for metadata in sample_data["metadatas"]:
                    file_type = metadata.get("file_type", "unknown")
                    project_id = metadata.get("project_id", "unknown")
                    
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    projects[project_id] = projects.get(project_id, 0) + 1
                
                stats['sample_file_types'] = file_types
                stats['sample_projects'] = len(projects)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def reset_collection(self) -> bool:
        """
        Reset the collection by deleting all embeddings.
        
        Returns:
            True if collection was reset successfully
        """
        try:
            # Delete the collection
            self.chroma_client.delete_collection(name=self.collection_name)
            
            # Recreate it
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "File content embeddings for semantic search"}
            )
            
            # Reset statistics
            self._indexing_stats = {
                'files_processed': 0,
                'chunks_created': 0,
                'embeddings_generated': 0,
                'processing_time': 0.0
            }
            
            logger.info(f"Collection {self.collection_name} reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False


# Global semantic search instance
semantic_search: Optional[SemanticSearch] = None


def get_semantic_search() -> SemanticSearch:
    """
    Get the global semantic search instance, initializing if necessary.
    
    Returns:
        SemanticSearch instance
    """
    global semantic_search
    
    if semantic_search is None:
        semantic_search = SemanticSearch()
    
    return semantic_search


def initialize_semantic_search(
    model_name: str = None,
    chroma_path: str = None,
    force_reinit: bool = False
) -> SemanticSearch:
    """
    Initialize the global semantic search instance.
    
    Args:
        model_name: Name of the sentence-transformer model
        chroma_path: Path to ChromaDB persistence directory
        force_reinit: Whether to force reinitialization
        
    Returns:
        SemanticSearch instance
    """
    global semantic_search
    
    if semantic_search is None or force_reinit:
        semantic_search = SemanticSearch(
            model_name=model_name,
            chroma_path=chroma_path
        )
        logger.info("Semantic search initialized successfully")
    
    return semantic_search


# Utility functions for integration
def index_project_files(
    project_id: str,
    force_reindex: bool = False
) -> Dict[str, Any]:
    """
    Index all files in a project.
    
    Args:
        project_id: ID of the project to index
        force_reindex: Whether to reindex existing files
        
    Returns:
        Dictionary with indexing statistics
    """
    search_engine = get_semantic_search()
    
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
        
        logger.info(f"Indexing {len(files)} files for project: {project.name}")
        
        return search_engine.batch_index_files(files, force_reindex)


def search_files(
    query: str,
    project_id: Optional[str] = None,
    file_types: Optional[List[str]] = None,
    top_k: int = 10
) -> List[SearchResult]:
    """
    Convenience function for semantic file search.
    
    Args:
        query: Search query
        project_id: Optional project ID filter
        file_types: Optional file type filter
        top_k: Maximum results to return
        
    Returns:
        List of search results
    """
    search_engine = get_semantic_search()
    return search_engine.search(
        query=query,
        project_id=project_id,
        file_types=file_types,
        top_k=top_k
    )


if __name__ == "__main__":
    # Example usage and testing
    import sys
    from database import init_database, create_project, create_file
    
    print("Initializing semantic search system...")
    
    # Initialize database
    init_database()
    
    # Initialize semantic search
    search_engine = initialize_semantic_search()
    
    # Test with sample data
    with db_session() as session:
        # Create test project
        try:
            project = create_project(
                session,
                name="Test Semantic Search",
                path="./test_semantic_search"
            )
            
            # Create sample files with different content
            sample_files = [
                {
                    "name": "machine_learning.txt",
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
                },
                {
                    "name": "deep_learning.txt", 
                    "content": "Deep learning is a branch of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."
                },
                {
                    "name": "natural_language.txt",
                    "content": "Natural language processing involves the interaction between computers and humans using natural language to analyze, understand and generate human language."
                }
            ]
            
            file_records = []
            for file_data in sample_files:
                file_record = create_file(
                    session,
                    project_id=project.id,
                    path=f"./test_semantic_search/{file_data['name']}",
                    name=file_data["name"],
                    file_type=".txt",
                    content=file_data["content"]
                )
                file_records.append(file_record)
            
            # Index files
            print("Indexing sample files...")
            stats = search_engine.batch_index_files(file_records)
            print(f"Indexing completed: {stats}")
            
            # Test searches
            test_queries = [
                "artificial intelligence",
                "neural networks",
                "text analysis",
                "learning algorithms"
            ]
            
            print("\nTesting semantic search:")
            for query in test_queries:
                results = search_engine.search(query, top_k=3)
                print(f"\nQuery: '{query}'")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.file_name} (similarity: {result.similarity_score:.3f})")
                    print(f"     Snippet: {result.content_snippet[:100]}...")
            
            # Test similar files
            if file_records:
                print(f"\nFinding files similar to {file_records[0].name}:")
                similar = search_engine.find_similar_files(file_records[0].id, top_k=2)
                for result in similar:
                    print(f"  {result.file_name} (similarity: {result.similarity_score:.3f})")
            
            # Show collection stats
            print(f"\nCollection statistics:")
            collection_stats = search_engine.get_collection_stats()
            for key, value in collection_stats.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"Error during testing: {e}")
            sys.exit(1)
    
    print("\nSemantic search system test completed successfully!")