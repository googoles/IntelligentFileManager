#!/usr/bin/env python3
"""
Simplified Semantic Search for Windows - Alternative to ChromaDB

This module provides semantic search functionality using numpy and sklearn
instead of ChromaDB, avoiding Rust compilation issues on Windows.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with metadata"""
    file_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]


class SemanticSearch:
    """
    Simplified semantic search implementation using numpy arrays
    instead of ChromaDB for better Windows compatibility.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic search with sentence transformer model"""
        self.model_name = model_name
        self.model = None
        
        # Ensure data directories exist
        Path("data/db").mkdir(parents=True, exist_ok=True)
        self.embeddings_dir = Path("data/db/embeddings")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.embeddings_file = self.embeddings_dir / "embeddings.npy"
        self.metadata_file = self.embeddings_dir / "metadata.json"
        self.index_file = self.embeddings_dir / "index.json"
        
        # In-memory storage
        self.embeddings = None
        self.metadata = []
        self.index = {}  # Maps file_id to indices
        
        self._load_model()
        self._load_data()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_data(self):
        """Load existing embeddings and metadata from disk"""
        try:
            if self.embeddings_file.exists():
                self.embeddings = np.load(self.embeddings_file)
                logger.info(f"Loaded {len(self.embeddings)} embeddings")
            else:
                self.embeddings = np.array([]).reshape(0, 384)  # Empty array with correct shape
                
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
                    
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")
            self.embeddings = np.array([]).reshape(0, 384)
            self.metadata = []
            self.index = {}
    
    def _save_data(self):
        """Save embeddings and metadata to disk"""
        try:
            if len(self.embeddings) > 0:
                np.save(self.embeddings_file, self.embeddings)
                
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
                
            logger.info("Saved embeddings and metadata to disk")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks for processing"""
        if not text:
            return []
            
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                # Keep 25% overlap
                overlap_words = len(current_chunk) // 4
                current_chunk = current_chunk[-overlap_words:]
                current_size = sum(len(w) + 1 for w in current_chunk)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    async def index_file(self, file_id: str, content: str, metadata: Dict[str, Any]):
        """Index a file's content for semantic search"""
        try:
            if not content:
                logger.warning(f"No content to index for file {file_id}")
                return
            
            # Remove existing entries for this file
            if file_id in self.index:
                old_indices = self.index[file_id]
                # Remove old embeddings and metadata
                mask = np.ones(len(self.embeddings), dtype=bool)
                for idx in old_indices:
                    mask[idx] = False
                self.embeddings = self.embeddings[mask]
                # Update metadata list
                self.metadata = [m for i, m in enumerate(self.metadata) if mask[i]]
                # Rebuild index
                self._rebuild_index()
            
            # Split content into chunks
            chunks = self._split_text(content)
            if not chunks:
                return
            
            # Generate embeddings for chunks
            chunk_embeddings = self.model.encode(chunks)
            
            # Add to storage
            start_idx = len(self.embeddings)
            new_indices = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                # Add embedding
                if len(self.embeddings) == 0:
                    self.embeddings = embedding.reshape(1, -1)
                else:
                    self.embeddings = np.vstack([self.embeddings, embedding])
                
                # Add metadata
                chunk_metadata = {
                    'file_id': file_id,
                    'chunk_index': i,
                    'content': chunk[:500],  # Store first 500 chars
                    **metadata
                }
                self.metadata.append(chunk_metadata)
                new_indices.append(start_idx + i)
            
            # Update index
            self.index[file_id] = new_indices
            
            # Save to disk
            self._save_data()
            
            logger.info(f"Indexed {len(chunks)} chunks for file {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to index file {file_id}: {e}")
    
    def _rebuild_index(self):
        """Rebuild the index mapping after removing entries"""
        self.index = {}
        for i, meta in enumerate(self.metadata):
            file_id = meta.get('file_id')
            if file_id:
                if file_id not in self.index:
                    self.index[file_id] = []
                self.index[file_id].append(i)
    
    async def search(
        self, 
        query: str, 
        project_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search for similar documents using semantic similarity
        
        Args:
            query: Search query text
            project_id: Optional project ID to filter results
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            if len(self.embeddings) == 0:
                logger.warning("No documents indexed yet")
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embeddings
            )[0]
            
            # Filter by project if specified
            if project_id:
                filtered_indices = [
                    i for i, meta in enumerate(self.metadata)
                    if meta.get('project_id') == project_id
                ]
                if not filtered_indices:
                    return []
                # Create mask for filtering
                mask = np.zeros(len(similarities), dtype=bool)
                mask[filtered_indices] = True
                # Apply mask
                filtered_similarities = similarities[mask]
                filtered_metadata = [self.metadata[i] for i in filtered_indices]
            else:
                filtered_similarities = similarities
                filtered_metadata = self.metadata
            
            # Get top-k results
            top_indices = np.argsort(filtered_similarities)[-top_k:][::-1]
            
            # Create results
            results = []
            seen_files = set()
            
            for idx in top_indices:
                if filtered_similarities[idx] < 0.1:  # Minimum similarity threshold
                    continue
                    
                meta = filtered_metadata[idx]
                file_id = meta.get('file_id')
                
                # Skip duplicate files (only keep best match per file)
                if file_id in seen_files:
                    continue
                seen_files.add(file_id)
                
                results.append(SearchResult(
                    file_id=file_id,
                    content=meta.get('content', ''),
                    similarity=float(filtered_similarities[idx]),
                    metadata=meta
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def find_similar_files(
        self,
        file_id: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Find files similar to a given file"""
        try:
            if file_id not in self.index:
                logger.warning(f"File {file_id} not indexed")
                return []
            
            # Get embeddings for this file
            file_indices = self.index[file_id]
            if not file_indices:
                return []
            
            # Use the first chunk's embedding as representative
            file_embedding = self.embeddings[file_indices[0]]
            
            # Calculate similarities
            similarities = cosine_similarity(
                file_embedding.reshape(1, -1),
                self.embeddings
            )[0]
            
            # Get top-k results (excluding self)
            top_indices = np.argsort(similarities)[-top_k-10:][::-1]
            
            # Create results
            results = []
            seen_files = set([file_id])  # Exclude self
            
            for idx in top_indices:
                if len(results) >= top_k:
                    break
                    
                if similarities[idx] < 0.3:  # Minimum similarity
                    continue
                    
                meta = self.metadata[idx]
                result_file_id = meta.get('file_id')
                
                if result_file_id in seen_files:
                    continue
                seen_files.add(result_file_id)
                
                results.append(SearchResult(
                    file_id=result_file_id,
                    content=meta.get('content', ''),
                    similarity=float(similarities[idx]),
                    metadata=meta
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Find similar files failed: {e}")
            return []
    
    def clear_index(self):
        """Clear all indexed data"""
        self.embeddings = np.array([]).reshape(0, 384)
        self.metadata = []
        self.index = {}
        self._save_data()
        logger.info("Cleared all indexed data")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed data"""
        return {
            'total_embeddings': len(self.embeddings),
            'total_files': len(self.index),
            'total_metadata': len(self.metadata),
            'embedding_shape': self.embeddings.shape if len(self.embeddings) > 0 else None,
            'model_name': self.model_name
        }