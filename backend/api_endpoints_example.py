"""
Example FastAPI endpoints for semantic search integration.

This module shows how to integrate the semantic search functionality
with FastAPI endpoints for the Research File Manager web interface.
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db_session, File, Project
from search import (
    get_semantic_search, 
    search_files, 
    index_project_files,
    SemanticSearch,
    SearchResult
)
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, description="Search query string")
    project_id: Optional[str] = Field(None, description="Optional project ID filter")
    file_types: Optional[List[str]] = Field(None, description="Optional file type filters")
    top_k: int = Field(10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class SearchResultResponse(BaseModel):
    """Search result response model."""
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


class SearchResponse(BaseModel):
    """Search response wrapper."""
    results: List[SearchResultResponse]
    total_results: int
    query: str
    execution_time_ms: float


class IndexRequest(BaseModel):
    """File indexing request model."""
    project_id: Optional[str] = Field(None, description="Project ID to index")
    file_ids: Optional[List[str]] = Field(None, description="Specific file IDs to index") 
    force_reindex: bool = Field(False, description="Force reindexing of existing files")


class IndexResponse(BaseModel):
    """Indexing response model."""
    total_files: int
    successful: int
    failed: int
    processing_time: float
    files_per_second: float


class SimilarFilesRequest(BaseModel):
    """Similar files request model."""
    file_id: str = Field(..., description="Reference file ID")
    top_k: int = Field(5, ge=1, le=50, description="Maximum similar files to return")
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity score")
    exclude_same_file: bool = Field(True, description="Exclude the reference file from results")


class CollectionStatsResponse(BaseModel):
    """Collection statistics response."""
    total_embeddings: int
    collection_name: str
    model_name: str
    chunk_size: int
    indexing_stats: Dict[str, Any]


# FastAPI app instance
app = FastAPI(
    title="Research File Manager - Semantic Search API",
    description="Semantic search endpoints for intelligent file discovery",
    version="1.0.0"
)


# Dependency to get database session
def get_db() -> Session:
    """Get database session dependency."""
    session = get_db_session()
    try:
        yield session
    finally:
        session.close()


# Dependency to get semantic search engine
def get_search_engine() -> SemanticSearch:
    """Get semantic search engine dependency."""
    return get_semantic_search()


@app.post("/api/search", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    search_engine: SemanticSearch = Depends(get_search_engine)
) -> SearchResponse:
    """
    Perform semantic search across indexed files.
    
    Args:
        request: Search request parameters
        search_engine: Semantic search engine instance
        
    Returns:
        Search results with metadata
    """
    import time
    
    start_time = time.time()
    
    try:
        # Perform search
        results = search_engine.search(
            query=request.query,
            project_id=request.project_id,
            file_types=request.file_types,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # Convert to response format
        result_responses = [
            SearchResultResponse(
                file_id=result.file_id,
                file_name=result.file_name,
                file_path=result.file_path,
                file_type=result.file_type,
                project_id=result.project_id,
                project_name=result.project_name,
                content_snippet=result.content_snippet,
                similarity_score=result.similarity_score,
                chunk_index=result.chunk_index,
                metadata=result.metadata
            )
            for result in results
        ]
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return SearchResponse(
            results=result_responses,
            total_results=len(result_responses),
            query=request.query,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"Search error for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/search", response_model=SearchResponse)
async def semantic_search_get(
    query: str = Query(..., description="Search query string"),
    project_id: Optional[str] = Query(None, description="Optional project ID filter"),
    top_k: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    similarity_threshold: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity score"),
    search_engine: SemanticSearch = Depends(get_search_engine)
) -> SearchResponse:
    """
    GET endpoint for semantic search (convenience method).
    
    Args:
        query: Search query string
        project_id: Optional project ID filter
        top_k: Maximum number of results
        similarity_threshold: Minimum similarity score
        search_engine: Semantic search engine instance
        
    Returns:
        Search results with metadata
    """
    request = SearchRequest(
        query=query,
        project_id=project_id,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
    
    return await semantic_search(request, search_engine)


@app.post("/api/similar-files", response_model=SearchResponse)
async def find_similar_files(
    request: SimilarFilesRequest,
    search_engine: SemanticSearch = Depends(get_search_engine),
    db: Session = Depends(get_db)
) -> SearchResponse:
    """
    Find files similar to a reference file.
    
    Args:
        request: Similar files request parameters
        search_engine: Semantic search engine instance
        db: Database session
        
    Returns:
        Similar files with similarity scores
    """
    import time
    
    start_time = time.time()
    
    try:
        # Verify file exists
        file_record = db.query(File).filter(File.id == request.file_id).first()
        if not file_record:
            raise HTTPException(status_code=404, detail=f"File {request.file_id} not found")
        
        # Find similar files
        results = search_engine.find_similar_files(
            file_id=request.file_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            exclude_same_file=request.exclude_same_file
        )
        
        # Convert to response format
        result_responses = [
            SearchResultResponse(
                file_id=result.file_id,
                file_name=result.file_name,
                file_path=result.file_path,
                file_type=result.file_type,
                project_id=result.project_id,
                project_name=result.project_name,
                content_snippet=result.content_snippet,
                similarity_score=result.similarity_score,
                chunk_index=result.chunk_index,
                metadata=result.metadata
            )
            for result in results
        ]
        
        execution_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=result_responses,
            total_results=len(result_responses),
            query=f"Similar to: {file_record.name}",
            execution_time_ms=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar files error for file {request.file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Similar files search failed: {str(e)}")


@app.post("/api/index", response_model=IndexResponse)
async def index_files(
    request: IndexRequest,
    search_engine: SemanticSearch = Depends(get_search_engine),
    db: Session = Depends(get_db)
) -> IndexResponse:
    """
    Index files for semantic search.
    
    Args:
        request: Indexing request parameters
        search_engine: Semantic search engine instance
        db: Database session
        
    Returns:
        Indexing statistics and results
    """
    try:
        if request.project_id:
            # Index entire project
            stats = index_project_files(
                project_id=request.project_id,
                force_reindex=request.force_reindex
            )
            
        elif request.file_ids:
            # Index specific files
            file_records = db.query(File).filter(
                File.id.in_(request.file_ids)
            ).all()
            
            if not file_records:
                raise HTTPException(status_code=404, detail="No files found with provided IDs")
            
            stats = search_engine.batch_index_files(
                file_records=file_records,
                force_reindex=request.force_reindex
            )
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Must provide either project_id or file_ids"
            )
        
        return IndexResponse(
            total_files=stats['total_files'],
            successful=stats['successful'],
            failed=stats['failed'],
            processing_time=stats['processing_time'],
            files_per_second=stats['files_per_second']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.get("/api/stats", response_model=CollectionStatsResponse)
async def get_collection_stats(
    search_engine: SemanticSearch = Depends(get_search_engine)
) -> CollectionStatsResponse:
    """
    Get semantic search collection statistics.
    
    Args:
        search_engine: Semantic search engine instance
        
    Returns:
        Collection statistics and metadata
    """
    try:
        stats = search_engine.get_collection_stats()
        
        return CollectionStatsResponse(
            total_embeddings=stats.get('total_embeddings', 0),
            collection_name=stats.get('collection_name', 'unknown'),
            model_name=stats.get('model_name', 'unknown'),
            chunk_size=stats.get('chunk_size', 0),
            indexing_stats=stats.get('indexing_stats', {})
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.delete("/api/index/project/{project_id}")
async def remove_project_embeddings(
    project_id: str,
    search_engine: SemanticSearch = Depends(get_search_engine),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Remove all embeddings for a project.
    
    Args:
        project_id: Project ID to remove embeddings for
        search_engine: Semantic search engine instance
        db: Database session
        
    Returns:
        Operation result
    """
    try:
        # Verify project exists
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        # Remove embeddings
        success = search_engine.remove_project_embeddings(project_id)
        
        if success:
            return {
                "message": f"Successfully removed embeddings for project {project.name}",
                "project_id": project_id
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Failed to remove project embeddings"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing project embeddings: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to remove embeddings: {str(e)}"
        )


@app.post("/api/index/reset")
async def reset_search_collection(
    search_engine: SemanticSearch = Depends(get_search_engine)
) -> Dict[str, str]:
    """
    Reset the entire search collection (WARNING: Destructive operation).
    
    Args:
        search_engine: Semantic search engine instance
        
    Returns:
        Operation result
    """
    try:
        success = search_engine.reset_collection()
        
        if success:
            return {"message": "Search collection reset successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset collection")
            
    except Exception as e:
        logger.error(f"Error resetting collection: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# Health check endpoint
@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    try:
        # Test search engine initialization
        search_engine = get_semantic_search()
        stats = search_engine.get_collection_stats()
        
        return {
            "status": "healthy",
            "service": "semantic-search",
            "embeddings_count": str(stats.get('total_embeddings', 0))
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


if __name__ == "__main__":
    # Example usage for testing endpoints locally
    import uvicorn
    
    print("Starting FastAPI development server...")
    print("Semantic Search API endpoints:")
    print("- POST /api/search - Perform semantic search")
    print("- GET  /api/search - Perform semantic search (query params)")
    print("- POST /api/similar-files - Find similar files")
    print("- POST /api/index - Index files")
    print("- GET  /api/stats - Get collection statistics") 
    print("- DELETE /api/index/project/{id} - Remove project embeddings")
    print("- POST /api/index/reset - Reset collection (destructive)")
    print("- GET  /api/health - Health check")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )