#!/usr/bin/env python3
"""
Research File Manager MVP - FastAPI Main Application

This is the main FastAPI application that provides REST API endpoints
for the Research File Manager system.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File as FastFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exception_handlers import http_exception_handler
from pydantic import BaseModel, Field
import uvicorn

# Import our backend modules
try:
    from database import (
        init_database, db_session, create_project, create_file,
        Project, File, get_all_projects, get_project_files,
        get_project_by_id, delete_project
    )
    from file_watcher import FileWatcherManager
    from organizer import FileOrganizer, create_project_structure
    from llm_service import LocalLLMService, LLMConfig, get_llm_service
    from config import config
    
    # Import OCR processor
    try:
        from ocr_processor import get_ocr_processor, is_ocr_available, get_ocr_capabilities
        OCR_AVAILABLE = True
    except ImportError:
        get_ocr_processor = None
        is_ocr_available = lambda: False
        get_ocr_capabilities = lambda: {"ocr_available": False, "error": "OCR not available"}
        OCR_AVAILABLE = False
    
    # Try to import ChromaDB-based search, fall back to simple search for Windows
    try:
        from search import SemanticSearch
        print("Using ChromaDB for semantic search")
    except ImportError:
        from search_simple import SemanticSearch
        print("Using simplified semantic search (Windows-compatible)")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all backend modules are available")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global managers
file_watcher_manager = None
semantic_search = None
file_organizer = None
llm_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global file_watcher_manager, semantic_search, file_organizer, llm_service
    
    logger.info("🚀 Starting Research File Manager MVP...")
    
    # Create required directories
    os.makedirs("data/projects", exist_ok=True)
    os.makedirs("data/db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize database
    try:
        init_database()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
    
    # Initialize global managers
    try:
        file_watcher_manager = FileWatcherManager()
        semantic_search = SemanticSearch()
        file_organizer = FileOrganizer()
        
        # Initialize LLM service with configuration
        llm_config = LLMConfig()
        llm_config.enabled = config.LLM_ENABLED
        llm_config.model_name = config.LLM_MODEL_NAME
        llm_config.timeout = config.LLM_TIMEOUT
        llm_config.max_context_length = config.LLM_MAX_CONTEXT_LENGTH
        llm_config.temperature = config.LLM_TEMPERATURE
        llm_config.fallback_enabled = config.LLM_FALLBACK_ENABLED
        llm_config.max_retries = config.LLM_MAX_RETRIES
        llm_config.chunk_size = config.LLM_CHUNK_SIZE
        
        llm_service = LocalLLMService(llm_config)
        
        logger.info("✅ Core services initialized")
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise
    
    # Start existing project watchers
    try:
        with db_session() as session:
            projects = get_all_projects(session)
            for project in projects:
                if Path(project.path).exists():
                    file_watcher_manager.start_watching(project.id, project.path)
                    logger.info(f"✅ Started watching project: {project.name}")
    except Exception as e:
        logger.error(f"⚠️  Failed to start existing watchers: {e}")
    
    logger.info("🎉 Research File Manager MVP is ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Research File Manager...")
    if file_watcher_manager:
        file_watcher_manager.stop_all()
    logger.info("✅ Shutdown complete")

# FastAPI app
app = FastAPI(
    title="Research File Manager MVP",
    description="AI-Powered File Organization & Semantic Search for Researchers",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    path: str = Field(..., min_length=1, description="Project directory path")
    template: Optional[str] = Field("research", description="Project template (research, minimal, data_science, software_dev)")

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    project_id: Optional[str] = Field(None, description="Filter by project ID")
    top_k: Optional[int] = Field(5, ge=1, le=50, description="Number of results")

class OrganizeRequest(BaseModel):
    project_id: str = Field(..., description="Project ID to organize")
    auto_move: Optional[bool] = Field(False, description="Automatically move files")

class OCRExtractRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Path to file for OCR processing")
    max_pages: Optional[int] = Field(20, ge=1, le=50, description="Maximum pages to process for PDFs")

class OCRStatusRequest(BaseModel):
    file_id: str = Field(..., description="File ID to check OCR status")

class ProjectResponse(BaseModel):
    id: str
    name: str
    path: str
    created_at: str
    file_count: int = 0

class FileResponse(BaseModel):
    id: str
    name: str
    type: str
    path: str
    size: int
    created_at: str

class SearchResult(BaseModel):
    file: FileResponse
    snippet: str
    score: float
    summary: Optional[str] = Field(None, description="AI-generated summary if available")
    llm_available: bool = Field(default=False, description="Whether LLM features are available for this result")

# LLM-related models
class SummarizeRequest(BaseModel):
    file_id: str = Field(..., description="File ID to summarize")
    content: Optional[str] = Field(None, description="Optional content override")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask about the file")
    file_id: str = Field(..., description="File ID to query")
    content: Optional[str] = Field(None, description="Optional content override")

class OrganizationSuggestionRequest(BaseModel):
    file_name: str = Field(..., description="Name of the file")
    content: Optional[str] = Field(None, description="File content for analysis")
    file_type: Optional[str] = Field(None, description="File extension")

class LLMResponse(BaseModel):
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    service_available: bool

# Error handlers
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return await http_exception_handler(request, exc)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc} - {request.url}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML page"""
    try:
        frontend_path = Path("frontend/index.html")
        if frontend_path.exists():
            return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
        else:
            return HTMLResponse(content="""
            <html>
                <head><title>Research File Manager MVP</title></head>
                <body>
                    <h1>🔬 Research File Manager MVP</h1>
                    <p>Frontend not found. Please create frontend/index.html</p>
                    <p>API Documentation: <a href="/docs">/docs</a></p>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve frontend")

@app.post("/projects", response_model=ProjectResponse)
async def create_project_endpoint(project: ProjectCreate, background_tasks: BackgroundTasks):
    """Create a new research project"""
    try:
        # Validate and create directory
        project_path = Path(project.path).resolve()
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create project structure
        created_folders = create_project_structure(str(project_path), project.template)
        
        # Save to database
        with db_session() as session:
            db_project = create_project(
                session,
                name=project.name,
                path=str(project_path),
                ontology={"template": project.template, "folders": created_folders}
            )
            
            # Start file watcher
            if file_watcher_manager:
                file_watcher_manager.start_watching(db_project.id, str(project_path))
            
            # Schedule background indexing
            background_tasks.add_task(index_existing_files, str(project_path), db_project.id)
            
            return ProjectResponse(
                id=db_project.id,
                name=db_project.name,
                path=db_project.path,
                created_at=db_project.created_at.isoformat(),
                file_count=0
            )
            
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to create project: {str(e)}")

@app.get("/projects", response_model=List[ProjectResponse])
async def list_projects():
    """List all projects"""
    try:
        with db_session() as session:
            projects = get_all_projects(session)
            
            result = []
            for project in projects:
                files = get_project_files(session, project.id)
                result.append(ProjectResponse(
                    id=project.id,
                    name=project.name,
                    path=project.path,
                    created_at=project.created_at.isoformat(),
                    file_count=len(files)
                ))
            
            return result
            
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve projects")

@app.get("/projects/{project_id}/files", response_model=List[FileResponse])
async def list_project_files(project_id: str):
    """List files in a project"""
    try:
        with db_session() as session:
            project = get_project_by_id(session, project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            files = get_project_files(session, project_id)
            
            return [
                FileResponse(
                    id=f.id,
                    name=f.name,
                    type=f.type,
                    path=f.path,
                    size=f.file_metadata.get('size', 0) if f.file_metadata else 0,
                    created_at=f.created_at.isoformat()
                )
                for f in files
            ]
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve files")

@app.delete("/projects/{project_id}")
async def delete_project_endpoint(project_id: str):
    """Delete a project and all its associated files"""
    try:
        with db_session() as session:
            # Check if project exists
            project = get_project_by_id(session, project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            project_name = project.name
            
            # Stop file watcher for this project if it's running
            if file_watcher_manager:
                try:
                    file_watcher_manager.stop_watching_project(project_id)
                    logger.info(f"Stopped file watcher for project: {project_name}")
                except Exception as e:
                    logger.warning(f"Failed to stop file watcher for project {project_name}: {e}")
            
            # Delete the project from database (files will cascade delete due to foreign key)
            success = delete_project(session, project_id)
            
            if not success:
                # This shouldn't happen since we checked existence above, but handle it
                raise HTTPException(status_code=404, detail="Project not found")
            
            logger.info(f"Successfully deleted project: {project_name} ({project_id})")
            
            return {
                "message": f"Project '{project_name}' deleted successfully",
                "project_id": project_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_files(query: SearchQuery):
    """Perform semantic search across files"""
    try:
        if not semantic_search:
            raise HTTPException(status_code=503, detail="Search service not available")
        
        # Perform search
        results = await semantic_search.search(
            query.query,
            project_id=query.project_id,
            top_k=query.top_k
        )
        
        # Format results with optional LLM enhancements
        formatted_results = []
        with db_session() as session:
            for result in results:
                file_record = session.query(File).filter(File.id == result.file_id).first()
                if file_record:
                    # Check if we should generate a summary for this result
                    summary = None
                    llm_available = llm_service and llm_service.is_available
                    
                    # Generate quick summary for top results if LLM is available
                    if (llm_available and 
                        result.similarity > 0.7 and  # Only for high-relevance results
                        file_record.content and 
                        len(file_record.content.strip()) > 100):  # Only for substantial content
                        
                        try:
                            # Quick summary with shorter context
                            summary_result = await llm_service.summarize_content(
                                result.content[:1000],  # Use search result content, truncated
                                {
                                    'file_name': file_record.name,
                                    'file_type': file_record.type
                                }
                            )
                            if summary_result.get('summary'):
                                summary = summary_result['summary']
                        except Exception as e:
                            logger.debug(f"Quick summary generation failed for file {file_record.id}: {e}")
                            # Don't fail the search if summary fails
                    
                    formatted_results.append(SearchResult(
                        file=FileResponse(
                            id=file_record.id,
                            name=file_record.name,
                            type=file_record.type,
                            path=file_record.path,
                            size=file_record.metadata.get('size', 0) if file_record.metadata else 0,
                            created_at=file_record.created_at.isoformat()
                        ),
                        snippet=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                        score=result.similarity,
                        summary=summary,
                        llm_available=llm_available
                    ))
        
        return formatted_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search operation failed")

@app.post("/organize")
async def organize_project_files(request: OrganizeRequest):
    """Auto-organize files in a project"""
    try:
        with db_session() as session:
            project = get_project_by_id(session, request.project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            if not file_organizer:
                raise HTTPException(status_code=503, detail="Organizer service not available")
            
            # Perform organization
            suggestions = file_organizer.organize_project(
                project.path,
                auto_move=request.auto_move
            )
            
            return {
                "project_id": request.project_id,
                "suggestions": {folder: len(files) for folder, files in suggestions.items()},
                "auto_moved": request.auto_move,
                "total_files": sum(len(files) for files in suggestions.values())
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Organization failed: {e}")
        raise HTTPException(status_code=500, detail="Organization operation failed")

@app.post("/upload")
async def upload_file(project_id: str, file: UploadFile = FastFile(...)):
    """Upload and auto-categorize a file"""
    try:
        with db_session() as session:
            project = get_project_by_id(session, project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            # Suggest category
            suggested_folder = "uploads"
            if file_organizer:
                suggested_folder = file_organizer.suggest_organization(file.filename)
            
            # Create target directory
            folder_path = Path(project.path) / suggested_folder
            folder_path.mkdir(exist_ok=True)
            
            # Save file
            file_path = folder_path / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            logger.info(f"File uploaded: {file_path}")
            
            return {
                "filename": file.filename,
                "saved_to": str(file_path),
                "suggested_category": suggested_folder,
                "size": len(content)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

# LLM endpoints
@app.post("/api/llm/summarize", response_model=LLMResponse)
async def summarize_file(request: SummarizeRequest):
    """Summarize file content using local LLM"""
    try:
        if not llm_service:
            return LLMResponse(
                success=False,
                error="LLM service not available",
                service_available=False
            )
        
        # Get file content if not provided
        content = request.content
        if not content:
            with db_session() as session:
                file_record = session.query(File).filter(File.id == request.file_id).first()
                if not file_record:
                    raise HTTPException(status_code=404, detail="File not found")
                content = file_record.content or ""
        
        if not content.strip():
            return LLMResponse(
                success=False,
                error="No content available to summarize",
                service_available=llm_service.is_available
            )
        
        # Get file context
        with db_session() as session:
            file_record = session.query(File).filter(File.id == request.file_id).first()
            context = {}
            if file_record:
                context = {
                    'file_name': file_record.name,
                    'file_type': file_record.type,
                    'project_id': file_record.project_id
                }
        
        # Generate summary
        summary_result = await llm_service.summarize_content(content, context)
        
        return LLMResponse(
            success=True,
            data=summary_result,
            service_available=llm_service.is_available
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return LLMResponse(
            success=False,
            error=str(e),
            service_available=llm_service.is_available if llm_service else False
        )

@app.post("/api/llm/query", response_model=LLMResponse)
async def query_file(request: QueryRequest):
    """Ask questions about file content using local LLM"""
    try:
        if not llm_service:
            return LLMResponse(
                success=False,
                error="LLM service not available",
                service_available=False
            )
        
        # Get file content if not provided
        content = request.content
        if not content:
            with db_session() as session:
                file_record = session.query(File).filter(File.id == request.file_id).first()
                if not file_record:
                    raise HTTPException(status_code=404, detail="File not found")
                content = file_record.content or ""
        
        if not content.strip():
            return LLMResponse(
                success=False,
                error="No content available to query",
                service_available=llm_service.is_available
            )
        
        # Get file context
        with db_session() as session:
            file_record = session.query(File).filter(File.id == request.file_id).first()
            context = {}
            if file_record:
                context = {
                    'file_name': file_record.name,
                    'file_type': file_record.type
                }
        
        # Generate answer
        query_result = await llm_service.query_content(request.query, content, context)
        
        return LLMResponse(
            success=True,
            data=query_result,
            service_available=llm_service.is_available
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return LLMResponse(
            success=False,
            error=str(e),
            service_available=llm_service.is_available if llm_service else False
        )

@app.post("/api/llm/suggest-organization", response_model=LLMResponse)
async def suggest_file_organization(request: OrganizationSuggestionRequest):
    """Get AI-powered organization suggestions for a file"""
    try:
        if not llm_service:
            return LLMResponse(
                success=False,
                error="LLM service not available",
                service_available=False
            )
        
        # Prepare file info for analysis
        file_info = {
            'file_name': request.file_name,
            'content': request.content or '',
            'file_type': request.file_type or Path(request.file_name).suffix
        }
        
        # Generate organization suggestion
        suggestion_result = await llm_service.suggest_organization(file_info)
        
        return LLMResponse(
            success=True,
            data=suggestion_result,
            service_available=llm_service.is_available
        )
        
    except Exception as e:
        logger.error(f"Organization suggestion failed: {e}")
        return LLMResponse(
            success=False,
            error=str(e),
            service_available=llm_service.is_available if llm_service else False
        )

@app.get("/api/llm/status", response_model=Dict[str, Any])
async def get_llm_status():
    """Get detailed LLM service status"""
    try:
        if not llm_service:
            return {
                "available": False,
                "error": "LLM service not initialized",
                "service_available": False
            }
        
        status = await llm_service.get_service_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get LLM status: {e}")
        return {
            "available": False,
            "error": str(e),
            "service_available": False
        }

@app.post("/api/ocr/extract")
async def extract_ocr(file: UploadFile = FastFile(...), 
                      max_pages: Optional[int] = None):
    """Extract text from uploaded image or PDF using OCR"""
    try:
        if not OCR_AVAILABLE or not is_ocr_available():
            raise HTTPException(status_code=503, detail="OCR service not available")
        
        # Get OCR processor
        ocr_processor = get_ocr_processor()
        if not ocr_processor:
            raise HTTPException(status_code=503, detail="OCR processor not initialized")
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Check if file can be processed
            if not ocr_processor.can_process_file(temp_file_path):
                status = ocr_processor.get_file_ocr_status(temp_file_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"File cannot be processed: {status.get('reason', 'Unknown reason')}"
                )
            
            # Process with OCR
            logger.info(f"Processing uploaded file with OCR: {file.filename}")
            ocr_result = await ocr_processor.extract_text_async(temp_file_path, max_pages)
            
            # Convert to response format
            result = OCRResult(
                text=ocr_result.text,
                confidence=ocr_result.confidence,
                language_detected=ocr_result.language_detected,
                processing_time=ocr_result.processing_time,
                word_count=ocr_result.word_count,
                error=ocr_result.error,
                metadata=ocr_result.metadata
            )
            
            logger.info(f"OCR completed for {file.filename}: {len(ocr_result.text)} chars extracted")
            return result
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR extraction failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="OCR extraction failed")

@app.get("/api/files/{file_id}/ocr-status")
async def get_file_ocr_status(file_id: str):
    """Get OCR processing status for a file"""
    try:
        with db_session() as session:
            file_record = session.query(File).filter(File.id == file_id).first()
            if not file_record:
                raise HTTPException(status_code=404, detail="File not found")
            
            # Check if OCR is available
            if not OCR_AVAILABLE or not is_ocr_available():
                return OCRStatusResponse(
                    file_id=file_id,
                    file_path=file_record.path,
                    can_process=False,
                    is_processed=False,
                    reason="OCR service not available"
                )
            
            # Get OCR processor
            ocr_processor = get_ocr_processor()
            if not ocr_processor:
                return OCRStatusResponse(
                    file_id=file_id,
                    file_path=file_record.path,
                    can_process=False,
                    is_processed=False,
                    reason="OCR processor not initialized"
                )
            
            # Check processing capability
            can_process = ocr_processor.can_process_file(file_record.path)
            status_info = ocr_processor.get_file_ocr_status(file_record.path)
            
            # Check if already processed (has OCR metadata)
            is_processed = False
            ocr_result_data = None
            if file_record.metadata and 'ocr_status' in file_record.metadata:
                is_processed = file_record.metadata.get('ocr_processed', False)
                # If processed, try to get stored OCR result
                if is_processed and file_record.content:
                    # Create a mock OCR result from stored data
                    ocr_result_data = OCRResult(
                        text=file_record.content[:500] + "..." if len(file_record.content) > 500 else file_record.content,
                        confidence=file_record.metadata.get('ocr_confidence', 0.0),
                        language_detected=file_record.metadata.get('ocr_language', 'unknown'),
                        processing_time=file_record.metadata.get('ocr_processing_time', 0.0),
                        word_count=len(file_record.content.split()) if file_record.content else 0
                    )
            
            return OCRStatusResponse(
                file_id=file_id,
                file_path=file_record.path,
                can_process=can_process,
                is_processed=is_processed,
                ocr_result=ocr_result_data,
                estimated_processing_time=status_info.get('estimated_processing_time'),
                reason=status_info.get('reason') if not can_process else None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get OCR status for file {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get OCR status")

@app.post("/api/files/{file_id}/ocr-reprocess")
async def reprocess_file_ocr(file_id: str, background_tasks: BackgroundTasks,
                           max_pages: Optional[int] = None):
    """Reprocess a file with OCR (useful for updating OCR results)"""
    try:
        with db_session() as session:
            file_record = session.query(File).filter(File.id == file_id).first()
            if not file_record:
                raise HTTPException(status_code=404, detail="File not found")
            
            if not OCR_AVAILABLE or not is_ocr_available():
                raise HTTPException(status_code=503, detail="OCR service not available")
            
            # Check if file exists
            if not os.path.exists(file_record.path):
                raise HTTPException(status_code=404, detail="File not found on disk")
            
            # Get OCR processor
            ocr_processor = get_ocr_processor()
            if not ocr_processor:
                raise HTTPException(status_code=503, detail="OCR processor not initialized")
            
            if not ocr_processor.can_process_file(file_record.path):
                status_info = ocr_processor.get_file_ocr_status(file_record.path)
                raise HTTPException(
                    status_code=400,
                    detail=f"File cannot be processed: {status_info.get('reason', 'Unknown reason')}"
                )
            
            # Schedule background OCR processing
            background_tasks.add_task(
                reprocess_file_with_ocr, 
                file_record.path, 
                file_id, 
                max_pages
            )
            
            return {
                "message": "OCR reprocessing scheduled",
                "file_id": file_id,
                "file_path": file_record.path
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule OCR reprocessing for file {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to schedule OCR reprocessing")

@app.get("/api/ocr/capabilities")
async def get_ocr_capabilities_endpoint():
    """Get OCR service capabilities and status"""
    try:
        capabilities = get_ocr_capabilities()
        return capabilities
    except Exception as e:
        logger.error(f"Failed to get OCR capabilities: {e}")
        return {
            "ocr_available": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "healthy",
            "file_watcher": "healthy" if file_watcher_manager else "unavailable",
            "semantic_search": "healthy" if semantic_search else "unavailable",
            "organizer": "healthy" if file_organizer else "unavailable",
            "llm_service": "healthy" if llm_service and llm_service.is_available else "unavailable",
            "ocr": "healthy" if OCR_AVAILABLE and is_ocr_available() else "unavailable"
        }
    }

# Background tasks
async def reprocess_file_with_ocr(file_path: str, file_id: str, max_pages: Optional[int] = None):
    """Background task to reprocess a file with OCR"""
    try:
        if not OCR_AVAILABLE or not is_ocr_available():
            logger.warning(f"OCR not available for reprocessing: {file_path}")
            return
        
        # Get OCR processor
        ocr_processor = get_ocr_processor()
        if not ocr_processor:
            logger.warning(f"OCR processor not available for reprocessing: {file_path}")
            return
        
        # Perform OCR
        logger.info(f"Starting OCR reprocessing for: {file_path}")
        ocr_result = await ocr_processor.extract_text_async(file_path, max_pages)
        
        if ocr_result.error:
            logger.error(f"OCR reprocessing failed for {file_path}: {ocr_result.error}")
            return
        
        # Update database with new OCR results
        with db_session() as session:
            file_record = session.query(File).filter(File.id == file_id).first()
            if file_record:
                # Update content
                file_record.content = ocr_result.text[:5000]  # Limit content size
                
                # Update metadata with OCR results
                if not file_record.metadata:
                    file_record.metadata = {}
                
                file_record.metadata.update({
                    'ocr_processed': True,
                    'ocr_confidence': ocr_result.confidence,
                    'ocr_language': ocr_result.language_detected,
                    'ocr_processing_time': ocr_result.processing_time,
                    'ocr_word_count': ocr_result.word_count,
                    'ocr_reprocessed_at': datetime.utcnow().isoformat(),
                    'content_source': 'ocr'
                })
                
                session.commit()
                
                # Re-index for search if semantic search is available
                if semantic_search and ocr_result.text:
                    try:
                        await semantic_search.index_file(
                            file_id,
                            ocr_result.text,
                            {'project_id': file_record.project_id, 'file_name': file_record.name}
                        )
                        logger.info(f"File re-indexed for search after OCR: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to re-index file for search after OCR: {e}")
                
                logger.info(f"OCR reprocessing completed for {file_path}: "
                          f"{len(ocr_result.text)} chars extracted, confidence: {ocr_result.confidence:.2f}")
            else:
                logger.warning(f"File record not found for OCR update: {file_id}")
        
    except Exception as e:
        logger.error(f"OCR reprocessing failed for {file_path}: {e}")

async def index_existing_files(project_path: str, project_id: str):
    """Background task to index existing files in a project"""
    try:
        logger.info(f"Starting background indexing for project: {project_id}")
        
        indexed_count = 0
        for root, dirs, files in os.walk(project_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                
                # Check if already indexed
                with db_session() as session:
                    existing = session.query(File).filter(File.path == file_path).first()
                    if existing:
                        continue
                
                try:
                    # Process file
                    file_stat = os.stat(file_path)
                    file_ext = os.path.splitext(file_name)[1]
                    
                    # Extract content for text files
                    content = ""
                    if file_ext in ['.txt', '.md', '.py', '.js', '.json']:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()[:5000]  # Limit content
                        except:
                            pass
                    
                    # Save to database
                    with db_session() as session:
                        file_record = create_file(
                            session,
                            project_id=project_id,
                            path=file_path,
                            name=file_name,
                            file_type=file_ext,
                            content=content,
                            file_metadata={
                                'size': file_stat.st_size,
                                'modified': file_stat.st_mtime
                            }
                        )
                        
                        # Index for search if content available
                        if content and semantic_search:
                            try:
                                await semantic_search.index_file(
                                    file_record.id,
                                    content,
                                    {'project_id': project_id, 'file_name': file_name}
                                )
                            except Exception as e:
                                logger.warning(f"Failed to index file for search: {e}")
                        
                        indexed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    continue
        
        logger.info(f"Background indexing completed: {indexed_count} files indexed")
        
    except Exception as e:
        logger.error(f"Background indexing failed: {e}")

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    )