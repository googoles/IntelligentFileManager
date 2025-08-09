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
        get_project_by_id
    )
    from file_watcher import FileWatcherManager
    from organizer import FileOrganizer, create_project_structure
    
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global file_watcher_manager, semantic_search, file_organizer
    
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
        
        # Format results
        formatted_results = []
        with db_session() as session:
            for result in results:
                file_record = session.query(File).filter(File.id == result.file_id).first()
                if file_record:
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
                        score=result.similarity
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
            "organizer": "healthy" if file_organizer else "unavailable"
        }
    }

# Background tasks
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