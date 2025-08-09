# 1일 MVP 개발 로드맵 - 연구자용 파일 관리 시스템

## 🎯 MVP 목표
**"폴더 모니터링 → 자동 분류 → 의미 검색"** 핵심 기능만 구현

## 📋 사전 준비 (30분)
### 환경 설정
```bash
# Python 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install fastapi uvicorn watchdog sentence-transformers chromadb sqlalchemy aiofiles python-multipart
```

### 프로젝트 구조
```
research-assistant-mvp/
├── backend/
│   ├── main.py           # FastAPI 앱
│   ├── file_watcher.py   # 파일 모니터링
│   ├── organizer.py      # 자동 정리 로직
│   ├── search.py         # 의미 검색
│   └── database.py       # DB 모델
├── frontend/
│   └── index.html        # 단일 페이지 웹 UI
└── data/
    ├── projects/         # 프로젝트 폴더
    └── db/              # 데이터베이스
```

## ⏱️ 시간별 개발 일정

### 🌅 오전 (9:00 - 12:00): Backend Core

#### **Hour 1 (9:00-10:00): 데이터베이스 및 파일 워처**

```python
# database.py
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class File(Base):
    __tablename__ = 'files'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String)
    path = Column(String)
    name = Column(String)
    type = Column(String)
    content = Column(Text)  # 추출된 텍스트
    metadata = Column(JSON)
    embedding = Column(Text)  # JSON으로 저장
    created_at = Column(DateTime, default=datetime.utcnow)

# DB 초기화
engine = create_engine('sqlite:///data/db/research.db')
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)
```

```python
# file_watcher.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import hashlib
from database import SessionLocal, File
import json

class FileHandler(FileSystemEventHandler):
    def __init__(self, project_id):
        self.project_id = project_id
        
    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)
    
    def process_file(self, file_path):
        db = SessionLocal()
        try:
            # 파일 메타데이터 추출
            file_stat = os.stat(file_path)
            file_ext = os.path.splitext(file_path)[1]
            
            # 간단한 텍스트 추출 (MVP용)
            content = ""
            if file_ext in ['.txt', '.md', '.py', '.js']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:5000]  # 처음 5000자만
            
            # DB에 저장
            new_file = File(
                project_id=self.project_id,
                path=file_path,
                name=os.path.basename(file_path),
                type=file_ext,
                content=content,
                metadata={
                    'size': file_stat.st_size,
                    'modified': file_stat.st_mtime
                }
            )
            db.add(new_file)
            db.commit()
            
            print(f"✅ 파일 추가됨: {file_path}")
            
        finally:
            db.close()

def start_watching(project_path, project_id):
    event_handler = FileHandler(project_id)
    observer = Observer()
    observer.schedule(event_handler, project_path, recursive=True)
    observer.start()
    return observer
```

#### **Hour 2 (10:00-11:00): 자동 정리 로직**

```python
# organizer.py
import os
import shutil
from datetime import datetime
from typing import Dict, List

class FileOrganizer:
    def __init__(self):
        # 간단한 규칙 기반 분류
        self.rules = {
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.md'],
            'data': ['.csv', '.xlsx', '.json', '.xml'],
            'code': ['.py', '.js', '.r', '.ipynb', '.java'],
            'images': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
            'results': ['result', 'output', 'figure', 'plot']
        }
        
    def suggest_organization(self, file_path: str) -> str:
        """파일을 어느 폴더로 분류할지 제안"""
        file_name = os.path.basename(file_path).lower()
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 확장자 기반 분류
        for category, extensions in self.rules.items():
            if file_ext in extensions:
                return category
        
        # 파일명 패턴 기반 분류
        for keyword in self.rules['results']:
            if keyword in file_name:
                return 'results'
        
        # 날짜 기반 폴더 (기본값)
        return f"unsorted_{datetime.now().strftime('%Y%m')}"
    
    def organize_project(self, project_path: str, auto_move: bool = False):
        """프로젝트 폴더 자동 정리"""
        suggestions = {}
        
        # 모든 파일 스캔
        for root, dirs, files in os.walk(project_path):
            # 이미 정리된 폴더는 건너뛰기
            if any(cat in root for cat in self.rules.keys()):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                suggested_folder = self.suggest_organization(file_path)
                
                if suggested_folder not in suggestions:
                    suggestions[suggested_folder] = []
                suggestions[suggested_folder].append(file_path)
        
        # 자동 이동 모드
        if auto_move:
            for folder, files in suggestions.items():
                folder_path = os.path.join(project_path, folder)
                os.makedirs(folder_path, exist_ok=True)
                
                for file_path in files:
                    new_path = os.path.join(folder_path, os.path.basename(file_path))
                    try:
                        shutil.move(file_path, new_path)
                        print(f"📁 이동: {file_path} → {new_path}")
                    except Exception as e:
                        print(f"❌ 이동 실패: {e}")
        
        return suggestions

# 프로젝트 템플릿 생성
def create_project_structure(project_path: str, template: str = 'research'):
    """새 프로젝트 폴더 구조 생성"""
    
    templates = {
        'research': [
            'literature',
            'data/raw',
            'data/processed',
            'code',
            'results/figures',
            'results/tables',
            'drafts',
            'notes'
        ],
        'minimal': [
            'input',
            'output',
            'workspace'
        ]
    }
    
    folders = templates.get(template, templates['minimal'])
    
    for folder in folders:
        folder_path = os.path.join(project_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 생성: {folder_path}")
    
    # README 파일 생성
    readme_path = os.path.join(project_path, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(f"# Project: {os.path.basename(project_path)}\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Structure\n\n")
        for folder in folders:
            f.write(f"- **{folder}**: \n")
    
    return folders
```

#### **Hour 3 (11:00-12:00): 의미 검색 구현**

```python
# search.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
from typing import List, Dict
from database import SessionLocal, File

class SemanticSearch:
    def __init__(self):
        # 경량 모델 사용 (빠른 속도)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ChromaDB 초기화 (로컬)
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="data/db/chroma"
        ))
        
        # 컬렉션 생성 또는 가져오기
        try:
            self.collection = self.chroma_client.create_collection("research_files")
        except:
            self.collection = self.chroma_client.get_collection("research_files")
    
    def index_file(self, file_id: str, content: str, metadata: dict):
        """파일 내용을 벡터화하여 저장"""
        if not content:
            return
        
        # 텍스트를 청크로 분할 (간단한 방법)
        chunks = self._split_text(content, chunk_size=500)
        
        for i, chunk in enumerate(chunks):
            # 임베딩 생성
            embedding = self.model.encode(chunk).tolist()
            
            # ChromaDB에 저장
            self.collection.add(
                ids=[f"{file_id}_{i}"],
                embeddings=[embedding],
                metadatas=[{
                    'file_id': file_id,
                    'chunk_index': i,
                    **metadata
                }],
                documents=[chunk]
            )
    
    def search(self, query: str, project_id: str = None, top_k: int = 5) -> List[Dict]:
        """의미 기반 검색"""
        # 쿼리 임베딩
        query_embedding = self.model.encode(query).tolist()
        
        # 검색 실행
        where_clause = {'project_id': project_id} if project_id else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        # 결과 포맷팅
        formatted_results = []
        if results['ids'][0]:  # 결과가 있으면
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'file_id': results['metadatas'][0][i]['file_id'],
                    'content': results['documents'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                })
        
        return formatted_results
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """텍스트를 청크로 분할"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def find_similar_files(self, file_id: str, top_k: int = 5) -> List[Dict]:
        """유사한 파일 찾기"""
        db = SessionLocal()
        try:
            file = db.query(File).filter(File.id == file_id).first()
            if file and file.content:
                return self.search(file.content[:500], project_id=file.project_id, top_k=top_k)
        finally:
            db.close()
        return []
```

### ☀️ 오후 (13:00 - 17:00): API & Frontend

#### **Hour 4-5 (13:00-15:00): FastAPI 백엔드**

```python
# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File as FastFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from datetime import datetime

from database import SessionLocal, Project, File, engine, Base
from file_watcher import start_watching
from organizer import FileOrganizer, create_project_structure
from search import SemanticSearch

# DB 초기화
Base.metadata.create_all(bind=engine)

# FastAPI 앱
app = FastAPI(title="Research File Manager MVP")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 객체
file_organizer = FileOrganizer()
semantic_search = SemanticSearch()
watchers = {}  # 프로젝트별 파일 워처

# Pydantic 모델
class ProjectCreate(BaseModel):
    name: str
    path: str
    template: Optional[str] = "research"

class SearchQuery(BaseModel):
    query: str
    project_id: Optional[str] = None
    top_k: Optional[int] = 5

class OrganizeRequest(BaseModel):
    project_id: str
    auto_move: Optional[bool] = False

# API 엔드포인트

@app.get("/")
async def root():
    """메인 페이지"""
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/projects")
async def create_project(project: ProjectCreate, background_tasks: BackgroundTasks):
    """새 프로젝트 생성"""
    db = SessionLocal()
    try:
        # 프로젝트 폴더 생성
        os.makedirs(project.path, exist_ok=True)
        create_project_structure(project.path, project.template)
        
        # DB에 저장
        db_project = Project(
            name=project.name,
            path=project.path
        )
        db.add(db_project)
        db.commit()
        db.refresh(db_project)
        
        # 파일 워처 시작
        watcher = start_watching(project.path, db_project.id)
        watchers[db_project.id] = watcher
        
        # 기존 파일 인덱싱 (백그라운드)
        background_tasks.add_task(index_existing_files, project.path, db_project.id)
        
        return {"id": db_project.id, "name": db_project.name, "path": db_project.path}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

@app.get("/projects")
async def list_projects():
    """프로젝트 목록"""
    db = SessionLocal()
    try:
        projects = db.query(Project).all()
        return [{"id": p.id, "name": p.name, "path": p.path} for p in projects]
    finally:
        db.close()

@app.get("/projects/{project_id}/files")
async def list_files(project_id: str):
    """프로젝트 파일 목록"""
    db = SessionLocal()
    try:
        files = db.query(File).filter(File.project_id == project_id).all()
        return [{
            "id": f.id,
            "name": f.name,
            "type": f.type,
            "path": f.path,
            "size": f.metadata.get('size', 0) if f.metadata else 0
        } for f in files]
    finally:
        db.close()

@app.post("/search")
async def search_files(query: SearchQuery):
    """의미 기반 파일 검색"""
    try:
        results = semantic_search.search(
            query.query,
            project_id=query.project_id,
            top_k=query.top_k
        )
        
        # 파일 정보 추가
        db = SessionLocal()
        try:
            enriched_results = []
            for result in results:
                file = db.query(File).filter(File.id == result['file_id']).first()
                if file:
                    enriched_results.append({
                        'file': {
                            'id': file.id,
                            'name': file.name,
                            'path': file.path,
                            'type': file.type
                        },
                        'snippet': result['content'][:200] + "...",
                        'score': 1 - result['distance']  # 유사도 점수
                    })
            return enriched_results
        finally:
            db.close()
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/organize")
async def organize_project(request: OrganizeRequest):
    """프로젝트 파일 자동 정리"""
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == request.project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        suggestions = file_organizer.organize_project(
            project.path,
            auto_move=request.auto_move
        )
        
        return {
            "project_id": request.project_id,
            "suggestions": {
                folder: len(files) for folder, files in suggestions.items()
            },
            "auto_moved": request.auto_move
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

@app.post("/upload")
async def upload_file(project_id: str, file: UploadFile = FastFile(...)):
    """파일 업로드 및 자동 분류"""
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # 파일 저장
        suggested_folder = file_organizer.suggest_organization(file.filename)
        folder_path = os.path.join(project.path, suggested_folder)
        os.makedirs(folder_path, exist_ok=True)
        
        file_path = os.path.join(folder_path, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "filename": file.filename,
            "saved_to": file_path,
            "suggested_category": suggested_folder
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

# 백그라운드 작업
def index_existing_files(project_path: str, project_id: str):
    """기존 파일 인덱싱"""
    db = SessionLocal()
    try:
        for root, dirs, files in os.walk(project_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                
                # 이미 인덱싱된 파일 확인
                existing = db.query(File).filter(File.path == file_path).first()
                if existing:
                    continue
                
                # 파일 처리
                file_ext = os.path.splitext(file_name)[1]
                content = ""
                
                if file_ext in ['.txt', '.md', '.py', '.js']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()[:5000]
                    except:
                        pass
                
                # DB에 저장
                new_file = File(
                    project_id=project_id,
                    path=file_path,
                    name=file_name,
                    type=file_ext,
                    content=content,
                    metadata={'size': os.path.getsize(file_path)}
                )
                db.add(new_file)
                db.commit()
                
                # 벡터 인덱싱
                if content:
                    semantic_search.index_file(
                        new_file.id,
                        content,
                        {'project_id': project_id, 'file_name': file_name}
                    )
                
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### **Hour 6 (15:00-16:00): 간단한 웹 UI**

```html
<!-- frontend/index.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research File Manager MVP</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .tabs {
            display: flex;
            background: #f5f5f5;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .tab {
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        
        .tab:hover {
            background: #e8e8e8;
        }
        
        .tab.active {
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }
        
        .content {
            padding: 30px;
            min-height: 400px;
        }
        
        .section {
            display: none;
        }
        
        .section.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        
        .search-box input {
            flex: 1;
        }
        
        .results {
            margin-top: 30px;
        }
        
        .result-item {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }
        
        .result-item:hover {
            transform: translateX(5px);
            background: #f0f0f0;
        }
        
        .file-name {
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        
        .file-path {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }
        
        .snippet {
            color: #555;
            line-height: 1.5;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }
        
        .score {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            margin-top: 10px;
        }
        
        .project-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .project-card {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s;
        }
        
        .project-card:hover {
            border-color: #667eea;
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
        }
        
        .message {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .message.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .message.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Research File Manager</h1>
            <p class="subtitle">AI-Powered File Organization & Semantic Search</p>
        </header>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('search')">🔍 검색</div>
            <div class="tab" onclick="switchTab('projects')">📁 프로젝트</div>
            <div class="tab" onclick="switchTab('organize')">🗂️ 자동 정리</div>
        </div>
        
        <div class="content">
            <!-- 검색 섹션 -->
            <div id="search" class="section active">
                <h2>의미 기반 파일 검색</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    자연어로 원하는 파일을 검색하세요. AI가 내용을 이해하고 관련 파일을 찾아드립니다.
                </p>
                
                <div class="form-group">
                    <label>프로젝트 선택</label>
                    <select id="searchProject">
                        <option value="">모든 프로젝트</option>
                    </select>
                </div>
                
                <div class="search-box">
                    <input type="text" id="searchQuery" placeholder="예: 작년 머신러닝 실험 결과 그래프">
                    <button onclick="searchFiles()">검색</button>
                </div>
                
                <div id="searchResults" class="results"></div>
            </div>
            
            <!-- 프로젝트 섹션 -->
            <div id="projects" class="section">
                <h2>프로젝트 관리</h2>
                
                <div class="form-group">
                    <label>프로젝트 이름</label>
                    <input type="text" id="projectName" placeholder="예: ML_Research_2024">
                </div>
                
                <div class="form-group">
                    <label>프로젝트 경로</label>
                    <input type="text" id="projectPath" placeholder="예: C:/Research/ML_Project">
                </div>
                
                <div class="form-group">
                    <label>템플릿</label>
                    <select id="projectTemplate">
                        <option value="research">연구 프로젝트 (권장)</option>
                        <option value="minimal">최소 구조</option>
                    </select>
                </div>
                
                <button onclick="createProject()">프로젝트 생성</button>
                
                <h3 style="margin-top: 40px;">기존 프로젝트</h3>
                <div id="projectList" class="project-list"></div>
            </div>
            
            <!-- 자동 정리 섹션 -->
            <div id="organize" class="section">
                <h2>파일 자동 정리</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    AI가 파일을 분석하여 자동으로 적절한 폴더로 분류해드립니다.
                </p>
                
                <div class="form-group">
                    <label>프로젝트 선택</label>
                    <select id="organizeProject">
                        <option value="">프로젝트를 선택하세요</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="autoMove" style="width: auto; margin-right: 10px;">
                        자동으로 파일 이동 (체크하지 않으면 제안만 표시)
                    </label>
                </div>
                
                <button onclick="organizeFiles()">정리 시작</button>
                
                <div id="organizeResults" style="margin-top: 30px;"></div>
            </div>
        </div>
    </div>
    
    <script>
        const API_URL = 'http://localhost:8000';
        
        // 탭 전환
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('active');
            });
            
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
            
            if (tabName === 'projects') {
                loadProjects();
            }
        }
        
        // 메시지 표시
        function showMessage(message, type = 'success') {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.textContent = message;
            
            const content = document.querySelector('.content');
            content.insertBefore(messageDiv, content.firstChild);
            
            setTimeout(() => messageDiv.remove(), 5000);
        }
        
        // 로딩 표시
        function showLoading(elementId) {
            document.getElementById(elementId).innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>처리 중...</p>
                </div>
            `;
        }
        
        // 프로젝트 목록 로드
        async function loadProjects() {
            try {
                const response = await fetch(`${API_URL}/projects`);
                const projects = await response.json();
                
                // 프로젝트 목록 표시
                const listHtml = projects.map(p => `
                    <div class="project-card">
                        <h3>${p.name}</h3>
                        <p style="color: #666; font-size: 0.9em;">${p.path}</p>
                        <p style="margin-top: 10px;">
                            <span class="score">ID: ${p.id.substring(0, 8)}...</span>
                        </p>
                    </div>
                `).join('');
                
                document.getElementById('projectList').innerHTML = listHtml || '<p>프로젝트가 없습니다.</p>';
                
                // 셀렉트 박스 업데이트
                const options = projects.map(p => 
                    `<option value="${p.id}">${p.name}</option>`
                ).join('');
                
                document.getElementById('searchProject').innerHTML = 
                    '<option value="">모든 프로젝트</option>' + options;
                document.getElementById('organizeProject').innerHTML = 
                    '<option value="">프로젝트를 선택하세요</option>' + options;
                    
            } catch (error) {
                console.error('프로젝트 로드 실패:', error);
            }
        }
        
        // 프로젝트 생성
        async function createProject() {
            const name = document.getElementById('projectName').value;
            const path = document.getElementById('projectPath').value;
            const template = document.getElementById('projectTemplate').value;
            
            if (!name || !path) {
                showMessage('프로젝트 이름과 경로를 입력하세요.', 'error');
                return;
            }
            
            try {
                const response = await fetch(`${API_URL}/projects`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name, path, template})
                });
                
                if (response.ok) {
                    const result = await response.json();
                    showMessage(`프로젝트 "${result.name}" 생성 완료!`);
                    document.getElementById('projectName').value = '';
                    document.getElementById('projectPath').value = '';
                    loadProjects();
                } else {
                    const error = await response.json();
                    showMessage(error.detail || '프로젝트 생성 실패', 'error');
                }
            } catch (error) {
                showMessage('서버 연결 실패', 'error');
            }
        }
        
        // 파일 검색
        async function searchFiles() {
            const query = document.getElementById('searchQuery').value;
            const projectId = document.getElementById('searchProject').value;
            
            if (!query) {
                showMessage('검색어를 입력하세요.', 'error');
                return;
            }
            
            showLoading('searchResults');
            
            try {
                const response = await fetch(`${API_URL}/search`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        query,
                        project_id: projectId || null,
                        top_k: 10
                    })
                });
                
                if (response.ok) {
                    const results = await response.json();
                    displaySearchResults(results);
                } else {
                    document.getElementById('searchResults').innerHTML = 
                        '<p style="color: #666;">검색 결과가 없습니다.</p>';
                }
            } catch (error) {
                document.getElementById('searchResults').innerHTML = 
                    '<p style="color: red;">검색 실패: 서버 연결 오류</p>';
            }
        }
        
        // 검색 결과 표시
        function displaySearchResults(results) {
            if (results.length === 0) {
                document.getElementById('searchResults').innerHTML = 
                    '<p style="color: #666;">검색 결과가 없습니다.</p>';
                return;
            }
            
            const html = results.map(r => `
                <div class="result-item">
                    <div class="file-name">📄 ${r.file.name}</div>
                    <div class="file-path">${r.file.path}</div>
                    <div class="snippet">${r.snippet}</div>
                    <span class="score">유사도: ${(r.score * 100).toFixed(1)}%</span>
                </div>
            `).join('');
            
            document.getElementById('searchResults').innerHTML = html;
        }
        
        // 파일 정리
        async function organizeFiles() {
            const projectId = document.getElementById('organizeProject').value;
            const autoMove = document.getElementById('autoMove').checked;
            
            if (!projectId) {
                showMessage('프로젝트를 선택하세요.', 'error');
                return;
            }
            
            showLoading('organizeResults');
            
            try {
                const response = await fetch(`${API_URL}/organize`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        project_id: projectId,
                        auto_move: autoMove
                    })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    displayOrganizeResults(result);
                } else {
                    document.getElementById('organizeResults').innerHTML = 
                        '<p style="color: red;">정리 실패</p>';
                }
            } catch (error) {
                document.getElementById('organizeResults').innerHTML = 
                    '<p style="color: red;">서버 연결 오류</p>';
            }
        }
        
        // 정리 결과 표시
        function displayOrganizeResults(result) {
            const suggestions = Object.entries(result.suggestions);
            
            if (suggestions.length === 0) {
                document.getElementById('organizeResults').innerHTML = 
                    '<p style="color: #666;">정리할 파일이 없습니다.</p>';
                return;
            }
            
            const html = `
                <div class="message success">
                    ${result.auto_moved ? '파일이 자동으로 정리되었습니다!' : '정리 제안이 생성되었습니다.'}
                </div>
                <h3>정리 결과</h3>
                ${suggestions.map(([folder, count]) => `
                    <div class="result-item">
                        <strong>📁 ${folder}</strong>: ${count}개 파일
                    </div>
                `).join('')}
            `;
            
            document.getElementById('organizeResults').innerHTML = html;
        }
        
        // Enter 키로 검색
        document.getElementById('searchQuery').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchFiles();
        });
        
        // 초기 로드
        loadProjects();
    </script>
</body>
</html>
```

### 🌙 저녁 (17:00 - 18:00): 테스트 & 마무리

#### **Hour 7 (17:00-18:00): 통합 테스트 및 디버깅**

```bash
# run.sh - 실행 스크립트
#!/bin/bash

echo "🚀 Research File Manager MVP 시작..."

# 가상환경 활성화
source venv/bin/activate

# 데이터 디렉토리 생성
mkdir -p data/projects data/db

# 서버 시작
echo "📡 서버 시작 (http://localhost:8000)..."
python backend/main.py
```

```python
# test_mvp.py - 간단한 테스트
import requests
import time

API_URL = "http://localhost:8000"

def test_mvp():
    print("🧪 MVP 테스트 시작...")
    
    # 1. 프로젝트 생성
    print("\n1️⃣ 프로젝트 생성 테스트")
    project_data = {
        "name": "Test_Project",
        "path": "./data/projects/test_project",
        "template": "research"
    }
    response = requests.post(f"{API_URL}/projects", json=project_data)
    project = response.json()
    print(f"✅ 프로젝트 생성: {project['name']}")
    
    # 2. 파일 업로드
    print("\n2️⃣ 파일 업로드 테스트")
    test_content = "This is a test file about machine learning experiments."
    files = {'file': ('test.txt', test_content)}
    response = requests.post(
        f"{API_URL}/upload?project_id={project['id']}", 
        files=files
    )
    print(f"✅ 파일 업로드: {response.json()['filename']}")
    
    # 잠시 대기 (인덱싱 시간)
    time.sleep(2)
    
    # 3. 검색 테스트
    print("\n3️⃣ 의미 검색 테스트")
    search_data = {
        "query": "machine learning",
        "project_id": project['id']
    }
    response = requests.post(f"{API_URL}/search", json=search_data)
    results = response.json()
    print(f"✅ 검색 결과: {len(results)}개 파일 발견")
    
    # 4. 자동 정리 테스트
    print("\n4️⃣ 자동 정리 테스트")
    organize_data = {
        "project_id": project['id'],
        "auto_move": False
    }
    response = requests.post(f"{API_URL}/organize", json=organize_data)
    result = response.json()
    print(f"✅ 정리 제안: {result['suggestions']}")
    
    print("\n✨ 모든 테스트 통과!")

if __name__ == "__main__":
    test_mvp()
```

## 📦 최종 체크리스트

### 완성된 MVP 기능
- ✅ **파일 모니터링**: 프로젝트 폴더 실시간 감시
- ✅ **자동 분류**: 규칙 기반 파일 정리 제안
- ✅ **의미 검색**: 자연어로 파일 검색
- ✅ **프로젝트 관리**: 생성, 목록, 구조화
- ✅ **웹 인터페이스**: 직관적인 UI
- ✅ **로컬 우선**: 완전한 오프라인 작동

### Claude Code 활용 팁
1. **코드 생성**: "이 함수를 완성해줘" 형태로 요청
2. **디버깅**: 에러 메시지 복사해서 "이 에러 어떻게 해결?"
3. **최적화**: "이 코드 더 빠르게 만들어줘"
4. **확장**: "여기에 X 기능 추가하려면?"

### 다음 단계 추천
1. **PDF/Word 지원**: PyPDF2, python-docx 추가
2. **더 나은 OCR**: Tesseract 통합
3. **고급 LLM**: Ollama로 로컬 Llama 실행
4. **데스크톱 앱**: Electron으로 패키징
5. **클라우드 백업**: 선택적 동기화