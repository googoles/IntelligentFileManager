# MVP 실행, 배포 및 확장 가이드

## 🎯 즉시 실행 가능한 One-Click 설정

### 자동 설치 스크립트 (setup.py)
```python
#!/usr/bin/env python3
"""
Research File Manager MVP - 원클릭 설치 스크립트
실행: python setup.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class MVPSetup:
    def __init__(self):
        self.root_dir = Path.cwd()
        self.venv_dir = self.root_dir / "venv"
        self.is_windows = platform.system() == "Windows"
        
    def print_banner(self):
        """설치 배너 출력"""
        print("""
        ╔══════════════════════════════════════════════╗
        ║   Research File Manager MVP - Auto Setup    ║
        ║           🚀 원클릭 자동 설치 🚀            ║
        ╚══════════════════════════════════════════════╝
        """)
        
    def check_python(self):
        """Python 버전 확인"""
        print("📌 Python 버전 확인...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 이상이 필요합니다!")
            sys.exit(1)
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        
    def create_venv(self):
        """가상환경 생성"""
        print("\n📌 가상환경 생성...")
        if self.venv_dir.exists():
            print("⚠️  기존 가상환경 발견, 건너뜀")
            return
            
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ 가상환경 생성 완료")
        
    def get_pip_cmd(self):
        """OS별 pip 경로 반환"""
        if self.is_windows:
            return str(self.venv_dir / "Scripts" / "pip.exe")
        return str(self.venv_dir / "bin" / "pip")
        
    def install_requirements(self):
        """패키지 설치"""
        print("\n📌 필수 패키지 설치...")
        pip_cmd = self.get_pip_cmd()
        
        requirements = [
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0",
            "watchdog==3.0.0",
            "sentence-transformers==2.2.2",
            "chromadb==0.4.18",
            "sqlalchemy==2.0.23",
            "aiofiles==23.2.1",
            "python-multipart==0.0.6",
            "pydantic==2.5.0",
            "python-dotenv==1.0.0"
        ]
        
        # 한 번에 설치 (더 빠름)
        subprocess.run([pip_cmd, "install"] + requirements)
        print("✅ 패키지 설치 완료")
        
    def create_directory_structure(self):
        """디렉토리 구조 생성"""
        print("\n📌 프로젝트 구조 생성...")
        
        directories = [
            "backend",
            "frontend", 
            "data/projects",
            "data/db",
            "data/db/chroma",
            "logs",
            "tests"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        print("✅ 디렉토리 구조 생성 완료")
        
    def create_env_file(self):
        """환경 변수 파일 생성"""
        print("\n📌 환경 설정 파일 생성...")
        
        env_content = """# Research File Manager MVP Configuration
ENV=development
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=sqlite:///data/db/research.db

# Model Settings  
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_CHUNK_SIZE=500

# File Processing
SUPPORTED_EXTENSIONS=.txt,.md,.py,.js,.json,.csv,.pdf
MAX_FILE_SIZE_MB=100

# Optional API Keys (클라우드 기능용)
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
            
        print("✅ .env 파일 생성 완료")
        
    def download_core_files(self):
        """핵심 파일 다운로드/생성"""
        print("\n📌 핵심 파일 생성...")
        
        # 여기서는 파일을 직접 생성
        # 실제로는 GitHub에서 다운로드하거나 템플릿 사용
        
        # requirements.txt 생성
        with open("requirements.txt", "w") as f:
            f.write("""fastapi==0.104.1
uvicorn[standard]==0.24.0
watchdog==3.0.0
sentence-transformers==2.2.2
chromadb==0.4.18
sqlalchemy==2.0.23
aiofiles==23.2.1
python-multipart==0.0.6
pydantic==2.5.0
python-dotenv==1.0.0
pypdf2==3.0.1
python-docx==1.1.0
easyocr==1.7.1
ollama==0.1.7""")
        
        print("✅ requirements.txt 생성 완료")
        
    def create_run_scripts(self):
        """실행 스크립트 생성"""
        print("\n📌 실행 스크립트 생성...")
        
        # Windows 배치 파일
        with open("run.bat", "w") as f:
            f.write("""@echo off
echo Starting Research File Manager MVP...
call venv\\Scripts\\activate
python backend\\main.py
pause""")
            
        # Unix 쉘 스크립트  
        with open("run.sh", "w") as f:
            f.write("""#!/bin/bash
echo "Starting Research File Manager MVP..."
source venv/bin/activate
python backend/main.py""")
            
        if not self.is_windows:
            os.chmod("run.sh", 0o755)
            
        print("✅ 실행 스크립트 생성 완료")
        
    def download_models(self):
        """ML 모델 사전 다운로드"""
        print("\n📌 ML 모델 다운로드 (첫 실행 속도 향상)...")
        
        try:
            from sentence_transformers import SentenceTransformer
            print("   다운로드 중... (약 90MB)")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ 임베딩 모델 다운로드 완료")
        except Exception as e:
            print(f"⚠️  모델 다운로드 실패 (첫 실행시 자동 다운로드): {e}")
            
    def create_sample_data(self):
        """샘플 데이터 생성"""
        print("\n📌 샘플 프로젝트 생성...")
        
        sample_dir = Path("data/projects/sample_project")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 샘플 파일들
        samples = {
            "README.md": "# Sample Research Project\n\nThis is a sample project for testing.",
            "data_analysis.py": "import pandas as pd\n# Sample analysis code\ndf = pd.read_csv('data.csv')",
            "notes.txt": "Research notes:\n- Experiment 1: Success\n- TODO: Review literature",
            "results.json": '{"accuracy": 0.95, "loss": 0.05}'
        }
        
        for filename, content in samples.items():
            (sample_dir / filename).write_text(content)
            
        print("✅ 샘플 프로젝트 생성 완료")
        
    def print_next_steps(self):
        """다음 단계 안내"""
        run_cmd = "run.bat" if self.is_windows else "./run.sh"
        
        print(f"""
        ╔══════════════════════════════════════════════╗
        ║            ✨ 설치 완료! ✨                 ║
        ╚══════════════════════════════════════════════╝
        
        🎯 실행 방법:
        {run_cmd}
        
        또는:
        
        {'venv\\Scripts\\activate' if self.is_windows else 'source venv/bin/activate'}
        python backend/main.py
        
        📱 브라우저에서 접속:
        http://localhost:8000
        
        📚 샘플 프로젝트:
        data/projects/sample_project/
        
        💡 도움말:
        - Ctrl+C로 서버 중지
        - logs/ 폴더에서 로그 확인
        - .env 파일에서 설정 변경
        """)
        
    def run(self):
        """전체 설치 프로세스 실행"""
        try:
            self.print_banner()
            self.check_python()
            self.create_venv()
            self.install_requirements()
            self.create_directory_structure()
            self.create_env_file()
            self.download_core_files()
            self.create_run_scripts()
            self.download_models()
            self.create_sample_data()
            self.print_next_steps()
            
            print("\n🎉 설치가 성공적으로 완료되었습니다!")
            
        except Exception as e:
            print(f"\n❌ 설치 중 오류 발생: {e}")
            print("문제 해결을 위해 수동 설치를 시도하세요.")
            sys.exit(1)

if __name__ == "__main__":
    setup = MVPSetup()
    setup.run()
```

## 🐳 Docker를 통한 원클릭 배포

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 데이터 디렉토리 생성
RUN mkdir -p data/projects data/db logs

# 포트 노출
EXPOSE 8000

# 실행
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  research-manager:
    build: .
    container_name: research-file-manager
    ports:
      - "8000:8000"
    volumes:
      # 데이터 영속성
      - ./data:/app/data
      - ./logs:/app/logs
      # 호스트 파일 시스템 마운트 (선택적)
      - ~/Documents/Research:/app/external/research
    environment:
      - ENV=production
      - DATABASE_URL=sqlite:///app/data/db/research.db
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
    restart: unless-stopped
    networks:
      - research-net

  # 선택적: PostgreSQL (확장성을 위해)
  postgres:
    image: postgres:15-alpine
    container_name: research-db
    environment:
      POSTGRES_USER: research
      POSTGRES_PASSWORD: secure_password
      POSTGRES_DB: research_files
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - research-net
    profiles:
      - full

networks:
  research-net:
    driver: bridge

volumes:
  postgres_data:
```

## 🔧 즉시 사용 가능한 확장 기능

### 1. PDF 텍스트 추출 (5분 추가)
```python
# backend/pdf_processor.py
import PyPDF2
from typing import Optional

class PDFProcessor:
    @staticmethod
    def extract_text(file_path: str) -> Optional[str]:
        """PDF에서 텍스트 추출"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"PDF 추출 실패: {e}")
            return None
    
    @staticmethod
    def extract_metadata(file_path: str) -> dict:
        """PDF 메타데이터 추출"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                info = reader.metadata
                
                return {
                    'title': info.get('/Title', ''),
                    'author': info.get('/Author', ''),
                    'subject': info.get('/Subject', ''),
                    'pages': len(reader.pages)
                }
        except:
            return {}
```

### 2. 실시간 협업 (WebSocket) - 10분 추가
```python
# backend/websocket_manager.py
from fastapi import WebSocket
from typing import List, Dict
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, project_id: str):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
        self.active_connections[project_id].append(websocket)
        
    def disconnect(self, websocket: WebSocket, project_id: str):
        self.active_connections[project_id].remove(websocket)
        
    async def broadcast_file_change(self, project_id: str, message: dict):
        """프로젝트 내 모든 사용자에게 파일 변경 알림"""
        if project_id in self.active_connections:
            for connection in self.active_connections[project_id]:
                await connection.send_json(message)

# main.py에 추가
manager = ConnectionManager()

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await manager.connect(websocket, project_id)
    try:
        while True:
            data = await websocket.receive_text()
            # 메시지 처리
    except:
        manager.disconnect(websocket, project_id)
```

### 3. Ollama 로컬 LLM 통합 (15분 추가)
```python
# backend/llm_service.py
import ollama
from typing import Optional, List
import asyncio

class LocalLLMService:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model = model_name
        self.client = ollama.Client()
        
    async def summarize_document(self, content: str) -> str:
        """문서 요약"""
        prompt = f"""Summarize the following document in 3-5 bullet points:

{content[:2000]}

Summary:"""
        
        response = await asyncio.to_thread(
            self.client.chat,
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response['message']['content']
    
    async def extract_keywords(self, content: str) -> List[str]:
        """키워드 추출"""
        prompt = f"""Extract 5-10 key technical terms from this text:

{content[:1000]}

Keywords (comma-separated):"""
        
        response = await asyncio.to_thread(
            self.client.chat,
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        keywords = response['message']['content'].split(',')
        return [k.strip() for k in keywords]
    
    async def suggest_file_category(self, content: str, filename: str) -> str:
        """파일 카테고리 제안"""
        prompt = f"""Given this filename and content, suggest the best category:
Filename: {filename}
Content preview: {content[:500]}

Categories: literature, data, code, results, notes, other

Best category:"""
        
        response = await asyncio.to_thread(
            self.client.chat,
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response['message']['content'].strip().lower()
```

## 📱 데스크톱 앱 패키징 (Electron)

### electron-app/package.json
```json
{
  "name": "research-file-manager",
  "version": "1.0.0",
  "description": "AI-Powered Research File Manager",
  "main": "main.js",
  "scripts": {
    "start": "electron .",
    "build-win": "electron-builder --win",
    "build-mac": "electron-builder --mac",
    "build-linux": "electron-builder --linux"
  },
  "devDependencies": {
    "electron": "^27.0.0",
    "electron-builder": "^24.0.0"
  },
  "build": {
    "appId": "com.research.filemanager",
    "productName": "Research File Manager",
    "directories": {
      "output": "dist"
    },
    "files": [
      "main.js",
      "preload.js",
      "renderer/**/*",
      "backend/**/*",
      "!backend/__pycache__"
    ],
    "win": {
      "target": "nsis",
      "icon": "assets/icon.ico"
    },
    "mac": {
      "target": "dmg",
      "icon": "assets/icon.icns"
    },
    "linux": {
      "target": "AppImage",
      "icon": "assets/icon.png"
    }
  }
}
```

### electron-app/main.js
```javascript
const { app, BrowserWindow, Menu, Tray, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;
let tray;

// Python 백엔드 시작
function startPythonBackend() {
  const script = path.join(__dirname, '../backend/main.py');
  const python = process.platform === 'win32' ? 'python' : 'python3';
  
  pythonProcess = spawn(python, [script]);
  
  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python: ${data}`);
  });
  
  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python Error: ${data}`);
  });
}

// 메인 윈도우 생성
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    icon: path.join(__dirname, 'assets/icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#1e1e1e'
  });

  // 백엔드가 준비될 때까지 대기
  setTimeout(() => {
    mainWindow.loadURL('http://localhost:8000');
  }, 3000);

  // 개발자 도구 (개발 모드에서만)
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
}

// 시스템 트레이
function createTray() {
  tray = new Tray(path.join(__dirname, 'assets/tray-icon.png'));
  
  const contextMenu = Menu.buildFromTemplate([
    { label: '열기', click: () => mainWindow.show() },
    { label: '설정', click: () => shell.openPath(path.join(__dirname, '../.env')) },
    { type: 'separator' },
    { label: '종료', click: () => app.quit() }
  ]);
  
  tray.setToolTip('Research File Manager');
  tray.setContextMenu(contextMenu);
  
  tray.on('click', () => {
    mainWindow.isVisible() ? mainWindow.hide() : mainWindow.show();
  });
}

// 앱 시작
app.whenReady().then(() => {
  startPythonBackend();
  createWindow();
  createTray();
});

// 앱 종료 처리
app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
```

## 🏃‍♂️ 빠른 시작 명령어 모음

### 1. Git Clone & 자동 설치 (1분)
```bash
# GitHub에서 클론 (실제 레포지토리 URL로 변경)
git clone https://github.com/yourusername/research-file-manager.git
cd research-file-manager

# 자동 설치
python setup.py

# 실행
./run.sh  # Mac/Linux
run.bat   # Windows
```

### 2. Docker 실행 (2분)
```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

### 3. 개발 모드 실행
```bash
# 가상환경 활성화
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 개발 서버 실행 (자동 리로드)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 별도 터미널에서 프론트엔드 개발 서버
cd frontend
python -m http.server 3000
```

## 📊 성능 모니터링 대시보드

### backend/monitoring.py
```python
from datetime import datetime, timedelta
from typing import Dict, List
import psutil
import asyncio
from collections import deque

class PerformanceMonitor:
    def __init__(self, max_history: int = 100):
        self.metrics_history = deque(maxlen=max_history)
        self.start_time = datetime.now()
        
    async def collect_metrics(self) -> Dict:
        """시스템 메트릭 수집"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'used': psutil.virtual_memory().used / (1024**3),  # GB
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'used': psutil.disk_usage('/').used / (1024**3),  # GB
                'percent': psutil.disk_usage('/').percent
            },
            'files_indexed': await self.get_indexed_files_count(),
            'active_searches': await self.get_active_searches(),
            'uptime': str(datetime.now() - self.start_time)
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    async def get_indexed_files_count(self) -> int:
        """인덱싱된 파일 수"""
        # DB 쿼리로 실제 구현
        return 1234  # 예시
    
    async def get_active_searches(self) -> int:
        """활성 검색 수"""
        return 5  # 예시
    
    def get_dashboard_data(self) -> Dict:
        """대시보드용 데이터"""
        return {
            'current': self.metrics_history[-1] if self.metrics_history else {},
            'history': list(self.metrics_history),
            'summary': {
                'avg_cpu': sum(m['cpu_percent'] for m in self.metrics_history) / len(self.metrics_history) if self.metrics_history else 0,
                'peak_memory': max((m['memory']['percent'] for m in self.metrics_history), default=0)
            }
        }

# API 엔드포인트 추가
monitor = PerformanceMonitor()

@app.get("/api/metrics")
async def get_metrics():
    """실시간 메트릭"""
    return await monitor.collect_metrics()

@app.get("/api/dashboard")
async def get_dashboard():
    """대시보드 데이터"""
    return monitor.get_dashboard_data()
```

## 🎯 프로덕션 체크리스트

### 보안
- [ ] API 인증 추가 (JWT)
- [ ] HTTPS 설정 (Let's Encrypt)
- [ ] 파일 업로드 검증
- [ ] SQL Injection 방지
- [ ] Rate Limiting

### 성능
- [ ] 인덱스 최적화
- [ ] 캐싱 레이어 (Redis)
- [ ] 비동기 작업 큐 (Celery)
- [ ] CDN 정적 파일

### 모니터링
- [ ] 로깅 시스템 (ELK Stack)
- [ ] 에러 추적 (Sentry)
- [ ] 성능 모니터링 (Prometheus)
- [ ] 알림 시스템

### 백업
- [ ] 자동 백업 스크립트
- [ ] 데이터베이스 복제
- [ ] 파일 시스템 스냅샷
- [ ] 재해 복구 계획

## 💡 트러블슈팅

### 자주 발생하는 문제와 해결

**1. 포트 이미 사용 중**
```bash
# 포트 확인
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# 프로세스 종료 또는 포트 변경
uvicorn backend.main:app --port 8001
```

**2. 모델 다운로드 실패**
```python
# 수동 다운로드
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
```

**3. 메모리 부족**
```python
# 청크 크기 조정
CHUNK_SIZE = 200  # 500에서 감소
BATCH_SIZE = 10   # 배치 처리 크기 감소
```

**4. 파일 권한 오류**
```bash
# Unix/Linux
chmod -R 755 data/
sudo chown -R $USER:$USER data/

# Windows (관리자 권한으로 실행)
```

이제 MVP가 완성되었습니다! 🎉 

하루 안에 제작 가능한 핵심 기능과 함께, 즉시 확장 가능한 구조를 갖추었습니다. Claude Code와 함께라면 각 부분을 빠르게 구현하고 개선할 수 있습니다.