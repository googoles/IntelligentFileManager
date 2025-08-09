@echo off
echo ========================================
echo Clean Start - Research File Manager MVP
echo ========================================
echo.

echo [1/3] Cleaning old database files...
if exist data\db\chroma rmdir /s /q data\db\chroma
if exist data\db\research.db del data\db\research.db
if exist data\db\research.db-shm del data\db\research.db-shm
if exist data\db\research.db-wal del data\db\research.db-wal
if exist data\db\embeddings rmdir /s /q data\db\embeddings

echo [2/3] Creating fresh directories...
mkdir data\projects 2>nul
mkdir data\db 2>nul
mkdir data\db\chroma 2>nul
mkdir logs 2>nul

echo [3/3] Starting application with clean database...
echo.
echo The application will create new database tables and ChromaDB collections.
echo.

call venv\Scripts\activate
python backend\main.py

pause