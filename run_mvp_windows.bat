@echo off
title Research File Manager MVP
echo ========================================
echo Research File Manager MVP - Windows
echo AI-Powered File Organization ^& Search
echo ========================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo ERROR: Virtual environment not found
    echo Please run fix_windows_setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Create required directories
if not exist data\projects mkdir data\projects
if not exist data\db mkdir data\db
if not exist logs mkdir logs

REM Set Python path
set PYTHONPATH=%PYTHONPATH%;%cd%\backend

REM Start the application
echo.
echo Starting Research File Manager...
echo.
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python backend\main.py

pause