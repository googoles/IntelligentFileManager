@echo off
echo ========================================
echo Windows Setup Fix for Research File Manager
echo ========================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [1/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo.
echo [2/6] Activating virtual environment...
call venv\Scripts\activate

echo.
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/6] Installing core dependencies...
pip install fastapi uvicorn[standard] python-multipart sqlalchemy pydantic watchdog aiofiles python-dotenv

echo.
echo [5/6] Installing ML dependencies (Windows-compatible)...
pip install sentence-transformers numpy scikit-learn

echo.
echo [6/6] Attempting ChromaDB installation...
echo Note: If ChromaDB fails, the app will use the simplified search engine

REM Try different ChromaDB installation methods
pip install chromadb --no-build-isolation >nul 2>&1
if errorlevel 1 (
    echo ChromaDB standard installation failed, trying alternative...
    pip install chromadb-client >nul 2>&1
    if errorlevel 1 (
        echo.
        echo WARNING: ChromaDB could not be installed
        echo The application will use the simplified search engine (search_simple.py)
        echo This provides the same functionality without Rust compilation requirements
    ) else (
        echo ChromaDB client installed successfully
    )
) else (
    echo ChromaDB installed successfully
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. The setup is complete with Windows-compatible dependencies
echo 2. If ChromaDB failed to install, the app will automatically use
echo    the simplified search engine which works perfectly on Windows
echo 3. Run the application with: run_mvp_windows.bat
echo.
pause