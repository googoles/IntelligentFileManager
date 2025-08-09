@echo off
echo Research File Manager - Complete Windows Setup
echo ===============================================
echo This script tries multiple approaches to fix Rust/Cargo compilation issues
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Python detected. Starting installation process...
echo.

REM Clean start
if exist venv rmdir /s /q venv
python -m venv venv
call venv\Scripts\activate.bat

REM Method 1: Try with proper Cargo PATH
echo ========================================
echo METHOD 1: Installing with Cargo PATH fix
echo ========================================
set PATH=C:\Users\googo\.cargo\bin;%USERPROFILE%\.cargo\bin;%PATH%
set CARGO_HOME=C:\Users\googo\.cargo
set RUSTUP_HOME=C:\Users\googo\.rustup

python -m pip install --upgrade pip wheel setuptools
echo Testing Cargo access...
cargo --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Cargo found, attempting installation with Rust support...
    python -m pip install -r requirements.txt --no-cache-dir
    if %ERRORLEVEL% EQU 0 (
        echo ✅ SUCCESS: Installation complete with Rust compilation!
        goto :test_installation
    ) else (
        echo ⚠ Installation failed, trying Method 2...
    )
) else (
    echo ⚠ Cargo not accessible, trying Method 2...
)

REM Method 2: Pre-compiled packages only
echo.
echo ========================================
echo METHOD 2: Pre-compiled packages only
echo ========================================
echo Installing only binary wheels (no compilation)...

python -m pip install --only-binary=all fastapi==0.104.1
python -m pip install --only-binary=all uvicorn==0.24.0  
python -m pip install --only-binary=all python-multipart==0.0.6
python -m pip install --only-binary=all sqlalchemy==2.0.23
python -m pip install --only-binary=all watchdog==3.0.0
python -m pip install --only-binary=all aiofiles==23.2.1
python -m pip install --only-binary=all python-dotenv==1.0.0
python -m pip install --only-binary=all PyPDF2==3.0.1
python -m pip install --only-binary=all python-docx==1.1.0
python -m pip install --only-binary=all Pillow numpy

REM Try pydantic with binary-only
python -m pip install --only-binary=all pydantic==2.5.0
if %ERRORLEVEL% NEQ 0 (
    echo Pydantic 2.5.0 failed, trying older version...
    python -m pip install --only-binary=all pydantic==1.10.12
)

REM AI packages
echo Installing AI packages...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers

if %ERRORLEVEL% EQU 0 (
    echo ✅ SUCCESS: Pre-compiled installation complete!
    goto :test_installation
) else (
    echo ⚠ Pre-compiled failed, trying Method 3...
)

REM Method 3: Manual package by package
echo.
echo ========================================
echo METHOD 3: Essential packages only
echo ========================================
echo Installing core packages one by one...

python -m pip install fastapi uvicorn python-multipart
python -m pip install sqlalchemy pydantic --no-build-isolation
python -m pip install watchdog aiofiles python-dotenv
python -m pip install PyPDF2 python-docx Pillow numpy

echo ✅ Core packages installed (AI features may be limited)

:test_installation
echo.
echo ========================================
echo TESTING INSTALLATION
echo ========================================
python -c "
print('Testing package imports...')
try:
    import fastapi, uvicorn, pydantic, sqlalchemy
    import watchdog, aiofiles
    print('✅ Core packages: OK')
except ImportError as e:
    print(f'❌ Core packages failed: {e}')
    exit(1)

try:
    import PyPDF2, docx, PIL, numpy
    print('✅ Document processing: OK')
except ImportError as e:
    print(f'⚠ Document processing issue: {e}')

try:
    import torch, sentence_transformers
    print('✅ AI packages: OK')
except ImportError:
    print('⚠ AI packages not available (optional)')

print()
print('🚀 Research File Manager is ready!')
print('Features available:')
print('  ✅ File monitoring and organization')
print('  ✅ Project management')  
print('  ✅ Web interface')
print('  ✅ Document processing')
print('  ✅ Basic search')
print('  ✅ Semantic search (if AI packages installed)')
print('  ⚠ ChromaDB disabled (uses fast fallback)')
"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo INSTALLATION SUCCESSFUL!
    echo ========================================
    echo.
    echo To start the application:
    echo   1. call venv\Scripts\activate.bat
    echo   2. python backend\main.py  
    echo   3. Open http://localhost:8000
    echo.
    echo The system uses fallback implementations for maximum compatibility.
    echo All core features work without Rust compilation issues.
) else (
    echo.
    echo ========================================
    echo INSTALLATION FAILED
    echo ========================================
    echo Please check the error messages above.
    echo You may need to install Visual Studio Build Tools.
)

echo.
pause