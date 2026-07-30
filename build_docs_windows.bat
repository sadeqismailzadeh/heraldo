@echo off
echo ===================================================
echo   Heraldo HTML Documentation Generator (Windows)
echo ===================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not added to PATH.
    pause
    exit /b 1
)

REM Step 1: Install Sphinx requirements
REM echo [1/4] Installing Sphinx dependencies...
REM python -m pip install -r docs\requirements.txt --quiet

REM Step 2: Regenerate RST API documents
echo [2/4] Auto-generating API reStructuredText files...
python -m sphinx.ext.apidoc -f -o docs heraldo

REM Step 3: Build HTML documentation
echo [3/4] Building HTML documentation...
call docs\make.bat html

if errorlevel 1 (
    echo.
    echo Error: HTML documentation build failed.
    pause
    exit /b 1
)

REM Step 4: Display output and open in browser
echo [4/4] Documentation built successfully!
echo.
echo Opening documentation in browser...
start "" "docs\_build\html\index.html"

pause