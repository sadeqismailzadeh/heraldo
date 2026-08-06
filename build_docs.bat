@echo off
setlocal enabledelayedexpansion
title Building heraldo Documentation...

cd /d "%~dp0"

echo ========================================================
echo            Building heraldo HTML Documentation
echo ========================================================
echo.

set "BUILD_CMD="

:: 1. Check if uv is available and ensure docs group dependencies are synced
where uv >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    echo [INFO] 'uv' detected. Syncing documentation dependencies...
    uv sync --group docs
    if !ERRORLEVEL! EQU 0 (
        set "BUILD_CMD=uv run sphinx-build -b html docs docs\_build\html"
    )
)

:: 2. If uv build command not set, check if virtualenv sphinx-build exists
if "!BUILD_CMD!"=="" (
    if exist ".venv\Scripts\sphinx-build.exe" (
        echo [INFO] Using virtual environment Sphinx...
        set "BUILD_CMD=.venv\Scripts\sphinx-build.exe -b html docs docs\_build\html"
    )
)

:: 3. Fallback to system sphinx-build
if "!BUILD_CMD!"=="" (
    where sphinx-build >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        echo [INFO] Using system sphinx-build...
        set "BUILD_CMD=sphinx-build -b html docs docs\_build\html"
    ) else (
        echo [ERROR] 'sphinx-build' command not found!
        echo Please install dependencies via 'uv sync --group docs' or 'pip install -r docs\requirements.txt'.
        echo.
        pause
        exit /b 1
    )
)

:: Clean old build
if exist "docs\_build" (
    echo Cleaning previous build...
    rmdir /s /q "docs\_build"
)

echo Building HTML documentation using MyST Markdown...
echo Running: !BUILD_CMD!
echo.

!BUILD_CMD!

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] Documentation build failed. Please check log output above.
    echo.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Documentation built successfully!
echo Opening index.html in your default browser...
start "" "docs\_build\html\index.html"

pause
