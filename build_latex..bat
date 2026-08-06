@echo off
setlocal enabledelayedexpansion
title Building heraldo LaTeX Documentation...

cd /d "%~dp0"

echo ========================================================
echo            Building heraldo LaTeX Documentation
echo ========================================================
echo.

set "BUILD_CMD="

:: 1. Check if uv is available and ensure docs group dependencies are synced
where uv >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    echo [INFO] 'uv' detected. Syncing documentation dependencies...
    uv sync --group docs
    if !ERRORLEVEL! EQU 0 (
        set "BUILD_CMD=uv run sphinx-build -b latex docs docs\_build\latex"
    )
)

:: 2. If uv build command not set, check if virtualenv sphinx-build exists
if "!BUILD_CMD!"=="" (
    if exist ".venv\Scripts\sphinx-build.exe" (
        echo [INFO] Using virtual environment Sphinx...
        set "BUILD_CMD=.venv\Scripts\sphinx-build.exe -b latex docs docs\_build\latex"
    )
)

:: 3. Fallback to system sphinx-build
if "!BUILD_CMD!"=="" (
    where sphinx-build >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        echo [INFO] Using system sphinx-build...
        set "BUILD_CMD=sphinx-build -b latex docs docs\_build\latex"
    ) else (
        echo [ERROR] 'sphinx-build' command not found!
        echo Please install dependencies via 'uv sync --group docs' or 'pip install -r docs\requirements.txt'.
        echo.
        pause
        exit /b 1
    )
)

:: Clean old LaTeX build
if exist "docs\_build\latex" (
    echo Cleaning previous LaTeX build...
    rmdir /s /q "docs\_build\latex"
)

echo Generating LaTeX source files...
echo Running: !BUILD_CMD!
echo.

!BUILD_CMD!

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] LaTeX generation failed. Please check log output above.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo [SUCCESS] LaTeX files generated in 'docs\_build\latex\'!
echo ========================================================
echo.



:: Check if xelatex / latexmk is installed
where latexmk >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    echo [INFO] 'latexmk' detected. Compiling PDF with XeLaTeX...
    cd docs\_build\latex
    latexmk -pdfxe heraldo.tex
    cd ..\..\..
    if exist "docs\_build\latex\heraldo.pdf" (
        echo.
        echo [SUCCESS] PDF created: docs\_build\latex\heraldo.pdf
        start "" "docs\_build\latex\heraldo.pdf"
        pause
        exit /b 0
    )
)

where xelatex >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    echo [INFO] 'xelatex' detected. Compiling PDF...
    cd docs\_build\latex
    xelatex.exe -synctex=1 -interaction=nonstopmode heraldo.tex
	xelatex.exe -synctex=1 -interaction=nonstopmode heraldo.tex
    cd ..\..\..
    if exist "docs\_build\latex\heraldo.pdf" (
        echo.
        echo [SUCCESS] PDF created: docs\_build\latex\heraldo.pdf
        start "" "docs\_build\latex\heraldo.pdf"
        pause
        exit /b 0
    )
)

echo [INFO] Opening output directory in Windows Explorer...
explorer "docs\_build\latex"

pause
