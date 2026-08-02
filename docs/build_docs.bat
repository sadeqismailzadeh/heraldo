@echo off
title Building heraldo Documentation...
cd /d "%~dp0"

echo ========================================================
echo            Building heraldo HTML Documentation
echo ========================================================
echo.

REM Check if sphinx is installed
where sphinx-build >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] sphinx-build command not found!
    echo Please install Sphinx requirements by running:
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Clean old build
if exist "_build" (
    echo Cleaning previous build...
    rmdir /s /q "_build"
)

echo Building HTML documentation using MyST Markdown...
call make.bat html

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Documentation build failed. Please check log output above.
    echo.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Documentation built successfully!
echo Opening index.html in your default browser...
start "" "_build\html\index.html"

pause