@echo off
setlocal enabledelayedexpansion
title Heraldo - One-Click Installer

echo ===================================================
echo               Heraldo One-Click Installer
echo ===================================================
echo.

:: Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] 'uv' package manager not found.
    echo [INFO] Downloading and installing uv...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"

    :: Add common uv installation locations to current session PATH
    if exist "%USERPROFILE%\.cargo\bin" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    if exist "%LOCALAPPDATA%\bin" set "PATH=%LOCALAPPDATA%\bin;%PATH%"
    if exist "%USERPROFILE%\.local\bin" set "PATH=%USERPROFILE%\.local\bin;%PATH%"

    where uv >nul 2>nul
    if !ERRORLEVEL! NEQ 0 (
        echo.
        echo [ERROR] 'uv' was installed but could not be found in PATH.
        echo Please restart your terminal window and run install.bat again.
        pause
        exit /b 1
    )
)

echo [INFO] 'uv' detected successfully.
echo [INFO] Setting up virtual environment and installing dependencies...
echo.

uv sync

if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERROR] Installation failed during 'uv sync'.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo [SUCCESS] Heraldo installation complete!
echo.
echo To run Python with Heraldo:
echo   Double-click run.bat or execute 'uv run python'
echo ===================================================
echo.

pause
