@ECHO OFF

pushd %~dp0

REM Command line options for sphinx-build
if "%SPHINXBUILD%" == "" (
	where uv >nul 2>nul
	if not errorlevel 1 (
		set SPHINXBUILD=uv run sphinx-build
	) else if exist "..\.venv\Scripts\sphinx-build.exe" (
		set SPHINXBUILD=..\.venv\Scripts\sphinx-build.exe
	) else (
		set SPHINXBUILD=sphinx-build
	)
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then add the directory where it was installed to your
	echo.PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
