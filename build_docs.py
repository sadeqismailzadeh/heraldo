"""
HTML Documentation Generator Script.
Works on Windows, Linux, and macOS.
"""
import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def build_docs():
    project_root = Path(__file__).parent.resolve()
    docs_dir = project_root / "docs"
    build_dir = docs_dir / "_build" / "html"

    print("===================================================")
    print("  Heraldo Documentation Generator")
    print("===================================================")

    # 1. Install Sphinx requirements
    req_file = docs_dir / "requirements.txt"
    if req_file.exists():
        print("[1/4] Checking and installing Sphinx dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)

    # 2. Run sphinx-apidoc to auto-generate RST files
    print("[2/4] Auto-generating API reStructuredText files...")
    apidoc_cmd = [
        sys.executable,
        "-m",
        "sphinx.ext.apidoc",
        "-f",
        "-o",
        str(docs_dir),
        str(project_root / "heraldo"),
    ]
    subprocess.run(apidoc_cmd, check=True)

    # 3. Build HTML documentation
    print("[3/4] Building HTML documentation...")
    sphinx_cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        str(docs_dir),
        str(build_dir),
    ]
    subprocess.run(sphinx_cmd, check=True)

    # 4. Open in default browser
    index_html = build_dir / "index.html"
    print("\n[4/4] Documentation built successfully!")
    print(f"Location: {index_html}")

    if index_html.exists():
        print("Opening in default browser...")
        webbrowser.open(index_html.as_uri())


if __name__ == "__main__":
    build_docs()