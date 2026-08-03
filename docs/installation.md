# Installation

`heraldo` requires **Python 3.13** and is managed with [uv](https://docs.astral.sh/uv/), a fast Python package and project manager. Two installation paths are available: a one-click script for Windows, or a manual `uv` setup that works on any platform.

## Option 1: One-Click Installer (Windows)

If you're on Windows, the fastest way to get started is the bundled `install.bat` script at the root of the repository.

1. Download or clone the repository.
2. Double-click `install.bat` (or run it from a terminal).
3. The script will:
   - Check whether `uv` is installed, and install it automatically via PowerShell if it's missing.
   - Run `uv sync` to create a virtual environment and install all dependencies.
4. Once you see `[SUCCESS] Heraldo installation complete!`, you're ready to go.

To run Python with `heraldo` available, either double-click `run.bat` or run:

```bat
uv run python
```

## Option 2: Manual Installation with uv (Windows / macOS / Linux)

1. Install `uv` if you don't already have it:

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   ```

2. Clone the repository and move into it:

   ```bash
   git clone <repository-url>
   cd heraldo
   ```

3. Install dependencies and create the virtual environment:

   ```bash
   uv sync
   ```

## Verifying the Installation

Run the following to confirm `heraldo` imports correctly:

```bash
uv run python -c "import heraldo; print('heraldo installed successfully')"
```

On import, `heraldo` applies a small compatibility patch to `scipy.integrate` (aliasing `simps` to `simpson` on newer SciPy versions). Seeing a patch message printed on import is expected and confirms the package loaded.

## Requirements

- Python `3.13.*` (enforced via `pyproject.toml`)
- Core dependencies (installed automatically by `uv sync`): `strawberryfields`, `numpy`, `scipy`, `numba`, `thewalrus`, `sympy`, `pandas`, `tqdm`, `matplotlib`

## Next Steps

Continue to [Usage](usage.md) for a walkthrough of your first optimization run.