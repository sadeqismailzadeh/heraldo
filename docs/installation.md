# Installation

`heraldo` requires **Python 3.13** and is managed with [uv](https://docs.astral.sh/uv/), a fast Python package and project manager. We mainly use `uv` for environment management and dependency locking to ensure reproducibility. Two installation paths are available: a one-click script for Windows, or a manual `uv` setup that works on any platform.



## Option 1: One-Click Installer (Windows)

If you're on Windows, the fastest way to get started is the bundled `install.bat` script at the root of the repository.

1. Download or clone the repository.
2. Double-click `install.bat` (or run it from a terminal).
3. The script will:
   - Check whether `uv` is installed, and install it automatically via PowerShell if it's missing.
   - Run `uv sync` to create a virtual environment and install all dependencies.
4. Once you see `[SUCCESS] Heraldo installation complete!`, you're ready to go.

> **Note**: If `uv` was not previously installed and the script downloaded it automatically, but `uv` commands are not recognized in your current session, restart your terminal window and run `install.bat` again so that the updated system `PATH` takes effect.


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
   git clone https://github.com/sadeqismailzadeh/GKP_state_code
   cd GKP_state_code
   ```

3. Install dependencies and create the virtual environment:

   ```bash
   uv sync
   ```

   
> **Virtual Environment Note**: Both installation options create a standard Python virtual environment in the `.venv` directory with `pip` available. If you are not familiar with `uv`, you can simply activate this virtual environment and use `pip` as usual, or continue using `uv` to manage packages in the environment.

## Verifying the Installation

Run the following in Windows Command Prompt (`cmd`) with the virtual environment activated to confirm `heraldo` imports correctly:

```batch
:: Activate virtual environment in Command Prompt
.venv\Scripts\activate

:: Verify heraldo import
python -c "import heraldo; print('heraldo installed successfully')"
```

Alternatively, you can run the command using `uv` without manually activating the environment:

```bash
uv run python -c "import heraldo; print('heraldo installed successfully')"
```

On import, `heraldo` applies a small compatibility patch to `scipy.integrate` (aliasing `simps` to `simpson` on newer SciPy versions). Seeing a patch message printed on import is expected and confirms the package loaded.

## Running an Example Script

To run an example script (such as `scripts/example.py`) after activating the virtual environment in Windows Command Prompt:

```batch
:: Activate the virtual environment
.venv\Scripts\activate

:: Run the script
python scripts\example.py
```

You can also run it directly using `uv`:

```bash
uv run python scripts/example.py
```

## Requirements

- Python `3.13.*` (enforced via `pyproject.toml`)
- Core dependencies (installed automatically by `uv sync`): `strawberryfields`, `numpy`, `scipy`, `numba`, `thewalrus`, `sympy`, `pandas`, `tqdm`, `matplotlib`

## Next Steps

Continue to [Quickstart](quickstart.md) for a walkthrough of your first optimization run.