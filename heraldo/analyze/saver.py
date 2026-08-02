"""Functions for saving and loading optimization run results to/from disk using pickle."""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def save_results(
    results: Dict[str, Any],
    filepath: Optional[Union[str, Path]] = None,
    directory: Optional[Union[str, Path]] = None,
    prefix: str = "run_results",
) -> str:
    """Saves run optimization results into a pickle file.

    Args:
        results (dict): Result dictionary returned by circuit evaluation or runner
            (e.g., from `BasinHoppingRunner.run()`).
        filepath (str or Path, optional): Destination file path for the .pkl file.
            If specified, `directory` and `prefix` arguments are ignored.
        directory (str or Path, optional): Directory to save the file in if `filepath` is not specified.
            Defaults to current working directory if None.
        prefix (str, optional): Prefix for auto-generated timestamp filename.
            Defaults to "run_results".

    Returns:
        str: Absolute file path where the results were saved.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.pkl"
        save_path = Path(directory) / filename if directory else Path.cwd() / filename
    else:
        save_path = Path(filepath)

    if not save_path.suffix:
        save_path = save_path.with_suffix(".pkl")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Results successfully saved to: {save_path.resolve()}")
    return str(save_path.resolve())


def load_results(filepath: Union[str, Path], reconstruct: bool = False) -> Dict[str, Any]:
    """Loads run optimization results from a pickle file.

    Args:
        filepath (str or Path): Path to the pickle (.pkl) file to load.
        reconstruct (bool, optional): If True, reconstructs circuit and target generator
            instances from saved configuration metadata and attaches them as 'circuit'
            and 'targets' keys in the returned dictionary. Defaults to False.

    Returns:
        dict: The loaded optimization result dictionary.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(path, "rb") as f:
        results = pickle.load(f)

    if reconstruct:
        reconstructed = reconstruct_objects(results)
        results.update(reconstructed)

    return results


def reconstruct_objects(results: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstructs circuit and target generator instances from results configuration metadata.

    Args:
        results (dict): Result dictionary containing 'circuit_config' and/or 'target_configs'.

    Returns:
        dict: Dictionary with 'circuit' and/or 'targets' instantiated objects.
    """
    from heraldo.factory import create_from_config

    out = {}
    if "circuit_config" in results and results["circuit_config"]:
        out["circuit"] = create_from_config(results["circuit_config"])

    if "target_configs" in results and results["target_configs"]:
        tc = results["target_configs"]
        if isinstance(tc, list):
            out["targets"] = [create_from_config(item) for item in tc]
        else:
            out["targets"] = create_from_config(tc)

    return out
