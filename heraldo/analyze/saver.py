"""Functions for saving and loading optimization run results to/from disk using pickle."""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def save_results(
    results: Dict[str, Any],
    filepath: Optional[Union[str, Path]] = None,
) -> str:
    """Saves run optimization results into a pickle file.

    Args:
        results (dict): Result dictionary returned by circuit evaluation or runner
            (e.g., from `BasinHoppingRunner.run()`).
        filepath (str or Path, optional): Destination file path for the .pkl file.
            If None, defaults to 'run_results_<timestamp>.pkl' in the current working directory.

    Returns:
        str: Absolute file path where the results were saved.
    """
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path.cwd() / f"run_results_{timestamp}.pkl"
    else:
        save_path = Path(filepath)

    if not save_path.suffix:
        save_path = save_path.with_suffix(".pkl")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a shallow copy and remove live object instances before pickling
    save_dict = results.copy()
    for key in ("circuit", "targets", "loss_fn"):
        save_dict.pop(key, None)

    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Results successfully saved to: {save_path.resolve()}")
    return str(save_path.resolve())


def load_results(filepath: Union[str, Path], reconstruct: bool = True) -> Dict[str, Any]:
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
    """Reconstructs circuit, target generator, and loss function instances from results configuration metadata.

    Args:
        results (dict): Result dictionary containing 'circuit_config', 'target_configs', and/or 'loss_config'.

    Returns:
        dict: Dictionary with 'circuit', 'targets', and/or 'loss_fn' instantiated objects.
    """
    from heraldo.serialization import create_from_config

    out = {}
    if "circuit_config" in results and results["circuit_config"]:
        out["circuit"] = create_from_config(results["circuit_config"])

    if "target_configs" in results and results["target_configs"]:
        tc = results["target_configs"]
        if isinstance(tc, list):
            out["targets"] = [create_from_config(item) for item in tc]
        else:
            out["targets"] = create_from_config(tc)

    if "loss_config" in results and results["loss_config"]:
        out["loss_fn"] = create_from_config(results["loss_config"])

    return out
