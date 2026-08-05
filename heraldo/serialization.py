import importlib
import inspect
from typing import Dict, Any, Union, Optional, List
from types import ModuleType
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

def create_from_config(config: Dict[str, Any], module_target: Optional[Union[str, ModuleType]] = None) -> Any:
    """Dynamically instantiates a class from a configuration dictionary.

    Args:
        config (dict): A dictionary containing:
            - 'class_name' (str): The name of the class to instantiate.
            - 'module' (str, optional): The Python module path if module_target is not passed.
            - 'params' (dict, optional): Keyword arguments to pass to the constructor.
        module_target (str or ModuleType, optional): Override module where the class is defined.

    Returns:
        Any: The instantiated object.

    Raises:
        ValueError: If 'class_name' is missing from config or module cannot be resolved.
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    if not isinstance(config, dict) or 'class_name' not in config:
        raise ValueError("Configuration dictionary must contain a 'class_name' key.")

    class_name = config['class_name']
    params = config.get('params', {})

    target_mod = module_target or config.get('module')

    if target_mod is None:
        for mod_str in ['heraldo.components.circuits', 'heraldo.components.targets', 'heraldo.components.objectives']:
            try:
                mod = importlib.import_module(mod_str)
                if hasattr(mod, class_name):
                    cls = getattr(mod, class_name)
                    return cls(**params)
            except ImportError:
                pass
        raise ValueError(f"Module not specified in config and class '{class_name}' not found in default modules.")

    # Resolve module
    if isinstance(target_mod, str):
        try:
            module = importlib.import_module(target_mod)
        except ImportError as e:
            raise ImportError(f"Could not import module '{target_mod}': {e}")
    else:
        module = target_mod

    # Resolve class
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        module_name = getattr(module, '__name__', str(module))
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.")

    # Instantiate
    return cls(**params)


def to_config(obj: Any) -> Any:
    """Inverse factory function: serializes an object or list of objects into configuration dictionaries.

    Inspects the object's class, module, and __init__ parameters/attributes to construct a
    dictionary compatible with create_from_config.

    Args:
        obj (Any): An object instance (e.g. circuit or target generator) or a list/tuple of instances.

    Returns:
        dict or list[dict] or None: Configuration dictionary or list of dictionaries.
    """
    if obj is None:
        return None

    if isinstance(obj, (list, tuple)):
        return [to_config(item) for item in obj]

    if hasattr(obj, "to_config") and callable(obj.to_config):
        return obj.to_config()

    if hasattr(obj, "get_config") and callable(obj.get_config):
        return obj.get_config()

    class_name = obj.__class__.__name__
    module_name = obj.__class__.__module__

    params = {}
    try:
        sig = inspect.signature(obj.__init__)
        for param_name in sig.parameters:
            if param_name in ('self', 'args', 'kwargs'):
                continue
            if hasattr(obj, param_name):
                val = getattr(obj, param_name)
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                elif isinstance(val, np.ndarray):
                    val = val.tolist()
                params[param_name] = val
    except (TypeError, ValueError):
        pass

    return {
        'class_name': class_name,
        'module': module_name,
        'params': params
    }




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

