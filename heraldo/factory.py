import importlib
import inspect
from typing import Dict, Any, Union, Optional, List
from types import ModuleType
import numpy as np


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
        for mod_str in ['heraldo.components.circuits', 'heraldo.components.targets']:
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
