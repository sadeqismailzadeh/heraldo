import importlib
from typing import Dict, Any, Union, Optional
from types import ModuleType

def create_from_config(config: Dict[str, Any], module_target: Union[str, ModuleType]) -> Any:
    """
    Dynamically instantiates a class from a specific module based on a configuration dictionary.

    Args:
        config (dict): A dictionary containing:
            - 'class_name' (str): The name of the class to instantiate.
            - 'params' (dict, optional): Keyword arguments to pass to the constructor.
        module_target (str or ModuleType): The Python module (or dotted string path) 
                                           where the class is defined.

    Returns:
        The instantiated object.

    Raises:
        ValueError: If 'class_name' is missing from config.
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    class_name = config.get('class_name')
    if not class_name:
        raise ValueError("Configuration dictionary must contain a 'class_name' key.")

    params = config.get('params', {})

    # Resolve module
    if isinstance(module_target, str):
        try:
            module = importlib.import_module(module_target)
        except ImportError as e:
            raise ImportError(f"Could not import module '{module_target}': {e}")
    else:
        module = module_target

    # Resolve class
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        module_name = getattr(module, '__name__', str(module))
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.")

    # Instantiate
    return cls(**params)