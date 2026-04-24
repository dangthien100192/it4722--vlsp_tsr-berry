import importlib.util

def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None
