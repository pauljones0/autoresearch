import pkgutil
import importlib
import pytest
import os
import sys

# Add project root to sys.path to ensure modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_all_modules():
    modules = []
    for pkg in ["bandit", "gpu_kernels", "model_scientist", "surrogate_triage", "meta"]:
        for module_info in pkgutil.walk_packages([pkg], prefix=pkg + "."):
            modules.append(module_info.name)
    return modules

@pytest.mark.parametrize("module_name", get_all_modules())
def test_import_all(module_name):
    importlib.import_module(module_name)