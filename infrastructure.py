from pathlib import Path
import importlib
import sys


_SHARED_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "aircraft-tools"

if not _SHARED_PACKAGE_ROOT.exists():
    raise FileNotFoundError(
        f"Shared package root not found: {_SHARED_PACKAGE_ROOT}"
    )

shared_package_path = str(_SHARED_PACKAGE_ROOT)
if shared_package_path not in sys.path:
    sys.path.insert(0, shared_package_path)

_shared_module = importlib.import_module("aircraft_tools.openvsp_infrastructure")
sys.modules[__name__] = _shared_module
