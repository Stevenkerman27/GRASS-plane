import sys

from project_paths import SHARED_AIRCRAFT_TOOLS_ROOT

_SHARED_PACKAGE_ROOT = SHARED_AIRCRAFT_TOOLS_ROOT

if not _SHARED_PACKAGE_ROOT.exists():
    raise FileNotFoundError(
        f"Shared package root not found: {_SHARED_PACKAGE_ROOT}"
    )

shared_package_path = str(_SHARED_PACKAGE_ROOT)
if shared_package_path not in sys.path:
    sys.path.insert(0, shared_package_path)

from aircraft_tools.prop_model import *  # noqa: F401,F403
