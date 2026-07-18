"""Authoritative filesystem locations shared by project compatibility shims."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SHARED_AIRCRAFT_TOOLS_ROOT = PROJECT_ROOT.parent / "aircraft-tools"
