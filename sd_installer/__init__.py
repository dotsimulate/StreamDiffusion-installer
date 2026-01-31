"""
StreamDiffusionTD Installer Module

A standalone installer that can be run outside of TouchDesigner for testing,
debugging, and repairing StreamDiffusion installations.

Usage:
    python -m sd_installer install --cuda cu128
    python -m sd_installer verify
    python -m sd_installer diagnose
    python -m sd_installer repair
"""

__version__ = "0.3.1"

from .installer import Installer
from .verifier import Verifier

__all__ = ["Installer", "Verifier", "__version__"]
