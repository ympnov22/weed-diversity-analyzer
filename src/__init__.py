"""
Weed Diversity Analyzer

A tool for analyzing weed species diversity from field images using deep learning
and ecological diversity metrics.
"""

__version__ = "1.0.0"
__author__ = "Devin AI"
__email__ = "devin-ai-integration[bot]@users.noreply.github.com"

from .utils.config import ConfigManager
from .utils.logger import setup_logger

config = ConfigManager()
logger = setup_logger(__name__)

__all__ = [
    "config",
    "logger",
    "ConfigManager",
    "setup_logger",
]
