"""
PyTorch wrapper to prevent file watcher errors in Streamlit.
Import torch through this module to avoid runtime errors with '__path__._path'
"""

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

# Re-export the most commonly used PyTorch components
__all__ = ['torch', 'nn', 'models', 'OrderedDict']