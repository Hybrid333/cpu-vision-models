"""cpuvision - depth, x4 upscaling and person masks on CPU, with weights from
GitHub releases and no PyTorch / Hugging Face dependency."""
from .depth import colorize, depth
from .segment import cutout, person_mask
from .upscale import upscale

__all__ = ['depth', 'colorize', 'person_mask', 'cutout', 'upscale']
__version__ = '0.1.0'
