"""Person segmentation with U^2-Net human-seg (ONNX, 320x320), no rembg needed.
~0.5-0.9 s per image on 2 CPU cores."""
import numpy as np
from PIL import Image

from .models import fetch, session

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_sess = None


def person_mask(img, folder=None):
    """PIL image -> PIL 'L' mask at the input size (255 = person)."""
    global _sess
    if _sess is None:
        _sess = session(fetch('u2net_human_seg', **({'folder': folder} if folder else {})))
    rgb = img.convert('RGB')
    x = np.asarray(rgb.resize((320, 320), Image.LANCZOS), np.float32)
    x = x / max(1e-6, x.max())                       # rembg's normalisation: x / max(x)
    x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
    out = _sess.run(None, {_sess.get_inputs()[0].name: x})[0][0, 0]
    out = (out - out.min()) / max(1e-6, out.max() - out.min())
    return Image.fromarray((out * 255).astype(np.uint8), 'L').resize(rgb.size, Image.LANCZOS)


def cutout(img, mask):
    """RGBA image with the mask as alpha."""
    rgba = img.convert('RGBA')
    rgba.putalpha(mask)
    return rgba
