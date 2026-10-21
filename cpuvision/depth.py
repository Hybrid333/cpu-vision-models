"""Monocular depth with MiDaS v2.1 small (ONNX, 256x256). ~60 ms on 2 CPU cores."""
import numpy as np
from PIL import Image

from .models import fetch, session

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_sess = None


def depth(img, folder=None):
    """PIL image -> float32 array (H, W) in 0..1, 1 = near."""
    global _sess
    if _sess is None:
        _sess = session(fetch('midas_small', **({'folder': folder} if folder else {})))
    rgb = img.convert('RGB')
    x = np.asarray(rgb.resize((256, 256), Image.BICUBIC), np.float32) / 255.0
    x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
    inv = _sess.run(None, {_sess.get_inputs()[0].name: x})[0][0]   # inverse depth
    inv = (inv - inv.min()) / max(1e-6, inv.max() - inv.min())
    return np.asarray(Image.fromarray(inv.astype(np.float32), 'F')
                      .resize(rgb.size, Image.BICUBIC), np.float32).clip(0, 1)


def colorize(d):
    """Depth (0..1) -> PIL RGB with a perceptual ramp (dark = far, bright = near)."""
    stops = np.array([[13, 8, 135], [126, 3, 168], [204, 71, 120],
                      [248, 149, 64], [240, 249, 33]], np.float32)
    t = d * (len(stops) - 1)
    i = np.clip(t.astype(int), 0, len(stops) - 2)
    f = (t - i)[..., None]
    return Image.fromarray((stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8))
