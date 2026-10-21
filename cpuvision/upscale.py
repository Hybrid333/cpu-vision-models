"""x4 super-resolution with Real-ESRGAN realesr-general-x4v3, run through
onnxruntime from a graph built straight from the .pth (see esrgan_onnx.py).
About 6.5 s for a 464x688 frame on 2 CPU cores - meant for stills and
keyframes, not whole clips."""
import os

import numpy as np
from PIL import Image

from .esrgan_onnx import build
from .models import DEFAULT_DIR, fetch, session

_sess = None


def _model(folder):
    pth = fetch('realesr_general_x4v3', folder)
    onnx_path = os.path.join(folder, 'realesr-general-x4v3.onnx')
    if not os.path.exists(onnx_path):
        build(pth, onnx_path)
    return onnx_path


def _run(sess, arr):
    x = arr.transpose(2, 0, 1)[None]
    y = sess.run(None, {'input': x})[0][0]
    return y.transpose(1, 2, 0)


def upscale(img, blend=0.6, tile=256, folder=DEFAULT_DIR):
    """PIL image -> PIL image at 4x. `blend` mixes the network output with
    bicubic (1.0 = network only); 0.6 keeps texture and avoids the slightly
    waxy look the compact model has on its own. Processes in tiles with a
    16 px overlap so memory stays flat on large inputs."""
    global _sess
    if _sess is None:
        _sess = session(_model(folder))
    rgb = img.convert('RGB')
    a = np.asarray(rgb, np.float32) / 255.0
    h, w, _ = a.shape
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    pad = 16
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ya, xa = max(0, y0 - pad), max(0, x0 - pad)
            yb, xb = min(h, y0 + tile + pad), min(w, x0 + tile + pad)
            r = _run(_sess, np.ascontiguousarray(a[ya:yb, xa:xb]))
            cy, cx = (y0 - ya) * 4, (x0 - xa) * 4
            th, tw = (min(h, y0 + tile) - y0) * 4, (min(w, x0 + tile) - x0) * 4
            out[y0 * 4:y0 * 4 + th, x0 * 4:x0 * 4 + tw] = r[cy:cy + th, cx:cx + tw]
    net = Image.fromarray((out.clip(0, 1) * 255 + 0.5).astype(np.uint8))
    if blend >= 1:
        return net
    bic = rgb.resize(net.size, Image.BICUBIC)
    return Image.blend(bic, net, blend)
